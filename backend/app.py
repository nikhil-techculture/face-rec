import os
import uuid
import base64
import binascii
import aiofiles
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from typing import Optional

from face_engine import (
    register_face,
    match_face,
    match_two_faces,
    list_registered_labels,
    delete_label,
    get_reference_image_path,
)

load_dotenv()

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 10))

app = FastAPI(
    title="Face Recognition API",
    description="Register reference faces and match uploaded images against them.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def save_upload(file: UploadFile) -> Path:
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Use JPG, PNG, or WEBP.")

    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    async with aiofiles.open(dest, "wb") as f:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_SIZE_MB}MB allowed.")
        await f.write(content)
    return dest


async def save_base64_upload(image_base64: str) -> Path:
    payload = image_base64.strip()
    mime = ""
    if payload.startswith("data:") and "," in payload:
        header, payload = payload.split(",", 1)
        mime = header.split(";")[0].replace("data:", "").lower()

    ext_by_mime = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    ext = ext_by_mime.get(mime, ".jpg")

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported base64 image type. Use JPG, PNG, or WEBP.")

    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Invalid base64 image payload.")

    if len(raw) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_SIZE_MB}MB allowed.")

    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    async with aiofiles.open(dest, "wb") as f:
        await f.write(raw)
    return dest


async def save_image_input(image: Optional[UploadFile], image_base64: Optional[str]) -> Path:
    if image is not None:
        return await save_upload(image)
    if image_base64:
        return await save_base64_upload(image_base64)
    raise HTTPException(status_code=400, detail="Provide either 'image' file or 'imageBase64'.")


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Face Recognition API", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.post("/setup", tags=["Face Management"])
async def setup_face(
    label: str = Form(..., description="Unique identifier for this face (e.g. user ID or name)"),
    image: Optional[UploadFile] = File(None, description="Reference face image"),
    imageBase64: Optional[str] = Form(None, description="Reference face image as base64 string")
):
    """
    Register a reference face image under a label.
    This image will be used as the ground truth for future match requests.
    """
    path = await save_image_input(image, imageBase64)
    try:
        result = register_face(label, str(path))
    finally:
        path.unlink(missing_ok=True)

    if not result["success"]:
        raise HTTPException(status_code=422, detail=result["message"])

    return JSONResponse(status_code=201, content=result)


@app.post("/match", tags=["Face Matching"])
async def match_face_endpoint(
    label: str = Form(..., description="Label of the registered reference face"),
    image: Optional[UploadFile] = File(None, description="Image to verify against the reference"),
    imageBase64: Optional[str] = Form(None, description="Image to verify as base64 string"),
    tolerance: float = Form(0.5, description="Match tolerance (0.4=strict, 0.6=lenient)")
):
    """
    Compare an uploaded face image against a registered reference face.
    Returns match status and confidence score (0-100).
    """
    if not 0.1 <= tolerance <= 0.9:
        raise HTTPException(status_code=400, detail="Tolerance must be between 0.1 and 0.9")

    path = await save_image_input(image, imageBase64)
    try:
        result = match_face(str(path), label, tolerance)
    finally:
        path.unlink(missing_ok=True)

    if result.get("match") and result.get("matched_image_url"):
        result["matched_image_url"] = f"/api{result['matched_image_url']}"

    return JSONResponse(status_code=200, content=result)


@app.post("/match-direct", tags=["Face Matching"])
async def match_direct_endpoint(
    referenceImageBase64: str = Form(..., description="Reference image as base64 string"),
    image: Optional[UploadFile] = File(None, description="Image to verify against reference"),
    imageBase64: Optional[str] = Form(None, description="Image to verify as base64 string"),
    tolerance: float = Form(0.5, description="Match tolerance (0.4=strict, 0.6=lenient)")
):
    """
    Compare uploaded image with a provided reference base64 image.
    """
    if not 0.1 <= tolerance <= 0.9:
        raise HTTPException(status_code=400, detail="Tolerance must be between 0.1 and 0.9")

    reference_path = await save_base64_upload(referenceImageBase64)
    path = await save_image_input(image, imageBase64)
    try:
        result = match_two_faces(str(reference_path), str(path), tolerance)
    finally:
        reference_path.unlink(missing_ok=True)
        path.unlink(missing_ok=True)

    return JSONResponse(status_code=200, content=result)


@app.get("/labels", tags=["Face Management"])
def get_labels():
    """List all registered face labels."""
    labels = list_registered_labels()
    return {"labels": labels, "count": len(labels)}


@app.delete("/labels/{label}", tags=["Face Management"])
def remove_label(label: str):
    """Delete a registered face label and its encoding."""
    result = delete_label(label)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result["message"])
    return result


@app.get("/images/{label}", tags=["Face Management"])
def get_reference_image(label: str):
    """Return the stored reference image for a label."""
    path = get_reference_image_path(label)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Reference image not found for label '{label}'.")
    return FileResponse(path)

import os
import uuid
import aiofiles
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from face_engine import register_face, match_face, list_registered_labels, delete_label

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


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Face Recognition API", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.post("/setup", tags=["Face Management"])
async def setup_face(
    label: str = Form(..., description="Unique identifier for this face (e.g. user ID or name)"),
    image: UploadFile = File(..., description="Reference face image")
):
    """
    Register a reference face image under a label.
    This image will be used as the ground truth for future match requests.
    """
    path = await save_upload(image)
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
    image: UploadFile = File(..., description="Image to verify against the reference"),
    tolerance: float = Form(0.5, description="Match tolerance (0.4=strict, 0.6=lenient)")
):
    """
    Compare an uploaded face image against a registered reference face.
    Returns match status and confidence score (0-100).
    """
    if not 0.1 <= tolerance <= 0.9:
        raise HTTPException(status_code=400, detail="Tolerance must be between 0.1 and 0.9")

    path = await save_upload(image)
    try:
        result = match_face(str(path), label, tolerance)
    finally:
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

import os
import uuid
import base64
import binascii
import aiofiles
from io import BytesIO
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from typing import Optional, Tuple
from PIL import Image, UnidentifiedImageError

from face_engine import (
    register_face,
    match_face,
    match_two_faces,
    list_registered_labels,
    delete_label,
    get_reference_image_path,
    validate_signature,
    process_liveness_frame,
)
from doc_parser import extract_text, name_found_in_document
from cheque_parser import read_cheque
from statement_parser import read_bank_statement

load_dotenv()

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_DOC_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 10))


def detect_image_extension(raw: bytes, fallback: str = ".jpg") -> str:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if raw.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return ".webp"
    return fallback


def normalize_image_bytes(raw: bytes, ext_hint: str) -> Tuple[bytes, str]:
    ext = (ext_hint or detect_image_extension(raw)).lower()
    if ext == ".jpeg":
        ext = ".jpg"
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Use JPG, PNG, or WEBP.")

    try:
        with Image.open(BytesIO(raw)) as image:
            has_alpha = "A" in image.getbands() or (image.mode == "P" and "transparency" in image.info)
            if has_alpha:
                rgba = image.convert("RGBA")
                white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                white_bg.alpha_composite(rgba)
                output = BytesIO()
                white_bg.convert("RGB").save(output, format="PNG")
                return output.getvalue(), ".png"

            if image.mode not in ("RGB", "L"):
                converted = image.convert("RGB")
                output = BytesIO()
                save_format = "PNG" if ext == ".png" else "WEBP" if ext == ".webp" else "JPEG"
                converted.save(output, format=save_format)
                return output.getvalue(), ext
    except UnidentifiedImageError:
        pass

    return raw, ext

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
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_SIZE_MB}MB allowed.")

    normalized_content, normalized_ext = normalize_image_bytes(content, ext)
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{normalized_ext}"
    async with aiofiles.open(dest, "wb") as f:
        await f.write(normalized_content)
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

    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Invalid base64 image payload.")

    if len(raw) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_SIZE_MB}MB allowed.")

    normalized_content, normalized_ext = normalize_image_bytes(raw, detect_image_extension(raw, ext))

    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{normalized_ext}"
    async with aiofiles.open(dest, "wb") as f:
        await f.write(normalized_content)
    return dest


async def save_image_input(image: Optional[UploadFile], image_base64: Optional[str]) -> Path:
    if image is not None:
        return await save_upload(image)
    if image_base64:
        return await save_base64_upload(image_base64)
    raise HTTPException(status_code=400, detail="Provide either 'image' file or 'imageBase64'.")


def file_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


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
        raise HTTPException(status_code=422, detail={
            "error_code": result.get("error_code", "UNKNOWN"),
            "message": result["message"]
        })

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

    if result.get("matched_image_url"):
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


async def save_document(file: UploadFile) -> Path:
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_DOC_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported document type '{ext}'. Use PDF, JPG, PNG, WEBP, BMP, or TIFF."
        )
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Document too large. Max {MAX_FILE_SIZE_MB}MB allowed.")
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    async with aiofiles.open(dest, "wb") as f:
        await f.write(content)
    return dest

async def save_document_base64(doc_base64: str) -> Path:
    payload = doc_base64.strip()
    mime = ""
    if payload.startswith("data:") and "," in payload:
        header, payload = payload.split(",", 1)
        mime = header.split(";")[0].replace("data:", "").lower()

    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Invalid base64 payload.")

    if len(raw) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_SIZE_MB}MB allowed.")

    if raw.startswith(b"%PDF") or mime == "application/pdf":
        ext = ".pdf"
        content = raw
    else:
        ext_hint = ".png" if mime == "image/png" else ".webp" if mime == "image/webp" else ".jpg"
        content, ext = normalize_image_bytes(raw, detect_image_extension(raw, ext_hint))

    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    async with aiofiles.open(dest, "wb") as f:
        await f.write(content)
    return dest


async def save_document_input(file: Optional[UploadFile], base64_str: Optional[str]) -> Path:
    if file is not None:
        return await save_document(file)
    if base64_str:
        return await save_document_base64(base64_str)
    raise HTTPException(status_code=400, detail="Provide either document file or base64 payload.")


@app.post("/read-cheque", tags=["Cheque Reader"])
@app.post("/api/read-cheque", tags=["Cheque Reader"])
async def read_cheque_endpoint(
    cheque: Optional[UploadFile] = File(None, description="Cheque document or image (PDF, JPG, PNG, WEBP, BMP, TIFF)"),
    document: Optional[UploadFile] = File(None, description="Alternative field name for cheque file"),
    image: Optional[UploadFile] = File(None, description="Alternative field name for cheque image"),
    chequeBase64: Optional[str] = Form(None, description="Cheque as base64 string or data URL"),
    imageBase64: Optional[str] = Form(None, description="Alternative base64 field name")
):
    """
    ## Read Cheque and Extract Information
    
    Extracts key information from a cheque image or PDF:
    - **account_number**: Bank account number (9-18 digits)
    - **ifsc_code**: 11-digit IFSC code (with automatic OCR validation/correction)
    - **bank_name**: Identified bank name
    - **branch_details**: Object with branch_name and address
    - **address**: Complete branch address
    - **account_holder_name**: Name of the drawer/account holder
    - **signatory_name**: Name of the authorized signatory (if corporate account)
    - **cheque_number**: 6-digit cheque number (if present in MICR line)
    - **micr_code**: 9-digit MICR code (if present)
    - **confidence**: Overall confidence score (0-1.0)
    - **confidence_scores**: Individual field confidences
    - **raw_text**: Full OCR text
    """
    file_input = cheque or document or image
    b64_input = chequeBase64 or imageBase64
    
    if file_input is None and not b64_input:
        raise HTTPException(status_code=400, detail="Provide 'cheque' file or 'chequeBase64'.")

    path = await save_document_input(file_input, b64_input)
    try:
        result = read_cheque(str(path))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse cheque: {str(e)}")
    finally:
        path.unlink(missing_ok=True)

    if not result.get("success", True):
        return JSONResponse(status_code=400, content=result)

    return JSONResponse(status_code=200, content=result)


@app.post("/read-statement", tags=["Bank Statement Reader"])
@app.post("/api/read-statement", tags=["Bank Statement Reader"])
async def read_statement_endpoint(
    statement: Optional[UploadFile] = File(None, description="Bank statement PDF or image"),
    document: Optional[UploadFile] = File(None, description="Alternative field name for statement file"),
    file: Optional[UploadFile] = File(None, description="Alternative field name for statement file"),
    statementBase64: Optional[str] = Form(None, description="Bank statement as base64 string or data URL"),
    documentBase64: Optional[str] = Form(None, description="Alternative base64 field name"),
    minMonths: int = Form(6, description="Required minimum statement duration in months (default: 6)")
):
    """
    ## Read Bank Statement and Extract Information
    
    Extracts key information from an uploaded bank statement PDF or image:
    - **bank_name**: Name of the bank (e.g. Canara Bank, HDFC Bank)
    - **account_number**: Account number (indicates if masked e.g. XXXXXXXXX7009)
    - **account_holder_name**: Name of the primary account holder
    - **ifsc_code**: Bank branch IFSC code
    - **branch_name**: Branch location name
    - **branch_code**: Branch code
    - **branch_address**: Branch physical address
    - **statement_period**: Start date, end date, days & months covered
    - **has_6_months**: Boolean flag indicating if statement satisfies >= 6 months (or custom minMonths)
    - **balances**: Opening and closing balances
    - **transactions_detected**: Estimated number of transaction rows
    """
    file_input = statement or document or file
    b64_input = statementBase64 or documentBase64

    if file_input is None and not b64_input:
        raise HTTPException(status_code=400, detail="Provide 'statement' file or 'statementBase64'.")

    path = await save_document_input(file_input, b64_input)
    try:
        result = read_bank_statement(str(path), min_months=minMonths)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse bank statement: {str(e)}")
    finally:
        path.unlink(missing_ok=True)

    if not result.get("success", True):
        return JSONResponse(status_code=400, content=result)

    return JSONResponse(status_code=200, content=result)



@app.post("/verify-document", tags=["Document Verification"])
async def verify_document_endpoint(
    username: str = Form(..., description="Full name to search inside the document"),
    document: UploadFile = File(..., description="Bank statement PDF or image (JPG/PNG/WEBP/BMP/TIFF)")
):
    """
    ## Verify name in a document (bank statement)

    Extracts text from the uploaded document and checks if `username` appears in it.

    - **PDF** â€” text extracted directly (no OCR needed)
    - **Image** â€” OCR via Tesseract (must be installed for image docs)

    **Response:**
    - `verified` â€” true if name found in document
    - `matched_text` â€” the part of the name that was matched
    - `doc_type` â€” `pdf` or `image`
    """
    doc_path = await save_document(document)
    try:
        doc_text, doc_type = extract_text(str(doc_path))
        name_result = name_found_in_document(username, doc_text)
    finally:
        doc_path.unlink(missing_ok=True)

    return JSONResponse(status_code=200, content={
        "verified": name_result["found"],
        "username": username,
        "matched_text": name_result.get("matched_text"),
        "match_type": name_result.get("match_type"),
        "doc_type": doc_type,
        "message": name_result["message"]
    })


@app.post("/validate-signature", tags=["Signature Validation"])
async def validate_signature_endpoint(
    image: Optional[UploadFile] = File(None, description="Signature image to validate"),
    imageBase64: Optional[str] = Form(None, description="Signature image as base64 string")
):
    """
    Validate if an image contains a proper signature.
    Detects and rejects:
    - Simple lines (starting/test lines)
    - Random curly/scribble lines
    - Empty or insufficient signatures
    
    Returns validation result with confidence score (0-100).
    """
    path = await save_image_input(image, imageBase64)
    try:
        result = validate_signature(str(path))
        result["normalized_base64"] = file_to_base64(path)
    finally:
        path.unlink(missing_ok=True)

    return JSONResponse(status_code=200, content=result)

@app.get("/proxy-image", tags=["Utility"])
def proxy_image(url: str):
    """Fetch an image from an external URL to bypass CORS on the frontend."""
    import requests
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Determine mime type from headers or default to jpeg
        content_type = response.headers.get('content-type', 'image/jpeg')
        b64_data = base64.b64encode(response.content).decode("ascii")
        
        return JSONResponse(status_code=200, content={
            "success": True, 
            "base64": f"data:{content_type};base64,{b64_data}"
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")

# /analyze-frame endpoint removed. Liveness processing now handled in frontend.

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 1001))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)

import face_recognition
import numpy as np
import pickle
import shutil
import cv2
from pathlib import Path
from typing import Optional
from insightface.app import FaceAnalysis
import onnxruntime as ort

# ── InsightFace global model initialization ──────────────────────────────────
face_app = FaceAnalysis(providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

MODELS_DIR = Path(__file__).parent / "models"
ENCODINGS_FILE = MODELS_DIR / "encodings.pkl"
REFERENCE_IMAGES_DIR = MODELS_DIR / "reference_images"

# ── Liveness Model global initialization ─────────────────────────────────────
LIVENESS_MODEL_PATH = MODELS_DIR / "anti_spoof_models" / "MiniFASNetV2.onnx"
try:
    liveness_session = ort.InferenceSession(str(LIVENESS_MODEL_PATH), providers=['CPUExecutionProvider'])
except Exception as e:
    print(f"Warning: Could not load liveness model: {e}")
    liveness_session = None


def get_faces(image_path: str):
    """Read image and detect all faces using InsightFace RetinaFace."""
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("Unable to read image")
    faces = face_app.get(image)
    return image, faces


def _load_encodings() -> dict:
    if ENCODINGS_FILE.exists():
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def _save_encodings(data: dict):
    MODELS_DIR.mkdir(exist_ok=True)
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)


def _record_from_value(value):
    # Backward-compatible shape: old data stores just encoding list.
    if isinstance(value, dict):
        return {
            "encoding": value.get("encoding"),
            "reference_image": value.get("reference_image")
        }
    return {"encoding": value, "reference_image": None}


def validate_face_pose(face) -> bool:
    """
    Validate that the face is looking roughly straight at the camera.
    Uses InsightFace keypoints (left_eye, right_eye, nose, mouth_left, mouth_right).
    Returns True if pose is acceptable for KYC.
    """
    landmarks = face.kps  # shape (5, 2)

    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]

    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    horizontal_offset = abs(nose[0] - eye_center_x)
    eye_distance = abs(right_eye[0] - left_eye[0])

    if eye_distance < 1:
        return False

    yaw_ratio = horizontal_offset / eye_distance
    return yaw_ratio < 0.18


def get_crop_box(bbox, w, h, scale=2.7):
    """Calculate crop box with a scale factor for liveness detection"""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = x2 - x1, y2 - y1
    size = int(max(bw, bh) * scale)
    nx1 = max(0, int(cx - size / 2))
    ny1 = max(0, int(cy - size / 2))
    nx2 = min(w, int(cx + size / 2))
    ny2 = min(h, int(cy + size / 2))
    return nx1, ny1, nx2, ny2

def check_liveness(image_path: str) -> dict:
    """
    Passive Liveness detection using MiniFASNet ONNX.
    Returns whether the face is real (3D) or a spoof (2D photo/screen).
    """
    if liveness_session is None:
        return {"live": True, "score": 1.0}

    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
             return {"live": False, "score": 0.0, "message": "Could not read image for liveness."}
             
        h, w = img_bgr.shape[:2]
        
        _, faces = get_faces(image_path)
        if not faces:
             return {"live": False, "score": 0.0, "message": "No face detected for liveness."}
             
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        x1, y1, x2, y2 = get_crop_box(largest_face.bbox, w, h, scale=2.7)
        face_crop = img_bgr[y1:y2, x1:x2]
        
        if face_crop.size == 0:
             return {"live": False, "score": 0.0, "message": "Invalid face crop."}
             
        # Preprocess for MiniFASNet (80x80)
        resized = cv2.resize(face_crop, (80, 80))
        # Convert to RGB, scale to [0,1], CHW format
        img_np = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        input_data = np.transpose(img_np, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)
        
        input_name = liveness_session.get_inputs()[0].name
        out = liveness_session.run(None, {input_name: input_data})[0]
        
        # Softmax
        out_exp = np.exp(out[0] - np.max(out[0]))
        probs = out_exp / np.sum(out_exp)
        
        # For MiniFASNet, class 1 is typically real face. Class 0 & 2 are spoofs.
        score = float(probs[1]) if len(probs) == 3 else float(probs[0])
        live = score > 0.8

        return {
            "live": bool(live),
            "score": round(score, 4),
            "message": "Liveness check passed" if live else "Liveness check failed (spoof detected)"
        }
        
    except Exception as e:
        print(f"Liveness check error: {e}")
        return {"live": False, "score": 0.0, "message": str(e)}


def validate_face_image(image_path: str) -> dict:
    """
    Validate a face image before encoding/matching using InsightFace.

    Rules:
    1. Image must be readable
    2. At least one face must be detected
    3. Exactly one face must be visible anywhere in the image
    4. The detected face should not be too small or too close
    5. Image should not be too blurry
    6. Face pose must be roughly frontal (KYC requirement)

    Note:
    - Background objects/clutter are allowed
    - A second person/face in the background is rejected strictly

    Returns:
        { "valid": bool, "error_code": str|None, "message": str, "face_count": int }
    """
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return {"valid": False, "error_code": "UNREADABLE", "face_count": 0,
                    "message": "Image could not be read. Ensure it is a valid JPG/PNG/WEBP file."}

        h, w = img_bgr.shape[:2]
        image_area = h * w

        # ── Blur check (Laplacian variance) ──────────────────────────────────
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < 20.0:
            return {"valid": False, "error_code": "TOO_BLURRY", "face_count": 0,
                    "message": f"Image is too blurry (score: {blur_score:.1f}). Use a clearer, well-lit photo."}

        # ── Face detection (InsightFace RetinaFace) ──────────────────────────
        _, faces = get_faces(image_path)

        if len(faces) == 0:
            return {"valid": False, "error_code": "NO_FACE", "face_count": 0,
                    "message": "No face detected in the image. Ensure the face is clearly visible and well-lit."}

        face_count = len(faces)

        if face_count > 1:
            return {
                "valid": False,
                "error_code": "MULTIPLE_FACES",
                "face_count": face_count,
                "message": f"{face_count} faces detected. Background objects are allowed, but a second visible person is not allowed."
            }

        # ── Face size check ───────────────────────────────────────────────────
        largest_face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        x1, y1, x2, y2 = largest_face.bbox.astype(int)
        face_w = x2 - x1
        face_h = y2 - y1
        face_area = face_w * face_h
        face_ratio = face_area / max(1, image_area)

        if face_ratio < 0.015:
            return {"valid": False, "error_code": "FACE_TOO_SMALL", "face_count": 1,
                    "message": "Face is too small or too far from the camera. Move closer and retake the photo."}

        if face_ratio > 0.95:
            return {"valid": False, "error_code": "FACE_TOO_CLOSE", "face_count": 1,
                    "message": "Face is too close to the camera. Move back slightly and retake the photo."}

        # ── Pose validation (KYC: must look at camera) ───────────────────────
        if not validate_face_pose(largest_face):
            return {"valid": False, "error_code": "BAD_POSE", "face_count": 1,
                    "message": "Please look straight at the camera."}

        # ── Liveness check (Spoof detection) ──────────────────────────────────
        liveness = check_liveness(image_path)
        if not liveness.get("live", True):
            return {"valid": False, "error_code": "LIVENESS_FAILED", "face_count": 1,
                    "message": "Liveness verification failed (spoof/photo detected)."}

        # Background/plainness is not enforced. Objects are allowed,
        # but there must be only one detectable human face in the frame.
        return {
            "valid": True,
            "error_code": None,
            "face_count": 1,
            "message": "Face image is valid. Any background is allowed, but only one visible person/face is permitted."
        }

    except Exception as e:
        return {"valid": False, "error_code": "PROCESSING_ERROR", "face_count": 0,
                "message": f"Error processing image: {str(e)}"}


def encode_image(image_path: str) -> Optional[list]:
    """
    Generate ArcFace 512-d embedding using InsightFace.
    Returns list[float] or None if no face found.
    """
    try:
        image, faces = get_faces(image_path)

        if len(faces) == 0:
            return None

        largest_face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        embedding = largest_face.embedding
        return embedding.tolist()

    except Exception:
        return None


def register_face(label: str, image_path: str) -> dict:
    """Register a reference face under a given label."""
    validation = validate_face_image(image_path)
    if not validation["valid"]:
        return {"success": False, "error_code": validation["error_code"], "message": validation["message"]}

    encoding = encode_image(image_path)
    if encoding is None:
        return {"success": False, "error_code": "ENCODING_FAILED", "message": "Face detected but encoding failed. Try a clearer photo."}

    data = _load_encodings()
    REFERENCE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(image_path).suffix.lower() or ".jpg"
    stored_file_name = f"{label}{ext}"
    stored_image_path = REFERENCE_IMAGES_DIR / stored_file_name
    shutil.copy2(image_path, stored_image_path)

    data[label] = {
        "encoding": encoding,
        "reference_image": stored_file_name
    }
    _save_encodings(data)
    return {"success": True, "message": f"Face registered successfully for label '{label}'."}


def match_face(image_path: str, label: str, tolerance: float = 0.75) -> dict:
    """
    Compare uploaded image against the stored encoding for a label.
    Returns match result with confidence score.
    tolerance: cosine similarity threshold (0.75 default, higher = stricter)
    """
    data = _load_encodings()

    if label not in data:
        return {
            "match": False,
            "confidence": 0.0,
            "message": f"No registered face found for label '{label}'. Please setup first."
        }

    validation = validate_face_image(image_path)
    if not validation["valid"]:
        return {"match": False, "confidence": 0.0,
                "error_code": validation["error_code"], "message": validation["message"]}

    unknown_encoding = encode_image(image_path)
    if unknown_encoding is None:
        return {"match": False, "confidence": 0.0,
                "error_code": "ENCODING_FAILED", "message": "Face detected but encoding failed. Try a clearer photo."}

    record = _record_from_value(data[label])
    known_encoding = np.array(record["encoding"])
    unknown_np = np.array(unknown_encoding)

    # Cosine similarity matching (ArcFace embeddings are normalized)
    similarity = float(np.dot(known_encoding, unknown_np) / (
        np.linalg.norm(known_encoding) * np.linalg.norm(unknown_np)
    ))

    is_match = similarity >= tolerance

    result = {
        "match": bool(is_match),
        "confidence": round(similarity * 100, 2),
        "similarity": round(similarity, 4),
        "message": "Face matched successfully." if is_match else "Face does not match."
    }
    if is_match:
        result["matched_image_url"] = f"/images/{label}"
    return result


def match_two_faces(reference_image_path: str, image_path: str, tolerance: float = 0.75) -> dict:
    """
    Compare a probe image against a reference image directly.
    Returns match result with confidence score using cosine similarity.
    """
    ref_validation = validate_face_image(reference_image_path)
    if not ref_validation["valid"]:
        return {"match": False, "confidence": 0.0,
                "error_code": ref_validation["error_code"],
                "message": f"Reference image invalid: {ref_validation['message']}"}

    probe_validation = validate_face_image(image_path)
    if not probe_validation["valid"]:
        return {"match": False, "confidence": 0.0,
                "error_code": probe_validation["error_code"],
                "message": f"Uploaded image invalid: {probe_validation['message']}"}

    reference_encoding = encode_image(reference_image_path)
    if reference_encoding is None:
        return {"match": False, "confidence": 0.0,
                "error_code": "ENCODING_FAILED", "message": "Reference face detected but encoding failed."}

    unknown_encoding = encode_image(image_path)
    if unknown_encoding is None:
        return {"match": False, "confidence": 0.0,
                "error_code": "ENCODING_FAILED", "message": "Uploaded face detected but encoding failed."}

    ref = np.array(reference_encoding)
    probe = np.array(unknown_encoding)

    similarity = float(np.dot(ref, probe) / (
        np.linalg.norm(ref) * np.linalg.norm(probe)
    ))

    is_match = similarity >= tolerance

    return {
        "match": bool(is_match),
        "confidence": round(similarity * 100, 2),
        "similarity": round(similarity, 4),
        "message": "Face matched successfully." if is_match else "Face does not match."
    }


def list_registered_labels() -> list:
    """Return all registered face labels."""
    data = _load_encodings()
    return list(data.keys())


def delete_label(label: str) -> dict:
    """Remove a registered face label."""
    data = _load_encodings()
    if label not in data:
        return {"success": False, "message": f"Label '{label}' not found."}

    record = _record_from_value(data[label])
    ref_file = record.get("reference_image")
    if ref_file:
        (REFERENCE_IMAGES_DIR / ref_file).unlink(missing_ok=True)

    del data[label]
    _save_encodings(data)
    return {"success": True, "message": f"Label '{label}' deleted."}


def get_reference_image_path(label: str) -> Optional[Path]:
    data = _load_encodings()
    if label not in data:
        return None

    record = _record_from_value(data[label])
    ref_file = record.get("reference_image")
    if not ref_file:
        return None

    ref_path = REFERENCE_IMAGES_DIR / ref_file
    if not ref_path.exists():
        return None
    return ref_path


def validate_signature(image_path: str) -> dict:
    """
    Validate if an image contains a proper signature.
    Checks and verifies:
    1. Rejects uploaded human photos/faces.
    2. Signature should not be too simple like a dot or a simple stroke.
    3. Changes background color to plain white.

    Returns validation result with confidence score (0-100).
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return {
                "valid": False,
                "confidence": 0.0,
                "message": "Could not read image file."
            }

        # Face detection check to reject human photos uploaded as signatures
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if face_recognition.face_locations(rgb_img, model="hog"):
            return {
                "valid": False,
                "confidence": 5.0,
                "message": "Human face/photo detected. Upload only signature on plain white background."
            }

        h, w = img.shape[:2]
        image_area = float(max(1, h * w))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Identify ink (dark strokes) vs background
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dark_mask = (gray < 200).astype(np.uint8) * 255
        ink_mask = cv2.bitwise_or(binary_otsu, dark_mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 2. Change background color to pure white (255, 255, 255) only if background is not already white
        non_ink_mask = cleaned == 0
        if np.count_nonzero(non_ink_mask) > 0:
            bg_white_ratio = float(np.count_nonzero(gray[non_ink_mask] >= 230)) / float(np.count_nonzero(non_ink_mask))
        else:
            bg_white_ratio = 1.0

        if bg_white_ratio < 0.85:
            white_bg = np.full_like(img, 255)
            white_bg[cleaned > 0] = img[cleaned > 0]
            cv2.imwrite(image_path, white_bg)

        # 3. Analyze ink pixels
        ink_pixels = int(np.count_nonzero(cleaned > 0))
        ink_ratio = float(ink_pixels) / image_area

        if ink_pixels < 25 or ink_ratio < 0.0003:
            return {
                "valid": False,
                "confidence": 0.0,
                "message": "Invalid signature: image is empty or no signature strokes detected.",
                "metrics": {
                    "ink_pixels": ink_pixels,
                    "ink_ratio": round(ink_ratio, 5)
                }
            }

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) >= 10]

        if not valid_contours:
            return {
                "valid": False,
                "confidence": 0.0,
                "message": "Invalid signature: signature is too simple (looks like a dot or tiny spot).",
                "metrics": {
                    "ink_pixels": ink_pixels,
                    "ink_ratio": round(ink_ratio, 5)
                }
            }

        all_points = np.vstack(valid_contours)
        x, y, bw, bh = cv2.boundingRect(all_points)
        aspect_ratio = float(bw) / float(max(1, bh))
        bbox_area = float(bw * bh)

        main_contour = max(valid_contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(main_contour))
        arc_length = cv2.arcLength(main_contour, False)
        approx = cv2.approxPolyDP(main_contour, 0.015 * arc_length, False)
        vertices = len(approx)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        components_count = max(0, num_labels - 1)

        lines = cv2.HoughLinesP(
            cleaned,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=max(30, int(w * 0.25)),
            maxLineGap=10,
        )
        long_lines = 0 if lines is None else len(lines)

        # Check if signature is too simple (dot or simple stroke)
        is_dot = (
            bw < 18 and bh < 18
        ) or (
            bbox_area < 250 and contour_area < 200
        ) or (
            ink_pixels < 60
        )

        is_simple_stroke = (
            (vertices <= 8 and components_count <= 10)
            or (long_lines >= 1 and vertices <= 10 and components_count <= 10)
            or (aspect_ratio > 8.0 and vertices <= 8)
            or (aspect_ratio < 0.2 and vertices <= 8)
        )

        if is_dot:
            return {
                "valid": False,
                "confidence": 10.0,
                "message": "Invalid signature: signature is too simple (looks like a dot or tiny spot).",
                "metrics": {
                    "bounding_box": [bw, bh],
                    "ink_pixels": ink_pixels,
                    "vertices": vertices,
                    "components": components_count
                }
            }

        if is_simple_stroke:
            return {
                "valid": False,
                "confidence": 15.0,
                "message": "Invalid signature: signature is too simple (looks like a simple stroke or line).",
                "metrics": {
                    "aspect_ratio": round(aspect_ratio, 3),
                    "vertices": vertices,
                    "long_lines": long_lines,
                    "components": components_count
                }
            }

        complexity_score = min(100.0, max(50.0, (vertices * 2.5) + (components_count * 1.2)))
        confidence = round(complexity_score, 2)

        return {
            "valid": True,
            "confidence": confidence,
            "message": "Valid signature detected.",
            "metrics": {
                "ink_pixels": ink_pixels,
                "ink_ratio": round(ink_ratio, 5),
                "vertices": int(vertices),
                "components": int(components_count),
                "aspect_ratio": round(aspect_ratio, 3),
                "bounding_box": [int(bw), int(bh)],
                "background": "white"
            }
        }

    except Exception as e:
        return {
            "valid": False,
            "confidence": 0.0,
            "message": f"Error validating signature: {str(e)}"
        }

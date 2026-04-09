import face_recognition
import numpy as np
import pickle
import os
import shutil
import cv2
from pathlib import Path
from typing import Optional

MODELS_DIR = Path(__file__).parent / "models"
ENCODINGS_FILE = MODELS_DIR / "encodings.pkl"
REFERENCE_IMAGES_DIR = MODELS_DIR / "reference_images"


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


def validate_face_image(image_path: str) -> dict:
    """
    Validate a face image before encoding/matching.

    Checks (in order):
    1. Image readable
    2. At least one face detected
    3. Exactly one face (reject if multiple)
    4. Face is large enough (not too far / tiny)
    5. Image is not too blurry
    6. Face occupies reasonable portion of image (not just a tiny speck)

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
        if blur_score < 30.0:
            return {"valid": False, "error_code": "TOO_BLURRY", "face_count": 0,
                    "message": f"Image is too blurry (score: {blur_score:.1f}). Use a clearer, well-lit photo."}

        # ── Face detection ────────────────────────────────────────────────────
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model="hog")
        face_count = len(face_locations)

        if face_count == 0:
            return {"valid": False, "error_code": "NO_FACE", "face_count": 0,
                    "message": "No face detected in the image. Ensure the face is clearly visible and well-lit."}

        if face_count > 1:
            return {"valid": False, "error_code": "MULTIPLE_FACES", "face_count": face_count,
                    "message": f"{face_count} faces detected. Upload a photo with only one person."}

        # ── Face size check ───────────────────────────────────────────────────
        top, right, bottom, left = face_locations[0]
        face_h = bottom - top
        face_w = right - left
        face_area = face_h * face_w
        face_ratio = face_area / max(1, image_area)

        if face_ratio < 0.015:
            return {"valid": False, "error_code": "FACE_TOO_SMALL", "face_count": 1,
                    "message": "Face is too small or too far from the camera. Move closer and retake the photo."}

        if face_ratio > 0.95:
            return {"valid": False, "error_code": "FACE_TOO_CLOSE", "face_count": 1,
                    "message": "Face is too close to the camera. Move back slightly and retake the photo."}

        # ── Background check — detect cluttered / non-plain background ────────
        # Strategy: mask out the face region, analyze the remaining background.
        # A plain background has low color variance and low edge density.
        bg_mask = np.ones((h, w), dtype=bool)
        # Expand face bounding box by 20% as buffer
        pad_y = int(face_h * 0.2)
        pad_x = int(face_w * 0.2)
        y1 = max(0, top - pad_y)
        y2 = min(h, bottom + pad_y)
        x1 = max(0, left - pad_x)
        x2 = min(w, right + pad_x)
        bg_mask[y1:y2, x1:x2] = False  # exclude face area

        bg_pixels_rgb = img_bgr[:, :, ::-1][bg_mask]  # BGR→RGB, apply mask

        if len(bg_pixels_rgb) < 100:
            # Image is almost entirely face — that's fine
            return {"valid": True, "error_code": None, "face_count": 1, "message": "Face image is valid."}

        # Color variance of background (high = colorful/cluttered)
        bg_std = float(np.std(bg_pixels_rgb.astype(np.float32)))

        # Edge density of background (high = lots of objects/lines)
        bg_gray = gray.copy()
        bg_gray[~bg_mask] = 0
        edges = cv2.Canny(bg_gray, threshold1=50, threshold2=150)
        bg_edge_pixels = int(np.count_nonzero(edges[bg_mask]))
        bg_pixel_count = int(np.count_nonzero(bg_mask))
        edge_density = bg_edge_pixels / max(1, bg_pixel_count)

        # Saturation of background (high = colorful objects present)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        bg_sat = hsv[:, :, 1][bg_mask]
        sat_mean = float(np.mean(bg_sat))

        # Scoring system — at least 2 of 3 signals must fire to reject.
        # Thresholds tuned so plain/neutral backgrounds pass and
        # clearly cluttered backgrounds (objects, walls, furniture) fail.
        #
        # bg_std > 58      → high color variance in background
        # edge_density > 0.10 → dense edges (shelves, objects, text)
        # sat_mean > 120   → highly saturated/colorful background
        score = (
            (1 if bg_std > 58 else 0) +
            (1 if edge_density > 0.10 else 0) +
            (1 if sat_mean > 120 else 0)
        )
        cluttered = score >= 2

        if cluttered:
            return {
                "valid": False,
                "error_code": "CLUTTERED_BACKGROUND",
                "face_count": 1,
                "message": "Background is not plain. Remove objects behind you and use a plain/neutral background."
            }

        return {"valid": True, "error_code": None, "face_count": 1, "message": "Face image is valid."}

    except Exception as e:
        return {"valid": False, "error_code": "PROCESSING_ERROR", "face_count": 0,
                "message": f"Error processing image: {str(e)}"}


def encode_image(image_path: str) -> Optional[list]:
    """Load image, detect face, return 128-d encoding or None if no face found."""
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        return None
    return encodings[0].tolist()


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


def match_face(image_path: str, label: str, tolerance: float = 0.5) -> dict:
    """
    Compare uploaded image against the stored encoding for a label.
    Returns match result with confidence score.
    tolerance: lower = stricter (0.4 strict, 0.6 lenient)
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

    face_distance = face_recognition.face_distance([known_encoding], unknown_np)[0]
    is_match = bool(face_distance <= tolerance)

    # Convert distance to a 0-100 confidence score
    confidence = round((1 - float(face_distance)) * 100, 2)

    result = {
        "match": is_match,
        "confidence": confidence,
        "distance": round(float(face_distance), 4),
        "message": "Face matched successfully." if is_match else "Face does not match."
    }
    if is_match:
        result["matched_image_url"] = f"/images/{label}"
    return result


def match_two_faces(reference_image_path: str, image_path: str, tolerance: float = 0.5) -> dict:
    """
    Compare a probe image against a reference image directly.
    Returns match result with confidence score.
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

    reference_np = np.array(reference_encoding)
    unknown_np = np.array(unknown_encoding)

    face_distance = face_recognition.face_distance([reference_np], unknown_np)[0]
    is_match = bool(face_distance <= tolerance)
    confidence = round((1 - float(face_distance)) * 100, 2)

    return {
        "match": is_match,
        "confidence": confidence,
        "distance": round(float(face_distance), 4),
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
    Filters out:
    - Simple lines (starting/test lines)
    - Random curly/scribble lines
    - Insufficient or empty signatures
    
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
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Strict white-background requirement for plan-page signatures.
        white_mask = gray >= 235
        white_ratio = float(np.count_nonzero(white_mask)) / image_area
        if white_ratio < 0.78:
            return {
                "valid": False,
                "confidence": 8.0,
                "message": "Background must be plain white. Upload a clean white-background signature image.",
                "metrics": {
                    "white_ratio": round(white_ratio, 4)
                }
            }

        # Reject photo-like inputs (skin/background-rich images) before contour scoring.
        sat_ratio = float(np.count_nonzero(hsv[:, :, 1] > 55)) / image_area
        if sat_ratio > 0.16:
            return {
                "valid": False,
                "confidence": 10.0,
                "message": "Image looks like a photo/non-document input. Use black/blue signature on white background.",
                "metrics": {
                    "white_ratio": round(white_ratio, 4),
                    "saturation_ratio": round(sat_ratio, 4)
                }
            }

        # Build ink mask: dark pixels likely to be pen strokes.
        ink_mask = gray < 200
        ink_ratio = float(np.count_nonzero(ink_mask)) / image_area
        if ink_ratio < 0.0015:
            return {
                "valid": False,
                "confidence": 10.0,
                "message": "Signature area too small (empty or too light).",
                "metrics": {
                    "white_ratio": round(white_ratio, 4),
                    "ink_ratio": round(ink_ratio, 4)
                }
            }

        if ink_ratio > 0.12:
            return {
                "valid": False,
                "confidence": 12.0,
                "message": "Too much dark content detected. Image appears non-signature/photo-like.",
                "metrics": {
                    "white_ratio": round(white_ratio, 4),
                    "ink_ratio": round(ink_ratio, 4)
                }
            }

        binary = np.zeros_like(gray, dtype=np.uint8)
        binary[ink_mask] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        component_areas = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= 20:
                component_areas.append(area)

        if not component_areas:
            return {
                "valid": False,
                "confidence": 10.0,
                "message": "No meaningful signature strokes detected.",
                "metrics": {
                    "white_ratio": round(white_ratio, 4),
                    "ink_ratio": round(ink_ratio, 4)
                }
            }

        components_count = len(component_areas)
        if components_count > 60:
            return {
                "valid": False,
                "confidence": 15.0,
                "message": "Too many disconnected strokes. Signature appears scribbled/noisy.",
                "metrics": {
                    "white_ratio": round(white_ratio, 4),
                    "ink_ratio": round(ink_ratio, 4),
                    "components": components_count
                }
            }

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= 20]
        if not contours:
            return {
                "valid": False,
                "confidence": 10.0,
                "message": "No valid signature contour detected."
            }

        main_contour = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(main_contour))
        area_ratio = contour_area / image_area

        all_points = np.vstack(contours)
        x, y, bw, bh = cv2.boundingRect(all_points)
        aspect_ratio = float(bw) / float(max(1, bh))
        bbox_ratio = float(bw * bh) / image_area

        if aspect_ratio > 12.0 or aspect_ratio < 0.3:
            return {
                "valid": False,
                "confidence": 15.0,
                "message": "Signature appears to be a simple line (not a valid signature).",
                "metrics": {
                    "aspect_ratio": round(aspect_ratio, 3),
                    "bbox_ratio": round(bbox_ratio, 4)
                }
            }

        if bbox_ratio > 0.55:
            return {
                "valid": False,
                "confidence": 10.0,
                "message": "Signature occupies too much area. Image appears non-signature/photo-like.",
                "metrics": {
                    "bbox_ratio": round(bbox_ratio, 4),
                    "white_ratio": round(white_ratio, 4)
                }
            }

        hull = cv2.convexHull(main_contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = contour_area / hull_area if hull_area > 0 else 0.0

        arc_length = cv2.arcLength(main_contour, False)
        approx = cv2.approxPolyDP(main_contour, 0.015 * arc_length, False)
        vertices = len(approx)

        lines = cv2.HoughLinesP(
            cleaned,
            rho=1,
            theta=np.pi / 180,
            threshold=70,
            minLineLength=max(40, int(w * 0.35)),
            maxLineGap=8,
        )
        long_lines = 0 if lines is None else len(lines)
        if long_lines >= 1 and vertices <= 10 and components_count <= 4:
            return {
                "valid": False,
                "confidence": 15.0,
                "message": "Signature appears to be a starting/test line, not a proper signature.",
                "metrics": {
                    "long_lines": long_lines,
                    "vertices": vertices,
                    "components": components_count
                }
            }

        is_valid = (
            white_ratio >= 0.78
            and sat_ratio <= 0.16
            and 0.0015 <= ink_ratio <= 0.12
            and 0.30 <= aspect_ratio <= 12.0
            and bbox_ratio <= 0.55
            and solidity >= 0.18
            and vertices >= 10
            and components_count <= 60
            and area_ratio >= 0.0008
        )

        complexity_score = min(100.0, (vertices * 2.0) + (components_count * 0.9) + (solidity * 30.0))
        confidence = round(complexity_score, 2)

        result = {
            "valid": bool(is_valid),
            "confidence": confidence if is_valid else min(confidence, 50.0),
            "message": "Valid signature detected." if is_valid else "Invalid signature: signature pattern is not acceptable.",
            "metrics": {
                "white_ratio": round(white_ratio, 4),
                "saturation_ratio": round(sat_ratio, 4),
                "ink_ratio": round(ink_ratio, 4),
                "solidity": round(solidity, 4),
                "vertices": int(vertices),
                "components": int(components_count),
                "aspect_ratio": round(aspect_ratio, 3),
                "bbox_ratio": round(bbox_ratio, 4),
                "area_ratio": round(area_ratio, 5)
            }
        }

        if not is_valid:
            if white_ratio < 0.78:
                result["message"] = "Invalid signature: background is not plain white."
            elif sat_ratio > 0.16:
                result["message"] = "Invalid signature: image looks like photo/non-document input."
            elif vertices < 10:
                result["message"] = "Invalid signature: too simple or line-like strokes."
            elif solidity < 0.18:
                result["message"] = "Invalid signature: random/fragmented scribbles detected."

        return result

    except Exception as e:
        return {
            "valid": False,
            "confidence": 0.0,
            "message": f"Error validating signature: {str(e)}"
        }


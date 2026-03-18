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


def encode_image(image_path: str) -> Optional[list]:
    """Load image, detect face, return 128-d encoding or None if no face found."""
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        return None
    return encodings[0].tolist()


def register_face(label: str, image_path: str) -> dict:
    """Register a reference face under a given label."""
    encoding = encode_image(image_path)
    if encoding is None:
        return {"success": False, "message": "No face detected in the setup image."}

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

    unknown_encoding = encode_image(image_path)
    if unknown_encoding is None:
        return {
            "match": False,
            "confidence": 0.0,
            "message": "No face detected in the uploaded image."
        }

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
    reference_encoding = encode_image(reference_image_path)
    if reference_encoding is None:
        return {
            "match": False,
            "confidence": 0.0,
            "message": "No face detected in the reference image."
        }

    unknown_encoding = encode_image(image_path)
    if unknown_encoding is None:
        return {
            "match": False,
            "confidence": 0.0,
            "message": "No face detected in the uploaded image."
        }

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
        img = cv2.imread(image_path)
        if img is None:
            return {
                "valid": False,
                "confidence": 0.0,
                "message": "Could not read image file."
            }

        # Convert to grayscale and invert (white background, black strokes)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_inv = cv2.bitwise_not(gray)
        
        # Threshold to get binary image
        _, binary = cv2.threshold(gray_inv, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours (signature strokes)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                "valid": False,
                "confidence": 0.0,
                "message": "No signature strokes detected in image."
            }
        
        # Get main contour (largest by area)
        main_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(main_contour)
        
        # Check if contour is too small (empty signature)
        image_area = img.shape[0] * img.shape[1]
        area_ratio = contour_area / image_area if image_area > 0 else 0
        
        if area_ratio < 0.001:  # Less than 0.1% of image
            return {
                "valid": False,
                "confidence": 10.0,
                "message": "Signature area too small (empty or too light)."
            }
        
        # Get contour bounding box
        x, y, w, h = cv2.boundingRect(main_contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Check for line (aspect ratio too high = horizontal line)
        if aspect_ratio > 8.0 or aspect_ratio < 0.125:
            return {
                "valid": False,
                "confidence": 15.0,
                "message": "Signature appears to be a simple line (not a valid signature)."
            }
        
        # Fit ellipse to check straightness
        if len(main_contour) >= 5:
            ellipse = cv2.fitEllipse(main_contour)
            ellipse_area = np.pi * (ellipse[1][0] / 2) * (ellipse[1][1] / 2)
            solidity = contour_area / ellipse_area if ellipse_area > 0 else 0
        else:
            solidity = 0.0
        
        # Low solidity indicates scattered/random strokes
        if solidity < 0.3:
            return {
                "valid": False,
                "confidence": 20.0,
                "message": "Signature appears to be random scribbles (not a valid signature)."
            }
        
        # Calculate complexity: number of meaningful contours
        valid_contours = [c for c in contours if cv2.contourArea(c) > contour_area * 0.05]
        complexity = len(valid_contours)
        
        # Check for curvature / complexity
        arc_length = cv2.arcLength(main_contour, False)
        approx = cv2.approxPolyDP(main_contour, 0.02 * arc_length, False)
        vertices = len(approx)
        
        # High vertex count indicates curves and complexity
        complexity_score = min(100, (vertices - 4) * 2 + complexity * 5)
        
        # Valid signature criteria:
        # - Solidity > 0.4 (not scattered)
        # - Vertices > 10 (has curves/complexity)
        # - Area ratio > 0.01 (substantial)
        
        is_valid = (
            solidity > 0.35 and 
            vertices > 8 and 
            area_ratio > 0.005 and
            aspect_ratio < 6.0
        )
        
        confidence = min(100, complexity_score)
        
        if is_valid:
            return {
                "valid": True,
                "confidence": confidence,
                "message": "Valid signature detected.",
                "metrics": {
                    "solidity": round(float(solidity), 3),
                    "vertices": int(vertices),
                    "aspect_ratio": round(float(aspect_ratio), 2),
                    "area_ratio": round(float(area_ratio), 4)
                }
            }
        else:
            reason = ""
            if solidity <= 0.35:
                reason = "Low solidity - appears to be random strokes."
            elif vertices <= 8:
                reason = "Insufficient complexity - too simple."
            elif area_ratio <= 0.005:
                reason = "Signature area too small."
            elif aspect_ratio >= 6.0:
                reason = "Signature is too linear - appears to be a line."
            
            return {
                "valid": False,
                "confidence": min(confidence, 50),
                "message": f"Invalid signature: {reason}",
                "metrics": {
                    "solidity": round(float(solidity), 3),
                    "vertices": int(vertices),
                    "aspect_ratio": round(float(aspect_ratio), 2),
                    "area_ratio": round(float(area_ratio), 4)
                }
            }
    
    except Exception as e:
        return {
            "valid": False,
            "confidence": 0.0,
            "message": f"Error validating signature: {str(e)}"
        }


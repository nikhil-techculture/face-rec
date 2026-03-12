import face_recognition
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Optional

MODELS_DIR = Path(__file__).parent / "models"
ENCODINGS_FILE = MODELS_DIR / "encodings.pkl"


def _load_encodings() -> dict:
    if ENCODINGS_FILE.exists():
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def _save_encodings(data: dict):
    MODELS_DIR.mkdir(exist_ok=True)
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)


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
    data[label] = encoding
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

    known_encoding = np.array(data[label])
    unknown_np = np.array(unknown_encoding)

    face_distance = face_recognition.face_distance([known_encoding], unknown_np)[0]
    is_match = bool(face_distance <= tolerance)

    # Convert distance to a 0-100 confidence score
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
    del data[label]
    _save_encodings(data)
    return {"success": True, "message": f"Label '{label}' deleted."}

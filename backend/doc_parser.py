import re
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image

# ─── Salutations to strip from names ──────────────────────────────────────────
SALUTATIONS = {
    "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "miss", "dr", "dr.",
    "prof", "prof.", "shri", "smt", "kumari", "er", "adv", "rev",
    "capt", "col", "lt", "maj", "gen", "sgt", "cpl",
}

# ─── Keywords that indicate a joint/shared account line ───────────────────────
JOINT_ACCOUNT_PATTERNS = [
    r"\bjoint\s+account\b",
    r"\bjoint\s+holder\b",
    r"\b(and|&|or)\s+[a-z\s\.]+\s+(joint|co[- ]?applicant|co[- ]?holder)",
    r"\bco[- ]?applicant\b",
    r"\bco[- ]?holder\b",
    r"\bsecond\s+holder\b",
    r"\b2nd\s+holder\b",
]

# ─── Payment / transaction noise lines to skip ────────────────────────────────
PAYMENT_LINE_PATTERNS = [
    r"\b(neft|rtgs|imps|upi|nach|ecs|ach)\b",
    r"\b(credit|debit|transfer|payment|deposit|withdrawal|balance|txn|transaction)\b",
    r"\b\d{2}[/-]\d{2}[/-]\d{2,4}\b",   # dates like 01/01/2024
    r"\b\d{6,}\b",                        # long numbers (account/txn numbers)
    r"[₹$€£]\s*[\d,]+",                  # currency amounts
    r"\b[\d,]+\.\d{2}\b",                # decimal amounts like 1,234.56
]


# ─── Text Extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    pages_text = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages_text)


def extract_text_from_image(path: str) -> str:
    try:
        import pytesseract
        for p in [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]:
            if Path(p).exists():
                pytesseract.pytesseract.tesseract_cmd = p
                break
        return pytesseract.image_to_string(Image.open(path))
    except Exception:
        return ""


def extract_text(file_path: str) -> tuple[str, str]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path), "pdf"
    elif ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}:
        return extract_text_from_image(file_path), "image"
    return "", "unsupported"


# ─── Name Normalization ────────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """
    Lowercase, strip salutations (Mr./Mrs./Miss etc.),
    replace _ - . with space, collapse whitespace.
    """
    name = name.strip().lower()
    name = re.sub(r"[_\-]", " ", name)
    name = re.sub(r"\.", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    # Remove leading salutation word
    words = name.split()
    if words and words[0] in SALUTATIONS:
        words = words[1:]
    # Remove trailing salutation (rare but possible)
    if words and words[-1] in SALUTATIONS:
        words = words[:-1]

    return " ".join(words)


def get_name_words(name: str) -> list[str]:
    """Return meaningful words (length > 1) from a normalized name."""
    return [w for w in name.split() if len(w) > 1]


# ─── Document Line Analysis ────────────────────────────────────────────────────

def is_payment_line(line: str) -> bool:
    """Return True if the line looks like a transaction/payment row."""
    line_lower = line.lower()
    return any(re.search(p, line_lower) for p in PAYMENT_LINE_PATTERNS)


def is_joint_account_line(line: str) -> bool:
    """Return True if the line indicates a joint/co-holder account."""
    line_lower = line.lower()
    return any(re.search(p, line_lower) for p in JOINT_ACCOUNT_PATTERNS)


def extract_candidate_lines(doc_text: str) -> tuple[list[str], list[str]]:
    """
    Split document into:
    - candidate_lines: lines that could contain account holder name
    - rejected_lines:  payment rows and joint-account lines (ignored for matching)
    """
    all_lines = [l.strip() for l in doc_text.splitlines() if l.strip()]
    candidate_lines = []
    rejected_lines = []

    for line in all_lines:
        if is_joint_account_line(line):
            rejected_lines.append(f"[JOINT] {line}")
        elif is_payment_line(line):
            rejected_lines.append(f"[PAYMENT] {line}")
        else:
            candidate_lines.append(line)

    return candidate_lines, rejected_lines


# ─── Main Matching Function ────────────────────────────────────────────────────

def name_found_in_document(username: str, doc_text: str) -> dict:
    """
    Check if username appears in the document as the primary account holder.

    Rules:
    1. Strip salutations (Mr./Mrs./Miss/Dr. etc.) from both username and doc lines
    2. Skip payment/transaction lines entirely
    3. Skip joint-account lines (reject if name only appears there)
    4. Try full normalized name match first
    5. Fall back to all-words match (every word of name must appear in same line)
    6. Return detailed result with what was found/rejected
    """
    if not doc_text.strip():
        return {
            "found": False,
            "matched_text": None,
            "match_type": None,
            "rejected_lines": [],
            "message": "Could not extract text from document. Ensure it is a text-based PDF or install Tesseract for image OCR."
        }

    clean_username = normalize_name(username)
    name_words = get_name_words(clean_username)

    if not name_words:
        return {
            "found": False,
            "matched_text": None,
            "match_type": None,
            "rejected_lines": [],
            "message": "Username is too short or contains only salutations."
        }

    candidate_lines, rejected_lines = extract_candidate_lines(doc_text)

    # Normalize each candidate line for matching
    normalized_candidates = [normalize_name(line) for line in candidate_lines]

    # ── Pass 1: Full name exact match in any candidate line ──
    for norm_line in normalized_candidates:
        if clean_username in norm_line:
            return {
                "found": True,
                "matched_text": clean_username,
                "match_type": "full_name",
                "rejected_lines": rejected_lines,
                "message": f"Full name '{clean_username}' found in document."
            }

    # ── Pass 2: All words of name appear in the same candidate line ──
    for norm_line in normalized_candidates:
        line_words = set(norm_line.split())
        if all(w in line_words for w in name_words):
            return {
                "found": True,
                "matched_text": " ".join(name_words),
                "match_type": "all_words_in_line",
                "rejected_lines": rejected_lines,
                "message": f"All name parts {name_words} found together in a single line."
            }

    # ── Pass 3: All words appear anywhere in candidate text (not same line) ──
    full_candidate_text = " ".join(normalized_candidates)
    found_words = [w for w in name_words if w in full_candidate_text]
    missing_words = [w for w in name_words if w not in full_candidate_text]

    if not missing_words:
        return {
            "found": True,
            "matched_text": " ".join(found_words),
            "match_type": "all_words_scattered",
            "rejected_lines": rejected_lines,
            "message": f"All name parts {found_words} found across document (not on same line)."
        }

    # ── Check if name only appears in rejected (joint/payment) lines ──
    rejected_text = " ".join(
        normalize_name(l.split("] ", 1)[-1]) for l in rejected_lines
    )
    if all(w in rejected_text for w in name_words):
        return {
            "found": False,
            "matched_text": None,
            "match_type": "rejected_only",
            "rejected_lines": rejected_lines,
            "message": f"Name '{clean_username}' only found in joint-account or payment lines. Not accepted as primary holder."
        }

    if found_words:
        return {
            "found": False,
            "matched_text": " ".join(found_words),
            "match_type": "partial",
            "rejected_lines": rejected_lines,
            "message": f"Partial match only. Found: {found_words}, Missing: {missing_words}."
        }

    return {
        "found": False,
        "matched_text": None,
        "match_type": "no_match",
        "rejected_lines": rejected_lines,
        "message": f"Name '{clean_username}' not found in document."
    }

import re
import os
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from fastapi.responses import JSONResponse

# ─── Bank IFSC Prefix to Bank Name Mapping ─────────────────────────────────────
BANK_IFSC_MAP: Dict[str, str] = {
    "SBIN": "State Bank of India",
    "HDFC": "HDFC Bank",
    "ICIC": "ICICI Bank",
    "UTIB": "Axis Bank",
    "PUNB": "Punjab National Bank",
    "BARB": "Bank of Baroda",
    "CNRB": "Canara Bank",
    "KKBK": "Kotak Mahindra Bank",
    "UBIN": "Union Bank of India",
    "BKID": "Bank of India",
    "IDIB": "Indian Bank",
    "IOBA": "Indian Overseas Bank",
    "CBIN": "Central Bank of India",
    "UCBA": "UCO Bank",
    "PSIB": "Punjab & Sind Bank",
    "BDBL": "Bandhan Bank",
    "IDFB": "IDFC FIRST Bank",
    "FDRL": "Federal Bank",
    "INDB": "IndusInd Bank",
    "YESB": "Yes Bank",
    "RBLN": "RBL Bank",
    "KVBL": "Karur Vysya Bank",
    "SIBL": "South Indian Bank",
    "CSBK": "CSB Bank",
    "MAHB": "Bank of Maharashtra",
    "ANDB": "Andhra Bank",
    "CORP": "Corporation Bank",
    "ALLA": "Allahabad Bank",
    "SYNB": "Syndicate Bank",
    "ORBC": "Oriental Bank of Commerce",
    "VIJB": "Vijaya Bank",
    "DBSS": "DBS Bank India",
    "SCBL": "Standard Chartered Bank",
    "CITI": "Citibank",
    "HSBC": "HSBC Bank",
    "AIRP": "Airtel Payments Bank",
    "PYTM": "Paytm Payments Bank",
    "IPOS": "India Post Payments Bank",
    "JAKA": "Jammu & Kashmir Bank",
    "ESFB": "Equitas Small Finance Bank",
    "AUBL": "AU Small Finance Bank",
    "UTKS": "Utkarsh Small Finance Bank",
    "UJVN": "Ujjivan Small Finance Bank",
    "JSBP": "Janata Sahakari Bank",
    "SMCB": "Shivalik Small Finance Bank",
    "FINO": "Fino Payments Bank",
    "TMBL": "Tamilnad Mercantile Bank",
    "DCBL": "DCB Bank",
}

# ─── Common Indian Bank Names for Direct Text Search ──────────────────────────

KNOWN_BANK_NAMES: List[str] = [
    "STATE BANK OF INDIA", "HDFC BANK", "ICICI BANK", "AXIS BANK",
    "PUNJAB NATIONAL BANK", "BANK OF BARODA", "CANARA BANK",
    "KOTAK MAHINDRA BANK", "UNION BANK OF INDIA", "BANK OF INDIA",
    "INDIAN BANK", "INDIAN OVERSEAS BANK", "CENTRAL BANK OF INDIA",
    "UCO BANK", "PUNJAB & SIND BANK", "BANDHAN BANK", "IDFC FIRST BANK",
    "FEDERAL BANK", "INDUSIND BANK", "YES BANK", "RBL BANK",
    "KARUR VYSYA BANK", "SOUTH INDIAN BANK", "BANK OF MAHARASHTRA",
    "STANDARD CHARTERED BANK", "CITIBANK", "HSBC BANK",
    "AU SMALL FINANCE BANK", "EQUITAS SMALL FINANCE BANK",
    "AIRTEL PAYMENTS BANK", "PAYTM PAYMENTS BANK", "INDIA POST PAYMENTS BANK"
]

# ─── Common Indian Surnames & Middle Names for Unmerging OCR Text ────────────

COMMON_SURNAMES = [
    "MUKHERJEE", "CHATTERJEE", "BANERJEE", "CHOUDHARY", "CHOWDHURY", "MALHOTRA",
    "TRIPATHI", "AGGARWAL", "AGARWAL", "BHATTACHARYA", "CHAKRABORTY",
    "SHARMA", "VERMA", "GUPTA", "RAWAT", "SINGH", "PATEL", "MEHTA", "REDDY",
    "JOSHI", "TIWARI", "MISHRA", "PANDEY", "YADAV", "BANSAL", "MITTAL",
    "SAXENA", "DUTTA", "GOWDA", "MENON", "SHUKLA", "DUBEY", "CHAUBEY",
    "THAKUR", "KAPOOR", "BHATIA", "ARORA", "CHAWLA", "AHLUWALIA",
    "KUMARI", "KUMAR", "CHANDRA", "PRASAD", "PRAKASH", "SHEKHAR",
    "DEVI", "BHAI", "KANT", "KANTH", "NATH", "DUTT", "JEET", "PREET", "DEEP",
    "KHAN", "SHAH", "JAIN", "NAIR", "BOSE", "BHAT", "GARG", "KAUR", "LAL", "PAL",
    "RAO", "DAS", "SEN", "ROY", "RAM", "ALI"
]
# Sorted by length descending so longer suffixes (e.g. YADAV, KUMAR) match before short ones (e.g. RAM, ALI)
SORTED_SURNAMES = sorted(list(dict.fromkeys(COMMON_SURNAMES)), key=len, reverse=True)

DO_NOT_SPLIT_NAMES = {
    "ANJALI", "SONALI", "RUPALI", "DEEPALI", "CHAITALI", "MINAKSHI", "PARVATI", "SARASWATI"
}

# ─── Annotation / Title labels to discard from extracted values ───────────────

ANNOTATION_TITLES = {
    "ACCOUNT HOLDER NAME", "ACCOUNT HOLDER", "A/C HOLDER NAME", "A/C HOLDER",
    "BRANCH ADDRESS", "BANK NAME", "MICR CODE", "CHEQUE NUMBER",
    "IFSC CODE", "SIGNATURE", "PLEASE SIGN ABOVE THIS LINE", "PLEASE SIGN ABOVE",
    "PLEASE SIGN", "SIGN ABOVE THIS LINE", "SIGN HERE",
    "WEEKLY HOLIDAY ON SUNDAY", "WEEKLY HOLIDAY", "VALID FOR 3 MONTHS ONLY",
    "VALID FOR 3 MONTHS", "PAYABLE AT PAR", "CLEARING/TRANSFER", "OR BEARER",
    "NOT OVER RS", "RUPEES", "PAY", "ADDRESS", "BRANCH"
}

SIGNATURE_NOISE_TERMS = ANNOTATION_TITLES.union({
    "AUTHORISED SIGNATORY", "AUTHORIZED SIGNATORY", "AUTHORISED SIGNATORIES",
    "AUTHORIZED SIGNATORIES", "CHIEF EXECUTIVE OFFICER", "MANAGING DIRECTOR",
    "DIRECTOR", "PROPRIETOR", "PARTNER", "SECRETARY", "TREASURER", "CHAIRMAN",
    "CTS 2010", "CTS-2010", "BRANCH CODE", "CHEQUE NO", "CHEQUE ND", "CHEQUE", "CHK",
    "A/C NO", "AC NO"
})

class ChequeOCRBlock:
    def __init__(self, text: str, box: List[List[float]], confidence: float, img_w: int, img_h: int):
        self.text = text.strip()
        self.box = box
        self.confidence = float(confidence)

        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        self.xmin = min(xs)
        self.xmax = max(xs)
        self.ymin = min(ys)
        self.ymax = max(ys)

        self.width = max(1.0, self.xmax - self.xmin)
        self.height = max(1.0, self.ymax - self.ymin)
        self.cx = (self.xmin + self.xmax) / 2.0
        self.cy = (self.ymin + self.ymax) / 2.0

        # Normalized coordinates (0.0 to 1.0)
        self.norm_x = self.cx / max(1.0, img_w)
        self.norm_y = self.cy / max(1.0, img_h)
        self.norm_ymin = self.ymin / max(1.0, img_h)
        self.norm_ymax = self.ymax / max(1.0, img_h)
        self.norm_xmin = self.xmin / max(1.0, img_w)
        self.norm_xmax = self.xmax / max(1.0, img_w)

    def __repr__(self):
        return f"<Block '{self.text}' conf={self.confidence:.2f} y={self.norm_y:.2f} x={self.norm_x:.2f}>"

_rapid_ocr_engine = None

def get_ocr_engine():
    global _rapid_ocr_engine
    if _rapid_ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _rapid_ocr_engine = RapidOCR()
        except Exception as e:
            print(f"[ChequeParser] RapidOCR initialization warning: {e}")
            _rapid_ocr_engine = False
    return _rapid_ocr_engine

def load_cheque_image(file_path: str) -> Tuple[np.ndarray, str]:
    """Loads an image or renders the first page of a PDF into an RGB numpy ndarray."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        doc = fitz.open(file_path)
        if len(doc) == 0:
            raise ValueError("PDF document is empty.")

        page = doc[0]
        pix = page.get_pixmap(dpi=300)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        doc.close()

        if pix.n == 4:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

        return img_bgr, "pdf"    

    elif ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}:
        img = cv2.imread(file_path)
        if img is None:
            pil_img = Image.open(file_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img, "image"
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Expected PDF or image.")

def run_ocr(img_bgr: np.ndarray) -> List[ChequeOCRBlock]:
    """Runs RapidOCR and returns normalized ChequeOCRBlock items."""
    engine = get_ocr_engine()
    if not engine:
        raise RuntimeError("OCR engine not available. Please install rapidocr-onnxruntime.")

    h, w = img_bgr.shape[:2]
    result, _ = engine(img_bgr)

    blocks: List[ChequeOCRBlock] = []
    if not result:
        return blocks

    for item in result:
        box, text, conf = item
        if text and text.strip():
            blocks.append(ChequeOCRBlock(text=text, box=box, confidence=conf, img_w=w, img_h=h))

    return blocks

def get_orientation_score(blocks: List[ChequeOCRBlock]) -> int:
    """Calculates an orientation score to determine if a cheque is upright."""
    score = 0
    for b in blocks:
        t_up = b.text.upper()
        if re.search(r"[A-Z]{4}0[A-Z0-9]{6}", t_up) or any(bank in t_up for bank in KNOWN_BANK_NAMES):
            score += 15 if b.norm_y < 0.40 else -15

        if "PAY" in t_up:
            score += 10 if b.norm_y < 0.45 else -10

        if any(term in t_up for term in ["SIGN", "SIGNATORY", "ACCOUNT HOLDER", "PLEASE SIGN"]):
            score += 10 if b.norm_y > 0.50 else -10

        if re.search(r"\b[0-9]{9}\b", b.text) or "MICR" in t_up:
            score += 10 if b.norm_y > 0.65 else -10

    return score

def auto_orient_cheque(img_bgr: np.ndarray) -> Tuple[np.ndarray, List[ChequeOCRBlock]]:
    """Automatically detects cheque rotation (portrait vs landscape, upside down) and orients upright."""

    h, w = img_bgr.shape[:2]
    if h > w:
        # Cheque is scanned in portrait, test 90 CCW vs 90 CW
        ccw_img = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        blocks_ccw = run_ocr(ccw_img)
        score_ccw = get_orientation_score(blocks_ccw)

        cw_img = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
        blocks_cw = run_ocr(cw_img)
        score_cw = get_orientation_score(blocks_cw)

        if score_ccw >= score_cw:
            return ccw_img, blocks_ccw
        else:
            return cw_img, blocks_cw

    else:
        # Cheque is in landscape, check if it's upside down
        blocks = run_ocr(img_bgr)
        score = get_orientation_score(blocks)
        if score < -10:
            rot_180 = cv2.rotate(img_bgr, cv2.ROTATE_180)
            return rot_180, run_ocr(rot_180)

        return img_bgr, blocks

def check_image_quality(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates image quality with checkpoints:
    - Sharpness / Blur detection (Laplacian variance >= 40.0)
    - Brightness / Exposure check (Mean intensity 45.0 - 252.0)
    - Contrast check (Standard deviation >= 18.0)
    - Glare / Reflection check (Saturated pixels <= 45.0%)
    - Resolution check (min dimension >= 180, max dimension >= 500)
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Sharpness / Blur check (Laplacian variance)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharp_passed = lap_var >= 40.0
    sharp_status = "Good" if lap_var >= 80.0 else "Acceptable" if lap_var >= 40.0 else "Blurry"

    # 2. Brightness check (Mean intensity)
    mean_b = float(np.mean(gray))
    bright_passed = 45.0 <= mean_b <= 252.0
    bright_status = "Normal" if bright_passed else "Too Dark" if mean_b < 45.0 else "Overexposed"

    # 3. Contrast check (Standard deviation)
    std_c = float(np.std(gray))
    contrast_passed = std_c >= 18.0
    contrast_status = "Good" if contrast_passed else "Low Contrast"

    # 4. Glare check (percentage of saturated pixels)
    glare_pixels = np.sum(gray >= 254)
    glare_pct = float((glare_pixels / (h * w)) * 100.0)
    glare_passed = glare_pct <= 45.0 or (mean_b > 240 and std_c >= 18.0)
    glare_status = "Normal" if glare_passed else "Excessive Glare"

    # 5. Resolution check (minimum dimensions for cheques)
    min_dim = min(w, h)
    max_dim = max(w, h)
    res_passed = min_dim >= 180 and max_dim >= 500
    res_status = "High Resolution" if max_dim >= 1000 else "Acceptable" if res_passed else "Low Resolution"

    issues = []
    if not sharp_passed:
        issues.append(f"Image appears blurry (sharpness score: {lap_var:.1f}, min threshold: 40.0)")
    if mean_b < 45.0:
        issues.append(f"Image is too dark (brightness: {mean_b:.1f}, min: 45.0)")
    elif mean_b > 252.0 and std_c < 15.0:
        issues.append(f"Image is completely washed out / overexposed (brightness: {mean_b:.1f})")
    if not contrast_passed:
        issues.append(f"Image has low contrast (contrast: {std_c:.1f}, min: 18.0)")
    if not glare_passed:
        issues.append(f"Image has high glare/reflection ({glare_pct:.1f}% saturated pixels)")
    if not res_passed:
        issues.append(f"Image resolution is too low ({w}x{h}, minimum required: 500x180)")

    is_sufficient = sharp_passed and bright_passed and contrast_passed and glare_passed and res_passed

    return {
        "is_quality_sufficient": is_sufficient,
        "overall_status": "Passed" if is_sufficient else "Action Required",
        "checks": {
            "sharpness": {
                "passed": sharp_passed,
                "score": round(lap_var, 1),
                "threshold": 40.0,
                "status": sharp_status
            },
            "brightness": {
                "passed": bright_passed,
                "score": round(mean_b, 1),
                "min": 45.0,
                "max": 252.0,
                "status": bright_status
            },
            "contrast": {
                "passed": contrast_passed,
                "score": round(std_c, 1),
                "threshold": 18.0,
                "status": contrast_status
            },
            "glare": {
                "passed": glare_passed,
                "score": round(glare_pct, 1),
                "max": 45.0,
                "status": glare_status
            },
            "resolution": {
                "passed": res_passed,
                "width": w,
                "height": h,
                "status": res_status
            }
        },
        "issues": issues
    }

def get_ifsc_branch_suffixes(ifsc: Optional[str]) -> List[str]:
    """
    Extracts branch code endings from an IFSC code.
    In India, the last 6 characters of an 11-char IFSC represent the branch code.
    In most banks (e.g. HDFC, ICICI, SBI, etc.), the account number starts with the branch code or ending digits.
    """

    if not ifsc or len(ifsc) < 6:
        return []

    branch_code = ifsc[5:].upper()
    digits_only = re.sub(r"[^0-9]", "", branch_code)
    candidates = set()

    if digits_only:
        candidates.add(digits_only)
        stripped = digits_only.lstrip("0")
        if stripped:
            candidates.add(stripped)

        if len(digits_only) >= 4:
            candidates.add(digits_only[-4:])

        if len(digits_only) >= 3:
            candidates.add(digits_only[-3:])

    return [c for c in candidates if len(c) >= 2]

# ─── Text Normalization Helpers ───────────────────────────────────────────────

def split_single_merged_token(token: str) -> List[str]:
    """Recursively splits a single merged token like 'MUKULKUMARYADAV' into ['MUKUL', 'KUMAR', 'YADAV']."""
    t_up = token.upper().strip()
    if len(t_up) < 6 or t_up in DO_NOT_SPLIT_NAMES:
        return [token]

    for s in SORTED_SURNAMES:
        if t_up.endswith(s) and len(t_up) > len(s):
            prefix = t_up[:-len(s)].strip()
            min_prefix_len = 4 if len(s) <= 3 else 3
            if len(prefix) >= min_prefix_len and prefix.isalpha():
                if t_up in DO_NOT_SPLIT_NAMES:
                    return [token]
                sub_parts = split_single_merged_token(prefix)
                return sub_parts + [s]
    return [token]

def split_merged_name(name: str) -> str:
    """Splits merged OCR names like 'MUKULKUMARYADAV' or 'MANOJRAWAT' into 'MUKUL KUMAR YADAV'."""
    if not name:
        return name
    words = name.split()
    all_parts = []
    for w in words:
        all_parts.extend(split_single_merged_token(w))
    return " ".join(all_parts)

def sanitize_address_pincode(addr: str) -> str:
    """Fix common OCR letter confusions in Indian PIN codes (e.g. 4O0001 -> 400001)."""
    def fix_pin(m):
        return m.group(1).replace('O', '0').replace('o', '0')
    return re.sub(r"\b([1-9][0-9Oo]{5})\b", fix_pin, addr)

def format_title_name(name: Optional[str]) -> Optional[str]:
    """Nicely title-cases names while preserving acronyms like PVT, LTD, LLP."""
    if not name:
        return name
    words = name.split()
    preserve_upper = {"PVT", "LTD", "LIMITED", "INC", "CORP", "LLP", "PLC", "CO", "SBAC", "RTGS", "NEFT"}
    res = []
    for w in words:
        w_clean = re.sub(r"[^A-Za-z]", "", w)
        if w_clean.upper() in preserve_upper:
            res.append(w.upper())
        elif len(w_clean) <= 1:
            res.append(w.upper())
        else:
            res.append(w.capitalize())
    return " ".join(res)

# ─── Field Extractors ──────────────────────────────────────────────────────────

def extract_ifsc_code(blocks: List[ChequeOCRBlock], full_text: str) -> Tuple[Optional[str], float]:
    """Extracts 11-digit IFSC code with auto-correction for OCR glitches."""
    label_pattern = re.compile(
        r"(?:IFSC|IFS\s*CODE|RTGS\s*/?\s*NEFT\s*IFSC|RTGS\s*IFSC|NEFT\s*IFSC)[\s\.:#\-_]*([A-Za-z0-9]{11})",
        re.IGNORECASE
    )
    match = label_pattern.search(full_text)
    if match:
        raw_code = match.group(1).upper()
        corrected = sanitize_ifsc(raw_code)
        if is_valid_ifsc(corrected):
            return corrected, 0.99

    for block in blocks:
        m = re.search(r"\b([A-Za-z]{4}[0O1Il][A-Za-z0-9]{6})\b", block.text)
        if m:
            candidate = sanitize_ifsc(m.group(1).upper())
            if is_valid_ifsc(candidate):
                return candidate, block.confidence

    spaced_pattern = re.compile(r"\b([A-Za-z]{4})\s*([0O1Il][A-Za-z0-9]{6})\b")
    m_spaced = spaced_pattern.search(full_text)
    if m_spaced:
        candidate = sanitize_ifsc((m_spaced.group(1) + m_spaced.group(2)).upper())
        if is_valid_ifsc(candidate):
            return candidate, 0.92

    return None, 0.0

def sanitize_ifsc(code: str) -> str:
    """Normalizes and fixes common OCR glitches in IFSC codes."""
    if len(code) != 11:
        return code
    chars = list(code.upper())
    if chars[4] in {'O', 'D', 'Q', 'U', 'o'}:
        chars[4] = '0'
    replacements = {'0': 'O', '1': 'I', '5': 'S', '8': 'B'}
    for i in range(4):
        if chars[i] in replacements:
            chars[i] = replacements[chars[i]]
    return "".join(chars)

def is_valid_ifsc(code: str) -> bool:
    return bool(re.match(r"^[A-Z]{4}0[A-Z0-9]{6}$", code))

def extract_bank_name(blocks: List[ChequeOCRBlock], ifsc: Optional[str], full_text: str) -> Tuple[Optional[str], float]:
    """Extracts Bank Name using known banks dictionary, header text match, or IFSC prefix."""
    upper_text = full_text.upper()
    header_blocks = [b for b in blocks if b.norm_y < 0.45]
    header_text = " ".join(b.text.upper() for b in header_blocks)

    for name in KNOWN_BANK_NAMES:
        if name in header_text or name in upper_text:
            return title_case_bank(name), 0.98

    for b in header_blocks:
        t = b.text.upper().strip()
        if re.search(r"\b(?:[A-Z\s]{3,30})\s+BANK(?:\s+LTD|\s+LIMITED)?\b", t):
            cleaned = re.sub(r"[^A-Z\s]", " ", t)
            words = [w for w in cleaned.split() if w not in {"BRANCH", "IFSC", "RTGS", "NEFT"}]
            if "BANK" in words:
                bank_idx = words.index("BANK")
                candidate_words = words[:bank_idx + 1]
                if len(words) > bank_idx + 2 and words[bank_idx + 1] == "OF" and words[bank_idx + 2] == "INDIA":
                    candidate_words = words[:bank_idx + 3]
                elif "OF INDIA" in upper_text and " ".join(candidate_words) in {"UNION BANK", "STATE BANK", "CENTRAL BANK", "BANK"}:
                    candidate_words.extend(["OF", "INDIA"])
                if len(candidate_words) >= 2:
                    return title_case_bank(" ".join(candidate_words)), b.confidence

    if ifsc and len(ifsc) >= 4:
        prefix = ifsc[:4].upper()
        if prefix in BANK_IFSC_MAP:
            return BANK_IFSC_MAP[prefix], 0.95

    return None, 0.0

def title_case_bank(name: str) -> str:
    words = name.split()
    small_words = {"OF", "AND", "&"}
    acronyms = {"HDFC", "ICICI", "IDFC", "SBI", "PNB", "RBL", "UCO", "CSB", "DBS", "HSBC", "AU"}
    formatted = []
    for i, w in enumerate(words):
        w_up = w.upper()
        if w_up in acronyms:
            formatted.append(w_up)
        elif w_up in small_words and i > 0:
            formatted.append(w.lower())
        else:
            formatted.append(w.capitalize())
    return " ".join(formatted)

def extract_account_number(
    blocks: List[ChequeOCRBlock],
    full_text: str,
    ifsc: Optional[str] = None
) -> Tuple[Optional[str], float, Dict[str, Any]]:
    """
    Extracts Account Number (9 to 18 digits in India) by verifying 2 conditions:
    Condition 1: Has an account number title ('A/c No.', 'Account No.', etc.) printed previous to the detail.
    Condition 2: In most cases, the account number starts with the ending digits of the IFSC code (branch code).
    """
    ifsc_suffixes = get_ifsc_branch_suffixes(ifsc)
    candidates = []

    # Case 1: Title and number in the SAME block (e.g. "A/c No. 05841050073037")
    for b in blocks:
        m = re.search(
            r"(?:A\/[C0O]|A\/c|ACCOUNT|ACC|AC)\s*(?:N[O0o]|NUM|NUMBER|CODE)?[\s\.:#\-_]*([0-9]{9,18})\b",
            b.text,
            re.IGNORECASE
        )
        if m:
            num = m.group(1).strip()
            if is_valid_account_number(num):
                matches_ifsc = any(num.startswith(s) for s in ifsc_suffixes) if ifsc_suffixes else False
                matched_suffix = next((s for s in ifsc_suffixes if num.startswith(s)), None)
                score = 60 + (40 if matches_ifsc else 0)
                candidates.append({
                    "number": num,
                    "confidence": b.confidence,
                    "has_previous_title": True,
                    "title_matched": b.text,
                    "matches_ifsc_branch": matches_ifsc,
                    "matched_ifsc_suffix": matched_suffix,
                    "score": score,
                    "block": b
                })

    # Case 2: Title in one block, and number in an ADJACENT / SUBSEQUENT block
    for i, b_title in enumerate(blocks):
        t_clean = b_title.text.upper().strip()
        has_title = any(re.search(pat, t_clean) for pat in [
            r"^(?:A\/[C0O]|A\/c|ACCOUNT|ACC|AC)\s*(?:N[O0o]|NUM|NUMBER|CODE)?[\s\.:#\-_]*$",
            r"^(?:CURRENT|SAVINGS|SB|CA)\s*(?:A\/[C0O])?\s*(?:N[O0o])?[\s\.:#\-_]*$",
            r"A\/[C0O]\s*N[O0o]",
            r"A\/[C0O]",
            r"AC\s*N[O0o]",
            r"ACCOUNT\s*N[O0o]",
            r"ACCOUNT\s*NUMBER"
        ])
        if has_title:
            for j, b_num in enumerate(blocks):
                if i == j:
                    continue
                raw_clean = b_num.text.replace(" ", "").replace("-", "")
                m_dig = re.search(r"^[0-9]{9,18}$", raw_clean)
                if m_dig:
                    num = m_dig.group(0)
                    if not is_valid_account_number(num):
                        continue

                    # Horizontally aligned to the right (same line)
                    is_horizontal = (
                        abs(b_title.norm_y - b_num.norm_y) <= 0.08 
                        and b_num.norm_x > b_title.norm_x 
                        and (b_num.norm_x - b_title.norm_x) <= 0.45
                    )
                    # Vertically below the title (stacked A/C box)
                    is_vertical = (
                        b_num.norm_y > b_title.norm_y 
                        and (b_num.norm_y - b_title.norm_y) <= 0.12 
                        and abs(b_num.norm_x - b_title.norm_x) <= 0.25
                    )

                    if is_horizontal or is_vertical:
                        matches_ifsc = any(num.startswith(s) for s in ifsc_suffixes) if ifsc_suffixes else False
                        matched_suffix = next((s for s in ifsc_suffixes if num.startswith(s)), None)
                        spatial_score = 50 if is_horizontal else 40
                        score = spatial_score + (40 if matches_ifsc else 0)
                        candidates.append({
                            "number": num,
                            "confidence": b_num.confidence,
                            "has_previous_title": True,
                            "title_matched": b_title.text,
                            "matches_ifsc_branch": matches_ifsc,
                            "matched_ifsc_suffix": matched_suffix,
                            "score": score,
                            "block": b_num
                        })

    # Case 3: Middle-band candidates (0.32 <= y <= 0.70)
    for b in blocks:
        if 0.32 <= b.norm_y <= 0.70:
            clean = b.text.replace(" ", "").replace("-", "")
            if re.match(r"^[0-9]{9,18}$", clean) and is_valid_account_number(clean):
                if not any(c["number"] == clean for c in candidates):
                    matches_ifsc = any(clean.startswith(s) for s in ifsc_suffixes) if ifsc_suffixes else False
                    matched_suffix = next((s for s in ifsc_suffixes if clean.startswith(s)), None)
                    score = 20 + (40 if matches_ifsc else 0)
                    candidates.append({
                        "number": clean,
                        "confidence": b.confidence,
                        "has_previous_title": False,
                        "title_matched": None,
                        "matches_ifsc_branch": matches_ifsc,
                        "matched_ifsc_suffix": matched_suffix,
                        "score": score,
                        "block": b
                    })

    if not candidates:
        return None, 0.0, {
            "has_previous_title": False,
            "matches_ifsc_branch": False,
            "title_matched": None,
            "matched_ifsc_suffix": None
        }

    # Sort candidates: condition 1 + condition 2 highest score first
    candidates.sort(key=lambda c: (-c["score"], -c["confidence"], abs(c["block"].norm_y - 0.52)))
    best = candidates[0]

    metadata = {
        "has_previous_title": best["has_previous_title"],
        "title_matched": best["title_matched"],
        "matches_ifsc_branch": best["matches_ifsc_branch"],
        "matched_ifsc_suffix": best["matched_ifsc_suffix"],
        "all_candidates_count": len(candidates)
    }
    return best["number"], best["confidence"], metadata

def is_valid_account_number(num_str: str) -> bool:
    if not num_str.isdigit():
        return False
    length = len(num_str)
    if length < 9 or length > 18:
        return False
    if len(set(num_str)) <= 1:
        return False
    return True

def extract_branch_details(
    blocks: List[ChequeOCRBlock],
    ifsc: Optional[str],
    full_text: str
) -> Tuple[Optional[str], Optional[str], float]:
    """Extracts (branch_name, branch_address, confidence)."""
    header_blocks = [b for b in blocks if b.norm_y < 0.25 and b.norm_x < 0.85]
    branch_name = None
    address_lines: List[str] = []
    conf = 0.85

    # Check for explicit "Branch: ..." or "Br: ..." or "<Location> Branch"
    branch_label_pattern = re.compile(
        r"(?:BRANCH|BR|BO)[\s\.:#\-_]+([A-Za-z0-9\s,\-\.]{3,50})",
        re.IGNORECASE
    )
    branch_prefix_pattern = re.compile(
        r"\b([A-Za-z0-9\s,\-\.]{3,30})\s+(?:BRANCH|BR)\b",
        re.IGNORECASE
    )
    for b in header_blocks:
        m = branch_label_pattern.search(b.text)
        if m:
            cand = m.group(1).strip()
            cand = re.sub(r"(?:IFSC|TEL|PH|PIN|PHONE).*$", "", cand, flags=re.IGNORECASE).strip()
            cand = cand.strip(" ,.-:")
            if (
                cand 
                and len(cand) >= 3 
                and cand.upper() not in ANNOTATION_TITLES 
                and cand.upper() not in {"ADDRESS", "BRANCH ADDRESS", "BRANCH", "BANK", "OF INDIA", "INDIA"}
                and not branch_name
            ):
                branch_name = format_title_name(cand)
                conf = max(conf, b.confidence)

        m_pre = branch_prefix_pattern.search(b.text)
        if m_pre and not branch_name:
            cand = m_pre.group(1).strip().strip(" ,.-:")
            if (
                cand 
                and len(cand) >= 3 
                and cand.upper() not in ANNOTATION_TITLES 
                and cand.upper() not in {"ADDRESS", "BRANCH ADDRESS", "BRANCH", "BANK", "OF INDIA", "INDIA"}
            ):
                branch_name = format_title_name(f"{cand} Branch")
                conf = max(conf, b.confidence)

    skip_terms = ANNOTATION_TITLES.union({
        "VALID FOR", "MONTHS ONLY", "PAY", "RUPEES", "DATE", "OR BEARER", "WEEKLY HOLIDAY", "OF INDIA", "INDIA"
    })

    for b in header_blocks:
        t_up = b.text.upper().strip()
        if any(st in t_up for st in ["RUPEES", "PAY", "BEARER", "VALID FOR", "DATE", "MONTHS"]):
            continue
        if any(t_up == st or t_up.startswith(st + ":") for st in skip_terms):
            continue
        if any(bank in t_up for bank in KNOWN_BANK_NAMES):
            continue
        if re.search(r"(?:IFSC|IFS\s*CODE)[\s\.:#\-_]*[A-Za-z0-9]+", b.text, re.IGNORECASE):
            continue

        raw_line = b.text.strip()
        # Clean glued tokens
        raw_line = re.sub(r"([0-9]{6})[A-Za-z]+", r"\1", raw_line)
        raw_line = re.sub(r"([A-Za-z]{3,})([0-9]{2,})", r"\1 \2", raw_line)
        raw_line = re.sub(r"([0-9]{2,})([A-Za-z]{3,})", r"\1 \2", raw_line)
        raw_line = re.sub(r"([A-Za-z]+)ROAD\b", r"\1 ROAD", raw_line, flags=re.IGNORECASE)
        raw_line = re.sub(r"([A-Za-z]+)MARKET\b", r"\1 MARKET", raw_line, flags=re.IGNORECASE)
        raw_line = re.sub(r"NEWDELHI\b", "NEW DELHI", raw_line, flags=re.IGNORECASE)
        raw_line = re.sub(r"BLOCKNO\b", "BLOCK NO", raw_line, flags=re.IGNORECASE)
        raw_line = sanitize_address_pincode(raw_line)
        cleaned_line = re.sub(r"\s+", " ", raw_line).strip(" ,.-")

        if cleaned_line and len(cleaned_line) >= 4 and cleaned_line not in address_lines:
            address_lines.append(cleaned_line)

    branch_address = ", ".join(address_lines) if address_lines else None

    if not branch_name and address_lines:
        for line in reversed(address_lines):
            m_pin = re.search(r"([A-Za-z\s]+)(?:-|–)?\s*([0-9]{6})?", line)
            if m_pin and len(m_pin.group(1).strip()) >= 3:
                cand_branch = m_pin.group(1).strip().strip(" ,-")
                if len(cand_branch) >= 3 and cand_branch.upper() not in ANNOTATION_TITLES:
                    branch_name = format_title_name(cand_branch)
                    break
        if not branch_name:
            branch_name = format_title_name(address_lines[0].split(",")[0])

    if not branch_name and ifsc and len(ifsc) == 11:
        branch_name = f"Branch ({ifsc[5:]})"

    return branch_name, format_title_name(branch_address), conf

def extract_account_holder_name(blocks: List[ChequeOCRBlock], full_text: str) -> Tuple[Optional[str], Optional[str], float]:
    """Extracts (account_holder_name, signatory_name, confidence)."""
    sig_zone_blocks = [
        b for b in blocks 
        if b.norm_x >= 0.50 and 0.50 <= b.norm_y <= 0.88
    ]
    sig_zone_blocks.sort(key=lambda b: (b.norm_y, b.norm_x))

    company_name = None
    company_conf = 0.0
    signatory_name = None
    signatory_conf = 0.0

    # 1. Check for explicit "For <Company/Individual>"
    for b in sig_zone_blocks:
        m = re.search(r"^\s*FOR\s+([A-Za-z0-9\s\.\&\(\)\-]+)", b.text, re.IGNORECASE)
        if m:
            cand = clean_holder_name(m.group(1))
            if is_valid_holder_name(cand):
                company_name = cand
                company_conf = b.confidence
                break

    # 2. Check candidate names in signature zone
    candidate_person_names = []
    for b in sig_zone_blocks:
        t_up = b.text.upper().strip()
        if any(t_up == st or t_up.startswith(st + ":") for st in ANNOTATION_TITLES):
            continue
        if any(term in t_up for term in ["SIGN", "SIGNATORY", "PLEASE SIGN", "CLEARING", "TRANSFER", "BRANCH"]):
            continue
        cleaned = clean_holder_name(b.text)
        if is_valid_holder_name(cleaned) and cleaned != company_name:
            split_name = split_merged_name(cleaned)
            candidate_person_names.append((split_name, b.confidence))

    if candidate_person_names:
        if len(candidate_person_names) > 1 and not company_name:
            # Check if candidates are distinct joint holders or signature + printed name of same person
            names = [n for n, _ in candidate_person_names]
            w_sets = [set(n.upper().split()) for n in names]
            has_overlap = any(bool(w_sets[i] & w_sets[j]) for i in range(len(w_sets)) for j in range(i + 1, len(w_sets)))
            if not has_overlap:
                names_combined = " / ".join(names)
                avg_conf = sum(conf for _, conf in candidate_person_names) / len(candidate_person_names)
                return names_combined, None, avg_conf
        signatory_name, signatory_conf = candidate_person_names[-1]

    if company_name:
        return company_name, signatory_name, max(company_conf, signatory_conf)
    elif signatory_name:
        return signatory_name, None, signatory_conf

    return None, None, 0.0

def clean_holder_name(text: str) -> str:
    cand = text.strip()
    cand = re.sub(r"^FOR\s*[\:\/\-]?\s*", "", cand, flags=re.IGNORECASE).strip()
    cand = re.sub(r"[^A-Za-z\s\.\&\-]", " ", cand)
    cand = re.sub(r"\s+", " ", cand).strip(" .-_")
    return cand

def is_valid_holder_name(text: str) -> bool:
    if not text or len(text) < 3:
        return False
    t_up = text.upper()
    if any(t_up == noise or t_up.startswith(noise + ":") for noise in SIGNATURE_NOISE_TERMS):
        return False
    if any(bank in t_up for bank in KNOWN_BANK_NAMES):
        return False
    if any(kw in t_up for kw in ["BEARER", "RUPEES", "CTS-2010", "PAYABLE", "BRANCH", "CLEARING", "CHEQUE", "CHK", "A/C", "AC NO"]):
        return False
    words = [w for w in text.split() if len(w) >= 2 and w.isalpha()]
    return len(words) >= 1

def extract_micr_and_cheque_number(
    blocks: List[ChequeOCRBlock],
    full_text: str
) -> Tuple[Optional[str], Optional[str], float]:
    """
    Extracts (cheque_number, micr_code, confidence) from bottom CTS-2010 band.
    In CTS-2010 cheques, the numbers in the bottom band strictly follow:
      - 1st number: Cheque Number (typically 6 digits, e.g., 000077)
      - 2nd number: MICR Code (9 digits: 3 city + 3 bank + 3 branch, e.g., 110229054)
      - 3rd number: Account identifier / RBI code (6 digits)
      - 4th number: Transaction code (2 or 3 digits)
    """
    digit_blocks = [b for b in blocks if re.search(r"\d", b.text)]
    if not digit_blocks:
        return None, None, 0.0

    # Focus strictly on CTS-2010 bottom band (norm_y >= 0.75)
    cts_candidates = [b for b in digit_blocks if b.norm_y >= 0.75]
    if not cts_candidates:
        return None, None, 0.0

    max_y = max(b.norm_y for b in cts_candidates)

    # Focus on blocks located in the bottom-most band (within 0.09 vertical delta or >= 0.80 of cheque height)
    band_blocks = []
    for b in cts_candidates:
        t_up = b.text.upper()
        if any(ann in t_up for ann in ["PLEASE", "SIGN", "HOLDER", "NAME", "DATE", "PAYABLE", "RUPEES", "BEARER"]):
            continue
        if abs(b.norm_y - max_y) <= 0.09 or b.norm_y >= 0.80:
            band_blocks.append(b)

    # Sort strictly left-to-right (horizontal reading order)
    band_blocks.sort(key=lambda b: b.norm_x)

    tokens = []
    avg_conf = 0.85
    if band_blocks:
        avg_conf = sum(b.confidence for b in band_blocks) / len(band_blocks)

    for b in band_blocks:
        # Split by whitespace and common OCR delimiter characters for MICR E-13B font
        raw_parts = re.split(r"[\s\:\;\⑈\⑆\|\'\"\`\-\_.\,\/\\]+", b.text)
        for part in raw_parts:
            digits = re.sub(r"\D", "", part)
            if digits:
                tokens.append(digits)

    # If first token contains merged cheque number (6 digits) and MICR code (9 digits), unpack them
    expanded_tokens = []
    for idx, t in enumerate(tokens):
        if idx == 0 and len(t) >= 15:
            expanded_tokens.append(t[:6])
            expanded_tokens.append(t[6:15])
            if len(t) > 15:
                expanded_tokens.append(t[15:])
        else:
            expanded_tokens.append(t)
    tokens = expanded_tokens

    cheque_no = None
    micr_code = None

    # Enforce CTS-2010 positional mapping:
    # 1st number is Cheque Number (typically 6 digits)
    if len(tokens) >= 1:
        cheque_no = tokens[0]
        # In E-13B MICR font, vertical delimiter ⑈ or " is often read as '1' or '0' (e.g. 1446783 -> 446783)
        if len(cheque_no) == 7 and cheque_no.startswith("1"):
            cheque_no = cheque_no[1:]
        elif len(cheque_no) > 6:
            cheque_no = cheque_no[:6]

    # 2nd number in CTS-2010 cheque bottom row is ALWAYS the MICR code (9 digits)
    if len(tokens) >= 2:
        candidate_micr = tokens[1]
        if len(candidate_micr) == 9:
            micr_code = candidate_micr
        elif len(candidate_micr) == 3 and len(tokens) >= 3 and len(tokens[2]) == 6:
            # City code (3 digits) split from bank/branch (6 digits)
            micr_code = candidate_micr + tokens[2]
        elif len(candidate_micr) == 3 and len(tokens) >= 4 and len(tokens[2]) == 3 and len(tokens[3]) == 3:
            # City (3 digits) + Bank (3 digits) + Branch (3 digits)
            micr_code = candidate_micr + tokens[2] + tokens[3]
        elif len(candidate_micr) == 6 and len(tokens) >= 3 and len(tokens[2]) == 3:
            # City + Bank (6 digits) + Branch (3 digits)
            micr_code = candidate_micr + tokens[2]
        elif len(candidate_micr) == 10:
            # Trailing OCR artifact delimiter often recognized as 1 or :
            micr_code = candidate_micr[:9]
        elif len(candidate_micr) > 9:
            micr_code = candidate_micr[:9]

    # Fallback to search among all bottom tokens from position 1 onwards if positional didn't yield 9 digits
    if not micr_code or len(micr_code) != 9:
        for t in tokens[1:]:
            if len(t) == 9:
                micr_code = t
                break
            elif len(t) == 10:
                micr_code = t[:9]
                break

    # Also parse from the bottom lines of full_text as a fallback if not yet found
    if not micr_code:
        bottom_lines = [l.strip() for l in full_text.splitlines() if l.strip()][-4:]
        for line in reversed(bottom_lines):
            m_9 = re.search(r"\b\d{9}\b", line)
            if m_9:
                micr_code = m_9.group(0)
                break
            m_merged = re.search(r"\b\d{15,}\b", line)
            if m_merged:
                val = m_merged.group(0)
                if not cheque_no:
                    cheque_no = val[:6]
                micr_code = val[6:15]
                break

    # Fallback search across entire full text for 9 digits if not yet found
    if not micr_code:
        m = re.search(r"\b\d{9}\b", full_text)
        if m:
            micr_code = m.group(0)

    return cheque_no, micr_code, round(avg_conf, 2)

# ─── Main Orchestrator ─────────────────────────────────────────────────────────

def read_cheque(file_path: str) -> Dict[str, Any]:
    """
    Reads a cheque document (PDF or Image) and extracts:
    - account_number
    - ifsc_code
    - bank_name
    - branch_name & branch_address
    - address
    - account_holder_name
    - signatory_name
    - cheque_number
    - micr_code
    - raw_text
    - confidence
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cheque file not found: {file_path}")

        # 1. Load image (renders PDF or reads image)
    raw_img, doc_type = load_cheque_image(file_path)

    # 2. Check image quality checkpoint FIRST
    # If image quality is low, do NOT proceed further (skip expensive OCR/orientation) and return error
    image_quality = check_image_quality(raw_img)
    if not image_quality["is_quality_sufficient"]:
        issues_str = "; ".join(image_quality["issues"]) if image_quality.get("issues") else "Image quality checks failed"
        return {
            "success": False,
            "error": f"Image quality is too low to process cheque: {issues_str}",
            "doc_type": doc_type,
            "issues": image_quality.get("issues", []),
            "account_number": None,
            "account_number_verification": None,
            "ifsc_code": None,
            "bank_name": None,
            "branch_details": None,
            "address": None,
            "account_holder_name": None,
            "signatory_name": None,
            "cheque_number": None,
            "micr_code": None,
            "confidence": 0.0,
            "confidence_scores": {}
        }

    # 3. Automatically orient cheque (handles portrait scans & upside-down orientation)
    upright_img, blocks = auto_orient_cheque(raw_img)

    # 3b. Verify bottom CTS-2010 band (norm_y >= 0.78)
    # If the bottom band OCR didn't catch the E-13B MICR line, run a dedicated crop of the bottom 25%
    bottom_digits = [b for b in blocks if b.norm_y >= 0.78 and re.search(r"\d", b.text)]
    if not bottom_digits:
        h_up, w_up = upright_img.shape[:2]
        crop_top = int(h_up * 0.75)
        bottom_crop = upright_img[crop_top:, :]
        engine = get_ocr_engine()
        res_crop, _ = engine(bottom_crop)
        if res_crop:
            for box, text, score in res_crop:
                abs_box = [[pt[0], pt[1] + crop_top] for pt in box]
                b_crop = ChequeOCRBlock(text, abs_box, float(score), w_up, h_up)
                blocks.append(b_crop)

    # 4. Compile full text
    full_text = "\n".join(b.text for b in blocks)

    # 4. Extract fields
    ifsc, ifsc_conf = extract_ifsc_code(blocks, full_text)
    bank_name, bank_conf = extract_bank_name(blocks, ifsc, full_text)
    acc_num, acc_conf, acc_verification = extract_account_number(blocks, full_text, ifsc=ifsc)
    branch_name, branch_address, branch_conf = extract_branch_details(blocks, ifsc, full_text)
    holder_name, signatory_name, holder_conf = extract_account_holder_name(blocks, full_text)
    holder_name = format_title_name(holder_name)
    signatory_name = format_title_name(signatory_name)
    cheque_num, micr, micr_conf = extract_micr_and_cheque_number(blocks, full_text)

    # Overall confidence calculation
    confs = [c for c in [ifsc_conf, bank_conf, acc_conf, branch_conf, holder_conf] if c > 0.0]
    overall_conf = round(float(np.mean(confs)), 3) if confs else 0.0

    return {
        "success": True,
        "doc_type": doc_type,
        "account_number": acc_num,
        "account_number_verification": acc_verification,
        "ifsc_code": ifsc,
        "bank_name": bank_name,
        "branch_details": {
            "branch_name": branch_name,
            "address": branch_address
        },
        "address": branch_address,
        "account_holder_name": holder_name,
        "signatory_name": signatory_name,
        "cheque_number": cheque_num,
        "micr_code": micr,
        "confidence": overall_conf,
        "confidence_scores": {
            "account_number": round(acc_conf, 2),
            "ifsc_code": round(ifsc_conf, 2),
            "bank_name": round(bank_conf, 2),
            "branch": round(branch_conf, 2),
            "account_holder_name": round(holder_conf, 2),
        }
    }


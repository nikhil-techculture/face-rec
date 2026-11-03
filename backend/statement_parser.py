"""
Bank Statement Parser
Extracts:
  - bank_name
  - account_number (and indicates if masked, e.g. XXXXXXXXX7009)
  - account_holder_name
  - ifsc_code & branch_details (branch_name, branch_code, branch_address)
  - statement_period:
      - start_date
      - end_date
      - days_covered
      - months_covered
      - has_6_months (boolean check against required threshold)
      - shortfall_days
      - status_message
  - balances:
      - opening_balance
      - closing_balance
  - total_transactions_count
Supports PDF (native text and OCR fallback) and image formats.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import fitz  # PyMuPDF


# ─── Bank IFSC Prefix Mapping ──────────────────────────────────────────────────
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
    "AIRP": "Airtel Payments Bank",
    "PYTM": "Paytm Payments Bank",
    "IPOS": "India Post Payments Bank",
    "ESFB": "Equitas Small Finance Bank",
    "AUBL": "AU Small Finance Bank",
    "UJVN": "Ujjivan Small Finance Bank",
}

KNOWN_BANK_NAMES = [
    "STATE BANK OF INDIA", "CANARA BANK", "HDFC BANK", "ICICI BANK", "AXIS BANK",
    "PUNJAB NATIONAL BANK", "BANK OF BARODA", "KOTAK MAHINDRA BANK", "UNION BANK OF INDIA",
    "BANK OF INDIA", "INDIAN BANK", "INDIAN OVERSEAS BANK", "CENTRAL BANK OF INDIA",
    "UCO BANK", "PUNJAB & SIND BANK", "BANDHAN BANK", "IDFC FIRST BANK",
    "FEDERAL BANK", "INDUSIND BANK", "YES BANK", "RBL BANK", "KARUR VYSYA BANK",
    "SOUTH INDIAN BANK", "CSB BANK", "BANK OF MAHARASHTRA", "AIRTEL PAYMENTS BANK",
    "PAYTM PAYMENTS BANK", "INDIA POST PAYMENTS BANK", "AU SMALL FINANCE BANK",
    "EQUITAS SMALL FINANCE BANK", "UJJIVAN SMALL FINANCE BANK"
]

SALUTATIONS = {
    "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "miss", "dr", "dr.",
    "shri", "smt", "kumari", "prof", "m/s", "messrs"
}


# ─── Helper Functions ──────────────────────────────────────────────────────────

def parse_date_flexible(date_str: str) -> Optional[datetime]:
    """Parses various date formats common in Indian bank statements."""
    clean = date_str.strip().strip(",.;- ")
    date_formats = [
        "%d-%b-%Y", "%d-%B-%Y", "%d-%b-%y",
        "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
        "%Y-%m-%d", "%Y/%m/%d",
        "%d %b %Y", "%d %B %Y", "%d %b, %Y",
    ]
    for fmt in date_formats:
        try:
            return datetime.strptime(clean, fmt)
        except ValueError:
            pass
    return None


def format_title_name(name: Optional[str]) -> Optional[str]:
    """Formats name into standard Title Case while preserving acronyms."""
    if not name or not name.strip():
        return None
    words = name.strip().split()
    formatted = []
    for w in words:
        if w.upper() in {"PVT", "LTD", "LLP", "INC", "CORP", "A/C", "DR", "CR"}:
            formatted.append(w.upper())
        elif len(w) == 1:
            formatted.append(w.upper())
        else:
            formatted.append(w.capitalize())
    return " ".join(formatted)


# ─── Extraction Sub-modules ────────────────────────────────────────────────────

def extract_bank_name(full_text: str, ifsc: Optional[str]) -> Tuple[Optional[str], float]:
    """Identifies the bank name via IFSC prefix, domain names, or text keywords."""
    # 1. From IFSC prefix
    if ifsc and len(ifsc) >= 4:
        prefix = ifsc[:4].upper()
        if prefix in BANK_IFSC_MAP:
            return BANK_IFSC_MAP[prefix], 0.99

    # 2. Known bank name mentions in header
    text_upper = full_text[:4000].upper()
    for bank in KNOWN_BANK_NAMES:
        if bank in text_upper:
            return format_title_name(bank), 0.95

    # 3. Bank domain matching
    domain_map = {
        "canarabank": "Canara Bank",
        "hdfcbank": "HDFC Bank",
        "icicibank": "ICICI Bank",
        "axisbank": "Axis Bank",
        "sbi.co.in": "State Bank of India",
        "kotak": "Kotak Mahindra Bank",
        "pnbindia": "Punjab National Bank",
        "bankofbaroda": "Bank of Baroda",
        "unionbank": "Union Bank of India",
        "idfcfirstbank": "IDFC FIRST Bank",
    }
    for dom, bname in domain_map.items():
        if dom in full_text.lower():
            return bname, 0.90

    # 4. Fallback search across full text
    for bank in KNOWN_BANK_NAMES:
        if bank in full_text.upper():
            return format_title_name(bank), 0.85

    return None, 0.0


def extract_ifsc_and_branch(full_text: str, first_page_text: str = "") -> Dict[str, Any]:
    """Extracts IFSC code, branch name, code, and address."""
    # IFSC code
    ifsc_match = re.search(r"\b([A-Z]{4}0[A-Z0-9]{6})\b", full_text)
    ifsc = ifsc_match.group(1).upper() if ifsc_match else None

    branch_code = None
    branch_name = None
    branch_address = None

    lines = [l.strip() for l in (first_page_text or full_text[:4000]).splitlines() if l.strip()]

    # Multi-line header parsing (e.g., Canara Bank format)
    for i, line in enumerate(lines):
        l_lower = line.lower()
        if l_lower == "branch code" and i + 1 < len(lines):
            branch_code = lines[i + 1].strip()
        elif l_lower == "branch name" and i + 1 < len(lines):
            cand = lines[i + 1].strip()
            if cand.lower() not in {"ifsc code", "address", "branch code"}:
                branch_name = format_title_name(cand)
        elif l_lower == "ifsc code" and i + 2 < len(lines):
            # Address line often follows IFSC Code block
            if lines[i + 2].lower() == "address" and i + 3 < len(lines):
                addr_parts = []
                for j in range(i + 3, min(i + 7, len(lines))):
                    if lines[j].lower() in {"date", "particulars", "deposits", "withdrawals", "balance", "opening balance"}:
                        break
                    addr_parts.append(lines[j])
                if addr_parts:
                    branch_address = re.sub(r"\s+", " ", " ".join(addr_parts)).strip()

    # Inline regex fallback (e.g. "Branch Code: 2742", "Branch: Connaught Place")
    if not branch_code:
        bcode_match = re.search(r"Branch\s*Code[\s\:\-]*(\d+)", full_text, re.IGNORECASE)
        if bcode_match:
            branch_code = bcode_match.group(1)

    if not branch_name:
        bname_match = re.search(r"Branch\s*(?:Name)?[\s\:\-]*([A-Za-z0-9\s\,\.\-]+?)(?:\n|IFSC|Code|Phone|Email|Address)", full_text, re.IGNORECASE)
        if bname_match:
            cand = bname_match.group(1).strip()
            if len(cand) >= 3 and not cand.lower().startswith("code"):
                branch_name = format_title_name(cand)

    if not branch_address:
        addr_match = re.search(r"Branch\s*Address[\s\:\-]*([A-Za-z0-9\s\,\.\-\#\/]+?)(?:\n\s*\n|Date|Particulars|Phone|Email)", full_text, re.IGNORECASE)
        if addr_match:
            branch_address = re.sub(r"\s+", " ", addr_match.group(1)).strip()

    return {
        "ifsc_code": ifsc,
        "branch_code": branch_code,
        "branch_name": branch_name,
        "branch_address": branch_address,
    }


def extract_account_number(full_text: str) -> Dict[str, Any]:
    """
    Extracts account number. Handles masked formats (XXXXXXXXX7009) as well as unmasked formats.
    """
    # 1. Masked account numbers, e.g. Statement for A/c XXXXXXXXX7009 or A/c No: ****1234
    masked_match = re.search(
        r"(?:A/c|Account|SB|CA|Account\s*No\.?)[\s\:\.\#]*([X\*]{3,14}\d{3,6})",
        full_text,
        re.IGNORECASE
    )
    if masked_match:
        masked_val = masked_match.group(1).strip()
        last_digits_match = re.search(r"(\d{3,6})$", masked_val)
        last_digits = last_digits_match.group(1) if last_digits_match else None
        return {
            "account_number": masked_val,
            "is_masked": True,
            "masked_account_number": masked_val,
            "unmasked_account_number": None,
            "last_4_digits": last_digits[-4:] if last_digits and len(last_digits) >= 4 else last_digits,
            "confidence": 0.95
        }

    # 2. Canara e-passbook header style: "Statement for A/c XXXXXXXXX7009"
    direct_masked = re.search(r"\b([X\*]{6,14}\d{4})\b", full_text)
    if direct_masked:
        masked_val = direct_masked.group(1)
        return {
            "account_number": masked_val,
            "is_masked": True,
            "masked_account_number": masked_val,
            "unmasked_account_number": None,
            "last_4_digits": masked_val[-4:],
            "confidence": 0.95
        }

    # 3. Unmasked standard account numbers (9 to 18 digits)
    patterns = [
        r"(?:Statement\s+(?:of\s+Account(?:\s+for)?|for\s+A/c)|Account\s*(?:Number|No\.?)|A/c\s*No\.?|SB\s*A/c\s*No\.?|Current\s*A/c\s*No\.?)[\s\:\.\#\-]*([0-9]{9,18})\b",
        r"(?:Account|A/c)[\s\:\.\#\-]+([0-9]{9,18})\b",
    ]
    for pat in patterns:
        m = re.search(pat, full_text, re.IGNORECASE)
        if m:
            acc = m.group(1).strip()
            return {
                "account_number": acc,
                "is_masked": False,
                "masked_account_number": None,
                "unmasked_account_number": acc,
                "last_4_digits": acc[-4:] if len(acc) >= 4 else acc,
                "confidence": 0.98
            }

    # 4. Fallback search for 9-18 digit account preceded by keyword on same or next line
    lines = full_text[:4000].splitlines()
    for i, line in enumerate(lines):
        if re.search(r"\b(account\s*number|a/c\s*no|acc\s*no)\b", line, re.IGNORECASE):
            digits = re.findall(r"\b\d{9,18}\b", line)
            if digits:
                return {
                    "account_number": digits[0],
                    "is_masked": False,
                    "masked_account_number": None,
                    "unmasked_account_number": digits[0],
                    "last_4_digits": digits[0][-4:],
                    "confidence": 0.90
                }
            if i + 1 < len(lines):
                next_digits = re.findall(r"\b\d{9,18}\b", lines[i + 1])
                if next_digits:
                    return {
                        "account_number": next_digits[0],
                        "is_masked": False,
                        "masked_account_number": None,
                        "unmasked_account_number": next_digits[0],
                        "last_4_digits": next_digits[0][-4:],
                        "confidence": 0.88
                    }

    return {
        "account_number": None,
        "is_masked": False,
        "masked_account_number": None,
        "unmasked_account_number": None,
        "last_4_digits": None,
        "confidence": 0.0
    }


def extract_account_holder_name(full_text: str, first_page_text: str) -> Optional[str]:
    """Extracts account holder name from header blocks."""
    lines = [l.strip() for l in first_page_text.splitlines() if l.strip()]

    # 1. Look for explicit "Name" line followed by name on next line (e.g. Canara format)
    for i, line in enumerate(lines):
        if line.lower() in {"name", "customer name", "account holder name", "account holder"}:
            if i + 1 < len(lines):
                candidate = lines[i + 1].strip()
                if (len(candidate) >= 3 and
                    not any(candidate.lower().startswith(kw) for kw in ["phone", "address", "branch", "customer id", "date", "+91"]) and
                    not re.search(r"^\d+$", candidate)):
                    return format_title_name(candidate)

    # 2. Look for inline Name : RAHUL GOEL
    name_match = re.search(
        r"(?:Customer\s*Name|Account\s*Holder(?:\s*Name)?|Name\s*of\s*Account\s*Holder)[\s\:\-]*([A-Za-z\s\.\&\-]+?)(?:\n|Address|Phone|Email|Cust\s*ID|Customer\s*ID|Date)",
        first_page_text,
        re.IGNORECASE
    )
    if name_match:
        cand = name_match.group(1).strip()
        clean_words = [w for w in cand.split() if w.lower().strip(".:") not in SALUTATIONS]
        if clean_words:
            return format_title_name(" ".join(clean_words))

    # 3. Check for standard CASA drawdown or transaction note mentioning holder name
    casa_match = re.search(r"DRAWDOWN\s+FROM\s+CASA\s*[\:\.\-]?\s*([A-Za-z\s]+)", full_text, re.IGNORECASE)
    if casa_match:
        return format_title_name(casa_match.group(1).strip())

    return None


def extract_statement_period(
    full_text: str,
    all_dates: List[datetime],
    min_months_required: int = 6
) -> Dict[str, Any]:
    """
    Computes statement duration and checks if it satisfies the minimum required months (default 6).
    """
    header_start_date = None
    header_end_date = None

    period_patterns = [
        r"for\s+the\s+period\s+([0-9A-Za-z\-\/]+)\s+to\s+([0-9A-Za-z\-\/]+)",
        r"(?:Statement\s*)?Period[\s\:\-]*([0-9A-Za-z\-\/]+)\s+(?:to|-)\s+([0-9A-Za-z\-\/]+)",
        r"From\s*(?:Date)?[\s\:\-]*([0-9A-Za-z\-\/]+)\s+To\s*(?:Date)?[\s\:\-]*([0-9A-Za-z\-\/]+)",
        r"Period\s*From[\s\:\-]*([0-9A-Za-z\-\/]+)\s+To[\s\:\-]*([0-9A-Za-z\-\/]+)",
    ]

    for pat in period_patterns:
        m = re.search(pat, full_text[:4000], re.IGNORECASE)
        if m:
            d1 = parse_date_flexible(m.group(1))
            d2 = parse_date_flexible(m.group(2))
            if d1 and d2 and d2 >= d1:
                header_start_date = d1
                header_end_date = d2
                break

    # Transaction date range
    txn_start_date = min(all_dates) if all_dates else None
    txn_end_date = max(all_dates) if all_dates else None

    # Reconcile start and end dates
    effective_start = header_start_date or txn_start_date
    effective_end = header_end_date or txn_end_date

    if not effective_start or not effective_end:
        return {
            "start_date": None,
            "end_date": None,
            "days_covered": 0,
            "months_covered": 0.0,
            "has_6_months": False,
            "required_months": min_months_required,
            "shortfall_days": min_months_required * 30,
            "status_message": "Could not determine statement dates from document."
        }

    days_covered = (effective_end - effective_start).days
    months_covered = round(days_covered / 30.417, 1)

    # 6-month check: 180 days is standard industry threshold for 6 months
    required_days = int(min_months_required * 30)
    has_required_period = days_covered >= required_days or months_covered >= float(min_months_required)
    shortfall = max(0, required_days - days_covered)

    if has_required_period:
        status_msg = f"Statement satisfies the {min_months_required}-month requirement (covers {days_covered} days, ~{months_covered} months)."
    else:
        status_msg = (
            f"Statement covers {days_covered} days (~{months_covered} months). "
            f"The {min_months_required}-month requirement is NOT met (short by {shortfall} days)."
        )

    return {
        "start_date": effective_start.strftime("%Y-%m-%d"),
        "end_date": effective_end.strftime("%Y-%m-%d"),
        "start_date_display": effective_start.strftime("%d-%b-%Y"),
        "end_date_display": effective_end.strftime("%d-%b-%Y"),
        "days_covered": days_covered,
        "months_covered": months_covered,
        "has_6_months": has_required_period,
        "required_months": min_months_required,
        "shortfall_days": shortfall,
        "status_message": status_msg
    }


def extract_balances(full_text: str) -> Dict[str, Any]:
    """Extracts opening and closing balance if stated."""
    opening_bal = None
    closing_bal = None

    open_match = re.search(r"Opening\s*Balance[\s\:\-]*([0-9\,]+\.[0-9]{2})", full_text, re.IGNORECASE)
    if not open_match:
        open_match = re.search(r"Opening\s*Balance\s*\n\s*([0-9\,]+\.[0-9]{2})", full_text, re.IGNORECASE)
    if open_match:
        opening_bal = open_match.group(1).replace(",", "")

    close_match = re.search(r"Closing\s*Balance[\s\:\-]*([0-9\,]+\.[0-9]{2})", full_text, re.IGNORECASE)
    if not close_match:
        close_match = re.search(r"Closing\s*Balance\s*\n\s*([0-9\,]+\.[0-9]{2})", full_text, re.IGNORECASE)
    if close_match:
        closing_bal = close_match.group(1).replace(",", "")

    return {
        "opening_balance": float(opening_bal) if opening_bal else None,
        "closing_balance": float(closing_bal) if closing_bal else None,
    }


# ─── Main Orchestrator ─────────────────────────────────────────────────────────

def read_bank_statement(file_path: str, min_months: int = 6) -> Dict[str, Any]:
    """
    Parses a bank statement document (PDF or Image) and extracts:
      - bank_name
      - account_number & masking status
      - account_holder_name
      - ifsc_code & branch details
      - statement_period (start date, end date, days, months, has_6_months)
      - balances (opening & closing balance)
      - total_pages & transaction count
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Bank statement file not found: {file_path}")

    ext = Path(file_path).suffix.lower()
    pages_text: List[str] = []
    all_dates: List[datetime] = []

    if ext == ".pdf":
        doc = fitz.open(file_path)
        total_pages = len(doc)
        for page in doc:
            pt = page.get_text()
            pages_text.append(pt)
            raw_dates = re.findall(r"\b(\d{2}[-\/\.]\d{2}[-\/\.]\d{4}|\d{2}-[A-Za-z]{3}-\d{4})\b", pt)
            for d_str in raw_dates:
                dt = parse_date_flexible(d_str)
                if dt and 2000 <= dt.year <= 2035:
                    all_dates.append(dt)
        doc.close()
        doc_type = "pdf"
    else:
        from cheque_parser import run_ocr
        import cv2
        img = cv2.imread(file_path)
        if img is not None:
            blocks = run_ocr(img)
            text = "\n".join(b.text for b in blocks)
            pages_text = [text]
            raw_dates = re.findall(r"\b(\d{2}[-\/\.]\d{2}[-\/\.]\d{4}|\d{2}-[A-Za-z]{3}-\d{4})\b", text)
            for d_str in raw_dates:
                dt = parse_date_flexible(d_str)
                if dt and 2000 <= dt.year <= 2035:
                    all_dates.append(dt)
        total_pages = 1
        doc_type = "image"

    full_text = "\n".join(pages_text)
    first_page_text = pages_text[0] if pages_text else ""

    if not full_text.strip():
        return {
            "success": False,
            "error": "No readable text found in document. Ensure the PDF is not password protected or corrupted.",
            "doc_type": doc_type,
            "total_pages": total_pages
        }

    branch_info = extract_ifsc_and_branch(full_text, first_page_text)
    bank_name, bank_conf = extract_bank_name(full_text, branch_info.get("ifsc_code"))
    acc_info = extract_account_number(full_text)
    holder_name = extract_account_holder_name(full_text, first_page_text)
    period_info = extract_statement_period(full_text, all_dates, min_months_required=min_months)
    balances = extract_balances(full_text)

    txn_count = len(re.findall(r"\b\d{2}[-\/\.]\d{2}[-\/\.]\d{4}\b", full_text))

    return {
        "success": True,
        "doc_type": doc_type,
        "total_pages": total_pages,
        "bank_name": bank_name,
        "account_number": acc_info["account_number"],
        "is_masked": acc_info["is_masked"],
        "masked_account_number": acc_info["masked_account_number"],
        "unmasked_account_number": acc_info["unmasked_account_number"],
        "last_4_digits": acc_info["last_4_digits"],
        "account_holder_name": holder_name,
        "ifsc_code": branch_info["ifsc_code"],
        "branch_code": branch_info["branch_code"],
        "branch_name": branch_info["branch_name"],
        "branch_address": branch_info["branch_address"],
        "statement_period": period_info,
        "has_6_months": period_info["has_6_months"],
        "months_covered": period_info["months_covered"],
        "days_covered": period_info["days_covered"],
        "balances": balances,
        "transactions_detected": txn_count
    }

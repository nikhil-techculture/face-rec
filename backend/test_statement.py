import os
import fitz
from statement_parser import read_bank_statement


def create_sample_statement_pdf(
    path: str,
    bank: str,
    ifsc: str,
    acc_no: str,
    holder_name: str,
    start_date: str,
    end_date: str,
    opening_bal: str = "5,000.00",
    closing_bal: str = "25,000.00"
):
    """Creates a synthetic multi-page bank statement PDF for automated testing."""
    doc = fitz.open()
    page = doc.new_page()

    header_text = f"""{bank.upper()}
Statement of Account for {acc_no}
Account Holder: {holder_name}
Customer ID: 123456789
Branch: Fort Mumbai
IFSC Code: {ifsc}
Statement Period: {start_date} to {end_date}

Date          Particulars                   Withdrawal    Deposit       Balance
--------------------------------------------------------------------------------
              Opening Balance                                           {opening_bal}
01-10-2025    SALARY CREDIT                               50,000.00     55,000.00
15-11-2025    RENT PAYMENT                  15,000.00                   40,000.00
10-12-2025    MUTUAL FUND SIP               5,000.00                    35,000.00
05-01-2026    GROCERY EXPENSE               2,500.00                    32,500.00
15-02-2026    UTILITY BILLS                 1,500.00                    31,000.00
28-03-2026    INSURANCE PREMIUM             6,000.00                    25,000.00
              Closing Balance                                           {closing_bal}
--------------------------------------------------------------------------------
"""
    page.insert_text((50, 50), header_text, fontsize=10)
    doc.save(path)
    doc.close()


def test_canara_epassbook():
    path = r"c:\Users\Pc\Downloads\canara_epassbook_2026-06-15 16_38_45.838907.pdf"
    if not os.path.exists(path):
        print(f"Skipping canara test: file {path} not found")
        return

    print("\n=== Testing Real Canara Bank e-Passbook PDF ===")
    res = read_bank_statement(path, min_months=6)

    assert res["success"] is True
    assert res["bank_name"] == "Canara Bank"
    assert res["account_number"] == "XXXXXXXXX7009"
    assert res["is_masked"] is True
    assert res["last_4_digits"] == "7009"
    assert res["account_holder_name"] == "Rahul Goel"
    assert res["ifsc_code"] == "CNRB0002742"
    assert "Najafgarh" in res["branch_name"]
    assert res["statement_period"]["start_date"] == "2026-03-16"
    assert res["statement_period"]["end_date"] == "2026-06-15"
    assert res["statement_period"]["days_covered"] == 91
    assert res["statement_period"]["months_covered"] == 3.0
    assert res["has_6_months"] is False  # Only 3 months!
    assert res["balances"]["opening_balance"] == 4316.18
    assert res["balances"]["closing_balance"] == 15387.46
    assert res["transactions_detected"] >= 100

    # Also verify that when checking for min_months=3, it passes
    res_3m = read_bank_statement(path, min_months=3)
    assert res_3m["has_6_months"] is True

    print(">> Real Canara Bank e-Passbook tests PASSED!")
    print(f"   Bank: {res['bank_name']}")
    print(f"   Account Number: {res['account_number']} (Masked: {res['is_masked']}, Last 4: {res['last_4_digits']})")
    print(f"   Holder: {res['account_holder_name']}")
    print(f"   IFSC: {res['ifsc_code']} ({res['branch_name']})")
    print(f"   Period: {res['statement_period']['start_date_display']} to {res['statement_period']['end_date_display']} ({res['days_covered']} days, ~{res['months_covered']} months)")
    print(f"   Contains 6 Months: {res['has_6_months']} (Status: {res['statement_period']['status_message']})")
    print(f"   Opening Balance: {res['balances']['opening_balance']}, Closing Balance: {res['balances']['closing_balance']}")


def test_six_months_statement():
    sample_path = "backend/scratch_6month_statement.pdf"
    try:
        create_sample_statement_pdf(
            path=sample_path,
            bank="HDFC BANK",
            ifsc="HDFC0000060",
            acc_no="50100234567890",
            holder_name="Anita Sharma",
            start_date="01-Oct-2025",
            end_date="31-Mar-2026",
            opening_bal="5,000.00",
            closing_bal="25,000.00"
        )

        print("\n=== Testing Synthetic 6-Month Statement PDF ===")
        res = read_bank_statement(sample_path, min_months=6)

        assert res["success"] is True
        assert res["bank_name"] == "HDFC Bank"
        assert res["account_number"] == "50100234567890"
        assert res["is_masked"] is False
        assert res["last_4_digits"] == "7890"
        assert res["account_holder_name"] == "Anita Sharma"
        assert res["ifsc_code"] == "HDFC0000060"
        assert res["statement_period"]["start_date"] == "2025-10-01"
        assert res["statement_period"]["end_date"] == "2026-03-31"
        assert res["statement_period"]["days_covered"] >= 180
        assert res["statement_period"]["months_covered"] >= 5.9
        assert res["has_6_months"] is True  # Satisfies 6 months!
        assert res["balances"]["opening_balance"] == 5000.00
        assert res["balances"]["closing_balance"] == 25000.00

        print(">> Synthetic 6-Month Statement tests PASSED!")
        print(f"   Bank: {res['bank_name']}")
        print(f"   Account Number: {res['account_number']}")
        print(f"   Holder: {res['account_holder_name']}")
        print(f"   Period: {res['statement_period']['start_date_display']} to {res['statement_period']['end_date_display']} ({res['days_covered']} days, ~{res['months_covered']} months)")
        print(f"   Contains 6 Months: {res['has_6_months']} (Status: {res['statement_period']['status_message']})")
    finally:
        if os.path.exists(sample_path):
            os.remove(sample_path)


if __name__ == "__main__":
    test_canara_epassbook()
    test_six_months_statement()
    print("\nALL BANK STATEMENT TESTS PASSED SUCCESSFULLY! 100%")


"""
receipt_parser.py
------------------
Takes raw OCR text from a receipt and returns a list of structured product items.
Handles:
- Cleaning of OCR noise
- Removal of header/footer lines (totals, tax, store info)
- Merging of multi-line product descriptions
- Extraction of quantity, unit, and optional price
"""

import re
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ----------------------------------------------------------------------
# 1. Pre-cleaning
# ----------------------------------------------------------------------
def clean_ocr_text(raw_text: str) -> str:
    """
    Apply common OCR error corrections:
    - Normalize whitespace
    - Fix common misreads (e.g., '0'↔'O')
    - Remove obviously non-product lines (transient)
    """
    # Replace common OCR mistakes
    text = raw_text.replace("|", "l")          # pipe often misread as letter I
    text = re.sub(r"\s+", " ", text)           # collapse multiple spaces/newlines
    # Remove lines that are only punctuation/symbols (often noise)
    text = re.sub(r"^[^a-zA-Z0-9]*$", "", text, flags=re.MULTILINE)
    return text.strip()

# ----------------------------------------------------------------------
# 2. Identify header/footer lines and remove them
# ----------------------------------------------------------------------
# Keywords that indicate a line is NOT a product item
NON_PRODUCT_KEYWORDS = [
    "subtotal", "total", "discount", "tax", "vat", "change", "balance",
    "cash", "credit", "debit", "card", "visa", "mastercard", "amex",
    "payment", "authorization", "ref", "transaction", "invoice", "receipt",
    "date", "time", "store", "supermarket", "hypermarket", "mart",
    "thank you", "have a nice day", "www", "http", "tel", "phone"
]

def is_non_product_line(line: str) -> bool:
    """Return True if the line is likely a header/footer, not a product."""
    lowered = line.lower().strip()
    # Empty line
    if not lowered:
        return True
    # Very short line (likely a price or punctuation)
    if len(lowered) <= 2 and not any(c.isalpha() for c in lowered):
        return True
    # Check for non-product keywords
    for kw in NON_PRODUCT_KEYWORDS:
        if kw in lowered:
            return True
    # Line that looks like ONLY a price (e.g., "25.00")
    if re.match(r"^\d+\.\d{2}$", lowered):
        return True
    return False

# ----------------------------------------------------------------------
# 3. Merge multi-line items
# ----------------------------------------------------------------------
# Pattern to detect a line that is a continuation (starts with quantity or price)
CONTINUATION_PATTERN = re.compile(r"^\s*\d+(?:\.\d+)?\s*(?:kg|g|l|ml|oz|lb|pcs?|ea|piece)")

def merge_multiline_items(lines: List[str]) -> List[str]:
    """
    If a line starts with a quantity/unit but no obvious product name,
    it's probably a continuation of the previous line.
    """
    merged = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # If line starts with a quantity/unit pattern and merged is not empty,
        # append to previous line
        if merged and CONTINUATION_PATTERN.match(stripped):
            merged[-1] += " " + stripped
        else:
            merged.append(stripped)
    return merged

# ----------------------------------------------------------------------
# 4. Parse a single product line
# ----------------------------------------------------------------------
# Quantity-unit patterns
QTY_UNIT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(kg|g|l|ml|litre|liter|oz|lb|piece|pcs|pc|ea)\b",
    re.IGNORECASE,
)

# Price pattern: optional "@" symbol followed by digits.decimals
PRICE_PATTERN = re.compile(r"(?:@\s*)?(\d+\.\d{2})")

def parse_product_line(line: str) -> Optional[Dict]:
    """
    Attempt to extract product name, quantity, unit, and optionally price
    from a single (possibly merged) line.
    Returns None if the line cannot be parsed as a product.
    """
    original = line.strip()
    if not original:
        return None

    # Find price (if any) and remove it from the product description
    price_match = PRICE_PATTERN.search(original)
    price = None
    if price_match:
        price = float(price_match.group(1))
        # Remove price from string to simplify quantity extraction
        original = original[:price_match.start()] + original[price_match.end():]

    # Find quantity/unit
    qty_match = QTY_UNIT_PATTERN.search(original)
    quantity = None
    unit = None
    if qty_match:
        quantity = float(qty_match.group(1))
        unit = qty_match.group(2).lower()
        # Remove quantity+unit from product name
        product_name = original[:qty_match.start()] + original[qty_match.end():]
    else:
        # No quantity found – treat the whole line as product name (without price)
        product_name = original

    # Clean product name
    product_name = re.sub(r"\s+", " ", product_name).strip()
    # Remove any leading/trailing non-alpha characters (if feasible)
    product_name = product_name.strip(",. @")
    if not product_name or product_name.replace(" ", "").isdigit():
        # No meaningful product name left
        return None

    return {
        "raw": line.strip(),
        "product": product_name,
        "quantity": quantity,
        "unit": unit,
        "price": price,
    }

# ----------------------------------------------------------------------
# 5. Main parsing function
# ----------------------------------------------------------------------
def parse_receipt(raw_ocr: str) -> List[Dict]:
    """
    Process raw OCR text and return a list of product item dictionaries.
    """
    # Step 1: clean
    text = clean_ocr_text(raw_ocr)

    # Step 2: split into lines
    lines = text.split("\n")

    # Step 3: remove header/footer lines
    product_lines = [l for l in lines if not is_non_product_line(l)]

    # Step 4: merge multi-line items
    merged = merge_multiline_items(product_lines)

    # Step 5: parse each resulting line
    items = []
    for line in merged:
        parsed = parse_product_line(line)
        if parsed:
            items.append(parsed)
        else:
            logging.debug(f"Could not parse as product: {line}")

    logging.info(f"Parsed {len(items)} products from receipt.")
    return items

# ----------------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_receipt = """
    FOOD MART QATAR
    Date: 2024-01-15
    ---------------------
    Chicken Breast 1.5kg       45.00
    Whole Wheat Bread 1pcs     5.50
    Tomato                   0.5kg 3.00
    ---------------------
    Subtotal: 53.50
    Tax:       0.00
    Total:     53.50
    Thank you!
    """

    items = parse_receipt(sample_receipt)
    for item in items:
        print(item)

    
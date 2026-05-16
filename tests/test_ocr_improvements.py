"""
QUICK START: Test OCR Improvements
===================================

This script demonstrates the changes made to reduce text boldness
and adds OCR engine selection.
"""

# Test 1: Process receipt with reduced-boldness preprocessing
# ============================================================

print("=" * 70)
print("TEST 1: Process Receipt with Reduced Text Boldness")
print("=" * 70)

from src.ocr import process_all_receipts, preprocess_receipt, extract_text
from pathlib import Path
import cv2

# Process first 3 receipts to see the improvement
results = process_all_receipts(
    debug=True,
    max_files=3  # Test on first 3 images
)

print("\n✓ Receipts processed with reduced text boldness")
print("  Check: data/ocr_output/ for extracted text")
print("  Check: data/receipts/preprocessed/ for preprocessed images")


# Test 2: Compare OCR engines (Tesseract vs EasyOCR)
# ==================================================

print("\n" + "=" * 70)
print("TEST 2: OCR Engine Selection")
print("=" * 70)

from src.ocr import extract_text_tesseract, extract_text_easyocr, load_image

# Load a sample receipt
receipt_path = Path("data/receipts/raw")
if receipt_path.exists():
    sample_image = list(receipt_path.glob("*.jpg"))[0] if list(receipt_path.glob("*.jpg")) else None
    
    if sample_image:
        img = load_image(str(sample_image))
        preprocessed = preprocess_receipt(img)
        
        print(f"\nTesting on: {sample_image.name}")
        
        # Test Tesseract
        print("\n[TESSERACT]")
        text_tess = extract_text_tesseract(preprocessed)
        print(f"  Extracted {len(text_tess)} characters")
        print(f"  First 100 chars: {text_tess[:100]}...")
        
        # Test EasyOCR (if available)
        print("\n[EASYOCR]")
        try:
            text_easy = extract_text_easyocr(preprocessed)
            print(f"  Extracted {len(text_easy)} characters")
            print(f"  First 100 chars: {text_easy[:100]}...")
        except ImportError:
            print("  ⚠ EasyOCR not installed")
            print("  Install with: pip install easyocr torch")


# Test 3: Configure OCR engine
# ==============================

print("\n" + "=" * 70)
print("TEST 3: Configure Preferred OCR Engine")
print("=" * 70)

print("""
Option A: Use Tesseract (default, fast)
    No changes needed - already configured

Option B: Use EasyOCR (better accuracy, slower)
    Edit src/ocr_config.py:
    
    OCR_ENGINE = "easyocr"
    USE_EASYOCR = True
    
    Then install: pip install easyocr torch

Option C: Train custom model on SROIE dataset
    1. Download SROIE from: https://www.kaggle.com/datasets/urbikn/sroie-datasetv2
    2. Extract to: data/sroie/
    3. Install: pip install paddleocr paddlepaddle
    4. Train: python src/train_ocr.py --dataset_dir data/sroie/ --epochs 50
    
    See: OCR_TRAINING_GUIDE.md for detailed instructions
""")


# Test 4: Receipt entity parsing
# ===============================

print("\n" + "=" * 70)
print("TEST 4: Receipt Entity Parsing (New Feature)")
print("=" * 70)

from src.ocr import parse_receipt_entities

sample_text = """
WHOLE FOODS MARKET
123 Main Street
New York, NY 10001

Date: 2024-01-15
Time: 14:32

Organic Banana 2.5kg        12.50
Almond Milk 1L              4.99
Wild Salmon 0.5kg           18.75
Olive Oil 750ml             8.99

Subtotal:                   45.23
Tax (8.5%):                  3.84
Total:                      49.07

Thank you for shopping!
"""

entities = parse_receipt_entities(sample_text)

print(f"  Merchant: {entities['merchant']}")
print(f"  Date: {entities['date']}")
print(f"  Total: ${entities['total']}")
print(f"  Items found: {len(entities['items'])}")
if entities['items']:
    for item in entities['items'][:3]:
        print(f"    - {item['product']} ({item['quantity']} {item['unit']}) ${item['price']}")


# Test 5: Morphology improvement
# ===============================

print("\n" + "=" * 70)
print("TEST 5: Morphology Operation Improvement")
print("=" * 70)

print("""
BEFORE (Old Implementation):
  - Kernel size: 3x3
  - Operations: OPEN + CLOSE (2 iterations)
  - Result: Text too thick/bold

AFTER (New Implementation):
  - Kernel size: 2x2
  - Light mode: OPEN only (1 iteration)
  - Aggressive mode: OPEN + CLOSE (1 iteration each)
  - Result: Natural text thickness, better OCR accuracy
  
See denoise_and_morphology() in src/ocr.py
""")


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
✓ Text boldness reduced (morphology operations optimized)
✓ Dual OCR engine support (Tesseract + EasyOCR)
✓ Receipt entity parsing (merchant, date, total, items)
✓ Training infrastructure ready (train_ocr.py)
✓ Complete training guide available (OCR_TRAINING_GUIDE.md)

NEXT STEPS:
1. Test OCR output with reduced boldness
2. Optional: Switch to EasyOCR for better accuracy
3. Advanced: Train custom model on SROIE dataset

For detailed instructions, see:
  - OCR_UPDATES_SUMMARY.md
  - OCR_TRAINING_GUIDE.md
""")

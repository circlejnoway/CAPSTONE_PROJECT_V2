"""
compare_ocr_methods.py
----------------------
Compare different OCR preprocessing methods to find the best approach.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

# Import our OCR functions
from src.ocr import (
    load_image, preprocess_receipt, extract_text,
    gamma_correction, enhance_contrast_clahe, denoise_bilateral,
    adaptive_threshold_image, denoise_morphological, scale_image_optimal
)

def old_preprocessing_pipeline(image: np.ndarray) -> np.ndarray:
    """
    Original preprocessing pipeline for comparison.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # CLAHE enhancement
    enhanced = enhance_contrast_clahe(gray, clip_limit=2.5)

    # Bilateral denoising
    denoised = denoise_bilateral(enhanced)

    # Adaptive thresholding
    binary = adaptive_threshold_image(denoised)

    # Morphological cleanup
    cleaned = denoise_morphological(binary)

    # Scale
    scaled = scale_image_optimal(cleaned)

    return scaled

def test_preprocessing_methods(image_path: str) -> Dict:
    """
    Test different preprocessing methods on a single image.
    Returns comparison results.
    """
    print(f"\nTesting preprocessing methods on: {Path(image_path).name}")
    print("="*60)

    # Load image
    image = load_image(image_path)

    results = {}

    # Method 1: Old pipeline
    print("Testing: Old pipeline...")
    start_time = time.time()
    old_processed = old_preprocessing_pipeline(image)
    old_text = extract_text(old_processed)
    old_time = time.time() - start_time
    results['old'] = {
        'text': old_text,
        'time': old_time,
        'chars': len(old_text),
        'letters': len(''.join(c for c in old_text if c.isalpha()))
    }
    print(".2f")

    # Method 2: New aggressive pipeline
    print("Testing: New aggressive pipeline...")
    start_time = time.time()
    new_processed = preprocess_receipt(image, debug=False, aggressive=True)
    new_text = extract_text(new_processed)
    new_time = time.time() - start_time
    results['new_aggressive'] = {
        'text': new_text,
        'time': new_time,
        'chars': len(new_text),
        'letters': len(''.join(c for c in new_text if c.isalpha()))
    }
    print(".2f")

    # Method 3: New standard pipeline
    print("Testing: New standard pipeline...")
    start_time = time.time()
    std_processed = preprocess_receipt(image, debug=False, aggressive=False)
    std_text = extract_text(std_processed)
    std_time = time.time() - start_time
    results['new_standard'] = {
        'text': std_text,
        'time': std_time,
        'chars': len(std_text),
        'letters': len(''.join(c for c in std_text if c.isalpha()))
    }
    print(".2f")

    return results

def print_comparison(results: Dict):
    """Print detailed comparison of results."""
    print("\n" + "="*80)
    print("PREPROCESSING METHOD COMPARISON")
    print("="*80)

    methods = {
        'old': 'Old Pipeline',
        'new_aggressive': 'New Aggressive',
        'new_standard': 'New Standard'
    }

    # Header
    print("<20")
    print("-"*80)

    # Results
    for key, method_name in methods.items():
        if key in results:
            data = results[key]
            print("<20")

    print("-"*80)

    # Best method analysis
    best_letters = max(results[method]['letters'] for method in results.keys())
    best_method = [m for m in results.keys() if results[m]['letters'] == best_letters][0]

    print(f"\n🏆 BEST METHOD: {methods[best_method]}")
    print(f"   Extracted {best_letters} letters")
    print(".2f")

    # Show sample text from best method
    best_text = results[best_method]['text']
    preview = best_text[:200] + "..." if len(best_text) > 200 else best_text
    print(f"\nSample text from best method:\n{preview}")

def batch_compare_methods(num_samples: int = 5):
    """Compare methods on multiple receipt images."""
    raw_dir = Path("data/receipts/raw")
    if not raw_dir.exists():
        print(f"ERROR: Raw receipts directory not found: {raw_dir}")
        return

    # Get sample images
    image_files = list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png"))
    image_files = image_files[:num_samples]

    if not image_files:
        print("ERROR: No image files found in data/receipts/raw/")
        return

    print(f"Comparing OCR methods on {len(image_files)} receipt images...")
    print("="*80)

    all_results = {}
    total_old_letters = 0
    total_new_aggressive_letters = 0
    total_new_standard_letters = 0

    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing {img_path.name}")
        try:
            results = test_preprocessing_methods(str(img_path))

            # Accumulate totals
            total_old_letters += results['old']['letters']
            total_new_aggressive_letters += results['new_aggressive']['letters']
            total_new_standard_letters += results['new_standard']['letters']

            all_results[img_path.name] = results

        except Exception as e:
            print(f"ERROR processing {img_path.name}: {e}")
            continue

    # Overall summary
    print("\n" + "="*80)
    print("OVERALL BATCH RESULTS")
    print("="*80)

    avg_old = total_old_letters / len(all_results)
    avg_new_aggressive = total_new_aggressive_letters / len(all_results)
    avg_new_standard = total_new_standard_letters / len(all_results)

    print("<20")
    print("-"*80)
    print("<20")

    improvement_aggressive = ((avg_new_aggressive - avg_old) / avg_old) * 100
    improvement_standard = ((avg_new_standard - avg_old) / avg_old) * 100

    print(".1f")
    print(".1f")

    if improvement_aggressive > improvement_standard:
        print("🏆 RECOMMENDED: New Aggressive Pipeline")
    else:
        print("🏆 RECOMMENDED: New Standard Pipeline")

def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare OCR preprocessing methods")
    parser.add_argument('--method', choices=['single', 'batch'], default='batch',
                       help='Comparison method')
    parser.add_argument('--file', type=str,
                       help='Image file for single comparison')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of samples for batch comparison')

    args = parser.parse_args()

    if args.method == 'single':
        if not args.file:
            print("ERROR: --file required for single method comparison")
            sys.exit(1)

        results = test_preprocessing_methods(args.file)
        print_comparison(results)

    elif args.method == 'batch':
        batch_compare_methods(num_samples=args.samples)

if __name__ == "__main__":
    main()
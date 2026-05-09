"""
ocr_quality_analyzer.py
------------------------
Comprehensive utility to analyze, test, and validate OCR improvements.
Provides visualization and detailed metrics for OCR quality assessment.

USAGE:
  python ocr_quality_analyzer.py --test batch --samples 10
  python ocr_quality_analyzer.py --test comparison --file data/receipts/raw/image.jpg
  python ocr_quality_analyzer.py --test confidence --file data/receipts/raw/image.jpg
"""

import sys
import re
import time
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ============================================================================
# QUALITY METRICS
# ============================================================================

def count_text_metrics(text: str) -> Dict:
    """Analyze extracted text and return quality metrics."""
    metrics = {
        'total_chars': len(text),
        'letters': len(re.sub(r'[^a-zA-Z]', '', text)),
        'digits': len(re.sub(r'[^0-9]', '', text)),
        'numbers': len(re.findall(r'\d+\.?\d*', text)),
        'lines': len(text.split('\n')),
        'words': len(text.split()),
        'avg_word_length': 0,
        'garbled_ratio': 0,
    }
    
    if metrics['words'] > 0:
        metrics['avg_word_length'] = metrics['total_chars'] / metrics['words']
    
    # Estimate garbled text
    garbled_count = 0
    for word in text.split():
        word_clean = re.sub(r'[^a-zA-Z]', '', word)
        if word_clean and len(word_clean) > 1:
            if len(set(word_clean)) < len(word_clean) * 0.3:
                garbled_count += 1
    
    total_words = len([w for w in text.split() if w])
    if total_words > 0:
        metrics['garbled_ratio'] = garbled_count / total_words
    
    return metrics

def print_metrics(filename: str, text: str, elapsed_time: float = 0):
    """Pretty print OCR metrics."""
    metrics = count_text_metrics(text)
    
    print(f"\n{'='*70}")
    print(f"File: {filename}")
    print(f"{'='*70}")
    print(f"Total characters:     {metrics['total_chars']:>6}")
    print(f"Letters:              {metrics['letters']:>6}")
    print(f"Digits:               {metrics['digits']:>6}")
    print(f"Numbers found:        {metrics['numbers']:>6}")
    print(f"Words:                {metrics['words']:>6}")
    print(f"Lines:                {metrics['lines']:>6}")
    print(f"Avg word length:      {metrics['avg_word_length']:>6.1f}")
    print(f"Garbled ratio:        {metrics['garbled_ratio']:>6.1%}")
    if elapsed_time > 0:
        print(f"Processing time:      {elapsed_time:>6.2f}s")
    print(f"{'='*70}")
    
    return metrics

def compare_ocr_results(text_before: str, text_after: str) -> Dict:
    """Compare OCR results before and after improvements."""
    metrics_before = count_text_metrics(text_before)
    metrics_after = count_text_metrics(text_after)
    
    improvements = {
        'char_improvement': metrics_after['letters'] - metrics_before['letters'],
        'word_improvement': metrics_after['words'] - metrics_before['words'],
        'char_pct': ((metrics_after['letters'] - metrics_before['letters']) / max(1, metrics_before['letters'])) * 100,
        'garbled_reduction': metrics_before['garbled_ratio'] - metrics_after['garbled_ratio'],
    }
    
    print(f"\n{'─'*70}")
    print("BEFORE vs AFTER Comparison")
    print(f"{'─'*70}")
    print(f"Letters extracted:    {metrics_before['letters']:>6} → {metrics_after['letters']:>6} ({improvements['char_pct']:+.1f}%)")
    print(f"Words extracted:      {metrics_before['words']:>6} → {metrics_after['words']:>6}")
    print(f"Garbled ratio:        {metrics_before['garbled_ratio']:>6.1%} → {metrics_after['garbled_ratio']:>6.1%}")
    print(f"{'─'*70}")
    
    return improvements

# ============================================================================
# VISUAL COMPARISON
# ============================================================================

def show_preprocessing_stages(image_path: str, save_path: str = "preprocessing_stages.png"):
    """
    Create side-by-side visualization of all preprocessing stages.
    Requires the improved ocr.py module.
    """
    try:
        from src.ocr import (
            load_image, 
            enhance_contrast_clahe,
            denoise_bilateral,
            deskew_image,
            estimate_rotation_angle,
            adaptive_threshold_image,
            denoise_morphological,
            scale_image_optimal
        )
    except ImportError:
        print("ERROR: Could not import improved OCR functions from src.ocr")
        return
    
    print(f"\nCreating preprocessing visualization...")
    
    # Load image
    try:
        original = load_image(image_path)
        print(f"✓ Loaded image: {original.shape}")
    except Exception as e:
        print(f"ERROR: Could not load image: {e}")
        return
    
    try:
        # Ensure grayscale for processing
        if len(original.shape) == 3 and original.shape[2] == 3:
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            gray = original if len(original.shape) == 2 else cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        
        # Create processing stages
        stage1_gray = gray
        stage2_clahe = enhance_contrast_clahe(gray, clip_limit=2.5)
        stage3_denoised = denoise_bilateral(stage2_clahe)
        angle = estimate_rotation_angle(stage3_denoised)
        stage4_deskewed = deskew_image(stage3_denoised, angle)
        stage5_thresh = adaptive_threshold_image(stage4_deskewed)
        stage6_morphed = denoise_morphological(stage5_thresh)
        stage7_scaled = scale_image_optimal(stage6_morphed)
        
        # Normalize sizes for grid
        target_height = 300
        
        stages = [
            ("1. Original Gray", stage1_gray),
            ("2. CLAHE Enhanced", stage2_clahe),
            ("3. Bilateral Denoise", stage3_denoised),
            ("4. Deskewed", stage4_deskewed),
            ("5. Adaptive Threshold", stage5_thresh),
            ("6. Morphological", stage6_morphed),
            ("7. Scaled (1200px)", stage7_scaled),
        ]
        
        processed_stages = []
        for label, stage_img in stages:
            h, w = stage_img.shape[:2]
            scale = target_height / h
            new_w = int(w * scale)
            resized = cv2.resize(stage_img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
            
            # Convert to BGR and add label
            if len(resized.shape) == 2:
                labeled = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            else:
                labeled = resized.copy()
            
            cv2.putText(labeled, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            processed_stages.append(labeled)
        
        # Create 2x4 grid layout
        row1 = np.hstack(processed_stages[:4])
        row2 = np.hstack(processed_stages[4:])
        comparison = np.vstack([row1, row2])
        
        # Save
        cv2.imwrite(save_path, comparison)
        print(f"✓ Saved visualization to: {save_path}")
        print(f"  Image size: {comparison.shape[1]}x{comparison.shape[0]} pixels")
    
    except Exception as e:
        print(f"ERROR during processing: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# BATCH TESTING
# ============================================================================

def test_sample_receipts(num_samples: int = 5):
    """Test OCR on sample receipts and display metrics."""
    try:
        from src.ocr import process_all_receipts
    except ImportError:
        print("ERROR: Could not import process_all_receipts from src.ocr")
        return
    
    raw_dir = Path("data/receipts/raw")
    if not raw_dir.exists():
        print(f"ERROR: Raw receipts directory not found: {raw_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"OCR BATCH TEST - Processing {num_samples} sample receipts")
    print(f"{'='*70}\n")
    
    start_total = time.time()
    results = process_all_receipts(debug=False, max_files=num_samples)
    total_time = time.time() - start_total
    
    all_metrics = []
    for i, (filename, text) in enumerate(list(results.items())[:num_samples], 1):
        metrics = count_text_metrics(text)
        all_metrics.append(metrics)
        
        # Show per-file summary
        status = "✓" if metrics['letters'] > 20 else "⚠"
        print(f"{status} [{i:2d}] {filename:40} | {metrics['letters']:>4} letters | {metrics['words']:>3} words")
    
    # Summary statistics
    if all_metrics:
        avg_letters = np.mean([m['letters'] for m in all_metrics])
        avg_words = np.mean([m['words'] for m in all_metrics])
        avg_garbled = np.mean([m['garbled_ratio'] for m in all_metrics])
        min_letters = min([m['letters'] for m in all_metrics])
        max_letters = max([m['letters'] for m in all_metrics])
        
        print(f"\n{'─'*70}")
        print("BATCH SUMMARY")
        print(f"{'─'*70}")
        print(f"Files processed:      {len(results)}")
        print(f"Total processing:     {total_time:.2f}s")
        print(f"Avg time per file:    {total_time/len(results):.2f}s")
        print(f"{'─'*70}")
        print(f"Letters per receipt:  min={min_letters:>3} | avg={avg_letters:>6.0f} | max={max_letters:>3}")
        print(f"Words per receipt:    avg={avg_words:>6.1f}")
        print(f"Garbled ratio:        avg={avg_garbled:>6.1%}")
        print(f"{'─'*70}\n")

# ============================================================================
# CONFIDENCE ANALYSIS
# ============================================================================

def analyze_confidence_scores(image_path: str):
    """Analyze Tesseract confidence scores for each word."""
    try:
        # Load image with error handling
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: Could not load image: {image_path}")
            return
        
        # Normalize to 3-channel BGR if needed
        if len(img.shape) == 2:
            # Grayscale - convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            # RGBA - remove alpha
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.shape[2] != 3:
            # Other formats - take first 3 channels
            img = img[:,:,:3]
        
        # Convert BGR to RGB for PIL
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        print(f"\nExtracting confidence data using Tesseract...")
        data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
        
        # Extract confidence for each word
        word_confidences = []
        for i, word in enumerate(data['text']):
            if word.strip():
                conf = int(data['conf'][i])
                word_confidences.append((word, conf))
        
        if not word_confidences:
            print("ERROR: No text detected in image")
            return
        
        # Sort by confidence
        word_confidences.sort(key=lambda x: x[1])
        
        print(f"\n{'='*70}")
        print(f"OCR CONFIDENCE ANALYSIS")
        print(f"File: {Path(image_path).name}")
        print(f"{'='*70}")
        
        # Lowest confidence (potential errors)
        print("\nLOWEST CONFIDENCE WORDS (potential OCR errors):")
        print(f"{'─'*70}")
        for word, conf in word_confidences[:15]:
            if conf < 30:
                status = "🔴 VERY LOW"
            elif conf < 60:
                status = "🟠 LOW"
            elif conf < 80:
                status = "🟡 MEDIUM"
            else:
                status = "🟢 HIGH"
            print(f"  {status}      {word:25} {conf:>3}% confidence")
        
        # Highest confidence
        print("\nHIGHEST CONFIDENCE WORDS (reliable extraction):")
        print(f"{'─'*70}")
        for word, conf in word_confidences[-10:]:
            print(f"  🟢 HIGH      {word:25} {conf:>3}% confidence")
        
        # Statistics
        confidences = [conf for _, conf in word_confidences]
        print(f"\n{'─'*70}")
        print("STATISTICS")
        print(f"{'─'*70}")
        print(f"Total words recognized:    {len(word_confidences):>6}")
        print(f"Average confidence:        {np.mean(confidences):>6.1f}%")
        print(f"Median confidence:         {np.median(confidences):>6.1f}%")
        print(f"Std deviation:             {np.std(confidences):>6.1f}%")
        print(f"Words < 50% confidence:    {sum(1 for c in confidences if c < 50):>6} ({sum(1 for c in confidences if c < 50)/len(confidences)*100:>5.1f}%)")
        print(f"Words < 75% confidence:    {sum(1 for c in confidences if c < 75):>6} ({sum(1 for c in confidences if c < 75)/len(confidences)*100:>5.1f}%)")
        print(f"Words >= 90% confidence:   {sum(1 for c in confidences if c >= 90):>6} ({sum(1 for c in confidences if c >= 90)/len(confidences)*100:>5.1f}%)")
        print(f"{'─'*70}\n")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run comprehensive OCR testing suite."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="OCR Quality Analyzer - Test and validate OCR preprocessing improvements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python ocr_quality_analyzer.py --test batch --samples 10
  python ocr_quality_analyzer.py --test comparison --file data/receipts/raw/receipt.jpg
  python ocr_quality_analyzer.py --test confidence --file data/receipts/raw/receipt.jpg
        """
    )
    
    parser.add_argument('--test', type=str, default='help', 
                       choices=['help', 'batch', 'comparison', 'confidence'],
                       help='Type of test to run')
    parser.add_argument('--file', type=str, help='Image file path (required for comparison and confidence tests)')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples for batch test (default: 5)')
    
    args = parser.parse_args()
    
    if args.test == 'help':
        print("\n" + "="*70)
        print("OCR QUALITY ANALYZER")
        print("="*70)
        print("\nAvailable Tests:\n")
        print("  batch")
        print("    Test OCR on multiple receipt files")
        print("    Usage: python ocr_quality_analyzer.py --test batch [--samples N]")
        print("    Example: python ocr_quality_analyzer.py --test batch --samples 10\n")
        print("  comparison")
        print("    Show preprocessing pipeline stages side-by-side")
        print("    Usage: python ocr_quality_analyzer.py --test comparison --file <image>")
        print("    Example: python ocr_quality_analyzer.py --test comparison --file data/receipts/raw/receipt.jpg")
        print("    Output: preprocessing_stages.png\n")
        print("  confidence")
        print("    Analyze Tesseract confidence scores per word")
        print("    Usage: python ocr_quality_analyzer.py --test confidence --file <image>")
        print("    Example: python ocr_quality_analyzer.py --test confidence --file data/receipts/raw/receipt.jpg\n")
        print("="*70 + "\n")
    
    elif args.test == 'comparison':
        if not args.file:
            print("ERROR: --file required for comparison test")
            print("Usage: python ocr_quality_analyzer.py --test comparison --file <image_path>")
            sys.exit(1)
        show_preprocessing_stages(args.file)
    
    elif args.test == 'confidence':
        if not args.file:
            print("ERROR: --file required for confidence test")
            print("Usage: python ocr_quality_analyzer.py --test confidence --file <image_path>")
            sys.exit(1)
        analyze_confidence_scores(args.file)
    
    elif args.test == 'batch':
        test_sample_receipts(num_samples=args.samples)

if __name__ == "__main__":
    main()

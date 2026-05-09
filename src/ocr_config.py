"""
ocr_config.py
-------------
Configuration file for OCR preprocessing parameters.
Modify these settings to optimize OCR performance for your receipt images.
"""

from pathlib import Path
import numpy as np
import cv2

# ============================================================================
# OCR CONFIGURATION
# ============================================================================


# Language settings
OCR_LANG = "eng"          # Language for Tesseract ('eng', 'eng+ara', etc.)

# Image processing parameters
MIN_IMAGE_WIDTH = 1200    # Minimum width for optimal Tesseract performance
MAX_IMAGE_WIDTH = 3200    # Maximum width (to avoid excessive processing)

# Quality thresholds
CONFIDENCE_THRESHOLD = 20  # Filter out text below this confidence score (%)

# ============================================================================
# PREPROCESSING MODES
# ============================================================================

# Choose preprocessing aggressiveness based on receipt quality
# Set to True for very noisy/low-quality receipts (slower but cleaner)
# Set to False for cleaner receipts (faster processing)
AGGRESSIVE_DENOISING = True

# ============================================================================
# ADVANCED TUNING PARAMETERS
# ============================================================================

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
CLAHE_CLIP_LIMIT = 2.5     # Higher values = more contrast (2.0-4.0 range)
CLAHE_TILE_SIZE = (8, 8)   # Local adaptation tile size

# Gamma correction (brightness adjustment)
GAMMA_CORRECTION = 1.2     # >1.0 = darker (better for bright receipts)

# Denoising parameters
MEDIAN_KERNEL_SIZE = 3      # Odd number (3, 5, 7) - larger = more smoothing
BILATERAL_SIGMA = 75        # Edge preservation strength
NL_MEANS_H = 10            # Non-local means denoising strength

# Thresholding parameters
ADAPTIVE_BLOCK_SIZE = 21    # Must be odd number
ADAPTIVE_C = 5             # Constant subtracted from mean

# Morphological operations
MORPH_KERNEL_SIZE = 3       # Size of morphological kernels

# Text sharpening
SHARPEN_STRENGTH = 0.5      # 0.0-1.0 range (higher = more sharpening)

# ============================================================================
# DEBUG AND LOGGING
# ============================================================================

# Enable detailed logging
DEBUG_MODE = True         # Save intermediate images and detailed logs
SAVE_PREPROCESSED = True   # Save final preprocessed images for inspection

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Processing limits
MAX_FILES_TO_PROCESS = 5  # None = process all, or set to number for testing
BATCH_SIZE = 10            # Process images in batches to manage memory

# ============================================================================
# PATHS (usually don't need to change)
# ============================================================================

RAW_RECEIPTS_DIR = Path("data/receipts/raw")
OUTPUT_TEXT_DIR = Path("data/ocr_output")
PREPROCESSED_DEBUG_DIR = Path("data/receipts/preprocessed")

# ============================================================================
# PRESETS FOR DIFFERENT RECEIPT TYPES
# ============================================================================

def set_preset_mobile_phone():
    """Preset for mobile phone receipt photos (common case)"""
    global AGGRESSIVE_DENOISING, CLAHE_CLIP_LIMIT, GAMMA_CORRECTION
    AGGRESSIVE_DENOISING = True
    CLAHE_CLIP_LIMIT = 3.0
    GAMMA_CORRECTION = 1.3
    print("✓ Set preset: Mobile phone photos")

def set_preset_professional_scan():
    """Preset for high-quality scanned receipts"""
    global AGGRESSIVE_DENOISING, CLAHE_CLIP_LIMIT, GAMMA_CORRECTION
    AGGRESSIVE_DENOISING = False
    CLAHE_CLIP_LIMIT = 2.0
    GAMMA_CORRECTION = 1.0
    print("✓ Set preset: Professional scans")

def set_preset_very_noisy():
    """Preset for extremely noisy/low-quality receipts"""
    global AGGRESSIVE_DENOISING, CLAHE_CLIP_LIMIT, GAMMA_CORRECTION, MEDIAN_KERNEL_SIZE
    AGGRESSIVE_DENOISING = True
    CLAHE_CLIP_LIMIT = 4.0
    GAMMA_CORRECTION = 1.5
    MEDIAN_KERNEL_SIZE = 5
    print("✓ Set preset: Very noisy receipts")

def set_preset_faint_text():
    """Preset for receipts with very faint text"""
    global AGGRESSIVE_DENOISING, CLAHE_CLIP_LIMIT, GAMMA_CORRECTION, SHARPEN_STRENGTH
    AGGRESSIVE_DENOISING = False
    CLAHE_CLIP_LIMIT = 1.5
    GAMMA_CORRECTION = 0.8  # Make brighter
    SHARPEN_STRENGTH = 0.8
    print("✓ Set preset: Faint text receipts")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_current_config():
    """Print current configuration settings"""
    print("\n" + "="*60)
    print("CURRENT OCR CONFIGURATION")
    print("="*60)
    print(f"Language:              {OCR_LANG}")
    print(f"Aggressive denoising:  {AGGRESSIVE_DENOISING}")
    print(f"Min image width:       {MIN_IMAGE_WIDTH}px")
    print(f"CLAHE clip limit:      {CLAHE_CLIP_LIMIT}")
    print(f"Gamma correction:      {GAMMA_CORRECTION}")
    print(f"Confidence threshold:  {CONFIDENCE_THRESHOLD}%")
    print(f"Debug mode:            {DEBUG_MODE}")
    print("="*60 + "\n")

def reset_to_defaults():
    """Reset all settings to default values"""
    global AGGRESSIVE_DENOISING, CLAHE_CLIP_LIMIT, GAMMA_CORRECTION
    global MEDIAN_KERNEL_SIZE, SHARPEN_STRENGTH
    
    AGGRESSIVE_DENOISING = True
    CLAHE_CLIP_LIMIT = 2.5
    GAMMA_CORRECTION = 1.2
    MEDIAN_KERNEL_SIZE = 3
    SHARPEN_STRENGTH = 0.5
    print("✓ Reset to default configuration")

# ============================================================================
# AUTO-CONFIGURATION BASED ON SAMPLE ANALYSIS
# ============================================================================

def analyze_sample_receipt(image_path: str) -> dict:
    """
    Analyze a sample receipt to suggest optimal configuration.
    Returns analysis results and recommended settings.
    """
    try:
        import cv2
        import numpy as np
        
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not load image"}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Analyze image characteristics
        analysis = {
            "dimensions": gray.shape,
            "mean_brightness": np.mean(gray),
            "std_brightness": np.std(gray),
            "contrast_ratio": np.max(gray) / max(np.min(gray), 1),
            "estimated_noise": estimate_noise_level(gray),
            "text_density": estimate_text_density(gray)
        }
        
        # Generate recommendations
        recommendations = {
            "aggressive_denoising": analysis["estimated_noise"] > 15,
            "clahe_clip_limit": 2.0 if analysis["contrast_ratio"] > 5 else 3.5,
            "gamma_correction": 1.0 if analysis["mean_brightness"] > 150 else 1.3,
            "reasoning": generate_reasoning(analysis)
        }
        
        return {
            "analysis": analysis,
            "recommendations": recommendations
        }
        
    except Exception as e:
        return {"error": str(e)}

def estimate_noise_level(image: np.ndarray) -> float:
    """Estimate noise level in the image"""
    # Simple noise estimation using local variance
    kernel = np.ones((5,5), np.float32)/25
    local_mean = cv2.filter2D(image.astype(np.float32), -1, kernel)
    local_var = cv2.filter2D((image.astype(np.float32) - local_mean)**2, -1, kernel)
    return np.mean(np.sqrt(local_var))

def estimate_text_density(image: np.ndarray) -> float:
    """Estimate the density of text in the image"""
    # Simple text density estimation using edge detection
    edges = cv2.Canny(image, 100, 200)
    return np.sum(edges > 0) / image.size

def generate_reasoning(analysis: dict) -> str:
    """Generate human-readable reasoning for recommendations"""
    reasons = []
    
    if analysis["estimated_noise"] > 15:
        reasons.append("High noise level detected - aggressive denoising recommended")
    
    if analysis["contrast_ratio"] < 3:
        reasons.append("Low contrast detected - higher CLAHE clip limit recommended")
    
    if analysis["mean_brightness"] < 100:
        reasons.append("Dark image detected - gamma correction recommended")
    
    if analysis["text_density"] < 0.05:
        reasons.append("Low text density - may need different preprocessing")
    
    return "; ".join(reasons) if reasons else "Standard settings should work well"

# ============================================================================
# MAIN (for testing configuration)
# ============================================================================

if __name__ == "__main__":
    print("OCR Configuration Module")
    print("Available presets:")
    print("  set_preset_mobile_phone()    - For mobile phone photos")
    print("  set_preset_professional_scan() - For high-quality scans")
    print("  set_preset_very_noisy()      - For extremely noisy receipts")
    print("  set_preset_faint_text()      - For receipts with faint text")
    print("  reset_to_defaults()          - Reset to default settings")
    print("  print_current_config()       - Show current settings")
    
    print_current_config()
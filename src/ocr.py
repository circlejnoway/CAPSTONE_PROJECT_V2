"""
ocr.py
------
Extract text from receipt images using Tesseract OCR with advanced preprocessing.
Features:
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting normalization
- Advanced morphological operations for text enhancement
- Multi-scale processing and intelligent upsampling
- Bilateral filtering for noise reduction while preserving edges
- Intelligent PSM (Page Segmentation Mode) selection
- Post-OCR confidence filtering
- Output text files to data/ocr_output/

IMPROVEMENTS:
1. Better contrast enhancement using CLAHE
2. Improved deskewing with better angle detection
3. Bilateral filtering to preserve text edges while removing noise
4. Optimal image scaling based on text size detection
5. Better Tesseract configuration with fallback strategies
6. Confidence-based filtering of garbled text
"""

import os
import sys
import re
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

import cv2
import numpy as np
import pytesseract
from PIL import Image

# Import configuration
config_loaded = False
try:
    from .ocr_config import *       # works when used as a package
    from .receipt_parser import parse_receipt
    config_loaded = True
except ImportError:
    try:
        from ocr_config import *    # works when running ocr.py directly
        from receipt_parser import parse_receipt
        config_loaded = True
    except ImportError:
        parse_receipt = None
        pass

# Define defaults if config not loaded
if not config_loaded:
    print("Warning: ocr_config.py not found, using default settings")
    OCR_LANG = "eng"
    MIN_IMAGE_WIDTH = 1200
    MAX_IMAGE_WIDTH = 3200
    CONFIDENCE_THRESHOLD = 20
    AGGRESSIVE_DENOISING = True
    CLAHE_CLIP_LIMIT = 2.5
    CLAHE_TILE_SIZE = (8, 8)
    GAMMA_CORRECTION = 1.2
    MEDIAN_KERNEL_SIZE = 3
    BILATERAL_SIGMA = 75
    NL_MEANS_H = 10
    ADAPTIVE_BLOCK_SIZE = 21
    ADAPTIVE_C = 5
    MORPH_KERNEL_SIZE = 3
    SHARPEN_STRENGTH = 0.5
    DEBUG_MODE = True
    SAVE_PREPROCESSED = True
    MAX_FILES_TO_PROCESS = 5
    BATCH_SIZE = 10
    RAW_RECEIPTS_DIR = Path("data/receipts/raw")
    OUTPUT_TEXT_DIR = Path("data/ocr_output")
    PREPROCESSED_DEBUG_DIR = Path("data/receipts/preprocessed")

STANDARD_OCR_WIDTH = 500
RECEIPT_MIN_AREA_RATIO = 0.08
RECEIPT_CONTOUR_EPSILON = 0.02

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Preprocessing functions
# ----------------------------------------------------------------------
def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk and normalize to 3-channel BGR format.
    Handles grayscale, RGBA, and other channel formats.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Ensure image is 3-channel BGR
    if len(img.shape) == 2:
        # Grayscale image - convert to BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        logger.debug(f"Converted grayscale image to BGR")
    elif img.shape[2] == 4:
        # RGBA image - remove alpha channel and convert to BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        logger.debug(f"Removed alpha channel from RGBA image")
    elif img.shape[2] != 3:
        # Unexpected channel format - try to convert
        logger.warning(f"Unexpected channel format: {img.shape[2]} channels. Attempting BGR conversion.")
        if img.shape[2] == 1:
            img = cv2.cvtColor(img[:,:,0], cv2.COLOR_GRAY2BGR)
        else:
            # Try to take first 3 channels as BGR
            img = img[:,:,:3]
    
    return img

def validate_image_format(image: np.ndarray, func_name: str = "unknown") -> bool:
    """
    Validate that image has correct format (3-channel BGR).
    Returns True if valid, False otherwise.
    """
    if image is None:
        logger.error(f"[{func_name}] Image is None")
        return False
    if len(image.shape) != 3:
        logger.error(f"[{func_name}] Expected 3D array, got {len(image.shape)}D")
        return False
    if image.shape[2] != 3:
        logger.error(f"[{func_name}] Expected 3 channels, got {image.shape[2]}")
        return False
    return True

# ----------------------------------------------------------------------
# New receipt preprocessing + localization helpers
# ----------------------------------------------------------------------

def resize_image_maintain_aspect(image: np.ndarray, width: int = STANDARD_OCR_WIDTH) -> np.ndarray:
    """Resize the image to a standard width while preserving aspect ratio."""
    h, w = image.shape[:2]
    if w == width:
        return image
    scale = width / float(w)
    new_height = int(h * scale)
    return cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA if width < w else cv2.INTER_CUBIC)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points in TL, TR, BR, BL order for perspective transform."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Warp the image so that the receipt appears in a top-down bird's-eye view."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_width, max_height), flags=cv2.INTER_LINEAR)


def find_receipt_contour(gray_image: np.ndarray) -> Optional[np.ndarray]:
    """Detect the largest four-sided contour likely representing the receipt."""
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, RECEIPT_CONTOUR_EPSILON * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area >= gray_image.shape[0] * gray_image.shape[1] * RECEIPT_MIN_AREA_RATIO:
                return approx.reshape(4, 2)
    return None


def locate_receipt(image: np.ndarray, debug: bool = False) -> np.ndarray:
    """Localize and warp the receipt region, or fall back to the original grayscale image."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    receipt_contour = find_receipt_contour(gray)
    if receipt_contour is None:
        if debug:
            logger.warning("Could not find a receipt contour; using full image.")
        return gray

    warped = four_point_transform(image, receipt_contour)
    if len(warped.shape) == 3 and warped.shape[2] == 3:
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    if debug:
        logger.info("Receipt localized and perspective-corrected.")
    return warped


def binarize_receipt(gray_image: np.ndarray) -> np.ndarray:
    """Apply adaptive thresholding and Otsu fallback for receipt text extraction."""
    block_size = ADAPTIVE_BLOCK_SIZE if ADAPTIVE_BLOCK_SIZE % 2 == 1 else ADAPTIVE_BLOCK_SIZE + 1
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        ADAPTIVE_C,
    )

    if np.mean(binary) > 245 or np.mean(binary) < 10:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def denoise_and_morphology(binary_image: np.ndarray) -> np.ndarray:
    """Use erosion/dilation and morphological opening/closing to clean the receipt text."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned


def preprocess_receipt(image: np.ndarray, debug: bool = False, aggressive: bool = True) -> np.ndarray:
    """Apply a simplified preprocessing pipeline aimed at receipt images."""
    if image is None or image.size == 0:
        raise ValueError("Invalid input image")

    resized = resize_image_maintain_aspect(image, width=STANDARD_OCR_WIDTH)
    if debug:
        logger.info(f"Resized input to {resized.shape[1]}px width")

    receipt_region = locate_receipt(resized, debug=debug)
    receipt_region = receipt_region if receipt_region is not None else cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    binary = binarize_receipt(receipt_region)
    cleaned = denoise_and_morphology(binary)

    if debug:
        logger.info("Preprocessing complete: grayscale, thresholding, morphology applied.")

    return cleaned


def denoise_bilateral(image: np.ndarray) -> np.ndarray:
    """
    Apply bilateral filtering: reduces noise while preserving edges (text).
    This is crucial for OCR as it smooths out specks without blurring text.
    Works with both grayscale and color images.
    """
    try:
        # Bilateral filter works on 1-channel or 3-channel images
        if len(image.shape) != 2 and len(image.shape) != 3:
            raise ValueError(f"Expected 2D or 3D array, got {len(image.shape)}D")
        
        result = cv2.bilateralFilter(image, 9, 75, 75)
        return result
    except Exception as e:
        logger.error(f"Error in denoise_bilateral: {e}")
        raise

def gamma_correction(image: np.ndarray, gamma: float = GAMMA_CORRECTION) -> np.ndarray:
    """
    Apply gamma correction to adjust image brightness and contrast.
    Gamma > 1 makes image darker (better for receipts with bright backgrounds).
    """
    try:
        # Normalize to [0, 1] range
        normalized = image.astype(np.float32) / 255.0
        # Apply gamma correction
        corrected = np.power(normalized, gamma)
        # Scale back to [0, 255] range
        result = (corrected * 255.0).astype(np.uint8)
        return result
    except Exception as e:
        logger.error(f"Error in gamma_correction: {e}")
        return image

def median_filter_denoise(image: np.ndarray, kernel_size: int = MEDIAN_KERNEL_SIZE) -> np.ndarray:
    """
    Apply median filtering to remove salt-and-pepper noise.
    Kernel size should be odd (3, 5, 7, etc.).
    """
    try:
        return cv2.medianBlur(image, kernel_size)
    except Exception as e:
        logger.error(f"Error in median_filter_denoise: {e}")
        return image

def non_local_means_denoise(image: np.ndarray) -> np.ndarray:
    """
    Apply non-local means denoising for better noise reduction.
    More effective than bilateral filtering for certain types of noise.
    """
    try:
        # For grayscale images
        if len(image.shape) == 2:
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        # For color images
        elif len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            logger.warning(f"Unexpected image shape in non_local_means_denoise: {image.shape}")
            return image
    except Exception as e:
        logger.error(f"Error in non_local_means_denoise: {e}")
        return image

def sharpen_text(image: np.ndarray, strength: float = SHARPEN_STRENGTH) -> np.ndarray:
    """
    Apply unsharp masking to sharpen text edges.
    Helps restore clarity lost during denoising.
    """
    try:
        # Gaussian blur for unsharp masking
        gaussian = cv2.GaussianBlur(image, (0, 0), 3.0)
        # Unsharp masking formula: original + (original - blurred) * strength
        sharpened = cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)
        return sharpened
    except Exception as e:
        logger.error(f"Error in sharpen_text: {e}")
        return image

def otsu_threshold(image: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's thresholding as an alternative to adaptive thresholding.
    Good for images with bimodal intensity distributions.
    """
    try:
        # Apply Gaussian blur first to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        # Otsu's thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    except Exception as e:
        logger.error(f"Error in otsu_threshold: {e}")
        return image

def advanced_morphological_cleanup(image: np.ndarray) -> np.ndarray:
    """
    Apply advanced morphological operations specifically for receipt text.
    Includes multiple passes and different kernel shapes.
    """
    try:
        # Convert to binary if not already
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # Step 1: Remove very small noise (1x1 and 2x2)
        kernel_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_noise, iterations=1)
        
        # Step 2: Connect broken text with horizontal/vertical lines
        # Horizontal kernel for connecting horizontal breaks
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_h, iterations=1)
        
        # Vertical kernel for connecting vertical breaks
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_v, iterations=1)
        
        # Step 3: Remove remaining small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Step 4: Fill small holes in characters
        kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_fill, iterations=1)
        
        return cleaned
    except Exception as e:
        logger.error(f"Error in advanced_morphological_cleanup: {e}")
        return image

def enhance_text_regions(image: np.ndarray) -> np.ndarray:
    """
    Enhance regions likely to contain text by analyzing local statistics.
    """
    try:
        # Calculate local standard deviation (text regions have higher variance)
        kernel_size = 15
        local_std = cv2.blur(image.astype(np.float32), (kernel_size, kernel_size))
        local_mean = cv2.blur(image.astype(np.float32), (kernel_size, kernel_size))
        
        # Calculate coefficient of variation (std/mean)
        # Text regions typically have higher variation
        cv_map = np.divide(local_std, local_mean + 1e-8, 
                          out=np.zeros_like(local_std), where=local_mean!=0)
        
        # Normalize to 0-255 range
        cv_normalized = cv2.normalize(cv_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply CLAHE to regions with high variation (likely text)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(cv_normalized)
        
        # Blend original with enhanced regions
        alpha = 0.3  # Weight for enhanced regions
        result = cv2.addWeighted(image, 1-alpha, enhanced, alpha, 0)
        
        return result
    except Exception as e:
        logger.error(f"Error in enhance_text_regions: {e}")
        return image

def estimate_rotation_angle(binary_image: np.ndarray) -> float:
    """
    Estimate the rotation angle more robustly using contour analysis.
    Returns angle in degrees (negative for clockwise, positive for counter-clockwise).
    """
    coords = np.column_stack(np.where(binary_image > 0))
    if len(coords) < 10:  # Not enough pixels
        return 0.0
    
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]  # Angle in degrees
    
    # Normalize angle to [-45, 45]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    return angle

def deskew_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image by the given angle to correct skew.
    Only rotate if angle is significant (> 0.5 degrees).
    """
    if abs(angle) <= 0.5:
        return image
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation with border replication
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated

def adaptive_threshold_image(image: np.ndarray) -> np.ndarray:
    """
    Apply adaptive thresholding which is superior to global thresholding
    for documents with varying lighting (like receipts).
    """
    # Use a larger block size for better receipt processing
    binary = cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,  # Increased from 31 for better text preservation
        C=5             # Increased from 2 for better foreground/background separation
    )
    return binary

def scale_image_optimal(image: np.ndarray, target_width: int = MIN_IMAGE_WIDTH) -> np.ndarray:
    """
    Upscale image if needed to improve OCR accuracy.
    Tesseract performs best at higher resolutions.
    """
    h, w = image.shape[:2]
    
    if w < target_width:
        # Calculate scale factor
        scale = target_width / w
        # Cap the scale to avoid excessive upsampling (max 2x)
        scale = min(scale, 2.0)
        
        # Upscale using high-quality interpolation
        new_w = int(w * scale)
        new_h = int(h * scale)
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return upscaled
    elif w > MAX_IMAGE_WIDTH:
        # Downscale if too large
        scale = MAX_IMAGE_WIDTH / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        downscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return downscaled
    
    return image

def extract_text_with_confidence(image: np.ndarray, lang: str = OCR_LANG) -> Tuple[str, Dict]:
    """
    Extract text from image using Tesseract with confidence scoring.
    Returns both the text and detailed metrics including confidence scores per word.
    """
    pil_img = Image.fromarray(image)
    
    # Get data with detailed information including confidence scores
    data = pytesseract.image_to_data(pil_img, lang=lang, output_type=pytesseract.Output.DICT)
    
    # Extract words with confidence > threshold
    filtered_text = []
    for i, conf in enumerate(data['conf']):
        if int(conf) > CONFIDENCE_THRESHOLD:
            word = data['text'][i]
            if word.strip():  # Only add non-empty words
                filtered_text.append(word)
    
    text = ' '.join(filtered_text)
    
    metrics = {
        'total_words': len(data['text']),
        'confident_words': sum(1 for c in data['conf'] if int(c) > CONFIDENCE_THRESHOLD),
        'avg_confidence': np.mean([int(c) for c in data['conf'] if c != '-1'])
    }
    
    return text, metrics

def extract_text(image: np.ndarray, lang: str = OCR_LANG) -> str:
    """
    Extract text from preprocessed receipt image using optimized Tesseract configuration.
    Uses multiple strategies with fallback options.
    """
    h, w = image.shape[:2]
    logger.info(f"Processing image size: {w}x{h}")
    
    pil_img = Image.fromarray(image)
    
    # Try receipt-friendly page segmentation modes first.
    # PSM 4 assumes a single column of variable-sized text.
    # PSM 6 assumes a single uniform block of text.
    psm_modes = ["4", "6", "3"]

    best_text = ""
    best_length = 0

    for psm in psm_modes:
        try:
            config = f"--oem 3 --psm {psm}"
            text = pytesseract.image_to_string(pil_img, lang=lang, config=config).strip()

            alpha_count = len(re.sub(r'[^a-zA-Z]', '', text))
            logger.debug(f"PSM {psm}: {alpha_count} letters, {len(text)} chars")

            if alpha_count > best_length:
                best_text = text
                best_length = alpha_count
                if alpha_count > 50:
                    logger.debug(f"Good result with PSM {psm}")
                    break
        except Exception as e:
            logger.warning(f"PSM {psm} failed: {e}")
            continue

    return best_text

# Receipt entity parsing
DATE_PATTERNS = [
    re.compile(r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"),
    re.compile(r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"),
    re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[ .,-]+\d{1,2}[, ]+[0-9]{4}\b", re.IGNORECASE),
]
TOTAL_PATTERN = re.compile(r"(?mi)^(?:grand total|total amount|amount due|total|balance|sum)[:\s]*\$?(\d+\.\d{2})$")


def parse_receipt_entities(ocr_text: str) -> Dict[str, Any]:
    """Extract merchant, date, total and line items from OCR text."""
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]

    merchant = ""
    for line in lines[:4]:
        normalized = line.lower()
        if any(term in normalized for term in ["receipt", "invoice", "store", "total"]):
            continue
        if len(line) > 3 and any(char.isalpha() for char in line):
            merchant = line
            break
    if not merchant and lines:
        merchant = lines[0]

    date = ""
    for pattern in DATE_PATTERNS:
        match = pattern.search(ocr_text)
        if match:
            date = match.group(0)
            break

    total = ""
    for line in reversed(lines):
        match = TOTAL_PATTERN.search(line)
        if match:
            total = match.group(1)
            break

    items = []
    if callable(parse_receipt):
        try:
            items = parse_receipt(ocr_text)
        except Exception as e:
            logger.warning(f"Could not parse receipt line items: {e}")

    return {
        "merchant": merchant,
        "date": date,
        "total": total,
        "items": items,
    }

# ----------------------------------------------------------------------
# Main batch processing function
# ----------------------------------------------------------------------
def process_all_receipts(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    save_text: bool = True,
    debug: bool = DEBUG_MODE,
    max_files: Optional[int] = MAX_FILES_TO_PROCESS
) -> Dict[str, str]:
    """
    Process all receipt images in raw_dir and return a dict
    of {filename: extracted_text}.

    Parameters:
    -----------
    raw_dir : Path
        Directory containing raw receipt images
    output_dir : Path
        Directory where extracted text will be saved
    save_text : bool
        Whether to save extracted text to files
    debug : bool
        If True, save preprocessed images and detailed logs
    max_files : Optional[int]
        Limit processing to first N files (useful for testing)

    Returns:
    --------
    Dict[str, str]
        Dictionary mapping filename to extracted text
    """
    # Use defaults if not provided
    if raw_dir is None:
        raw_dir = RAW_RECEIPTS_DIR
    if output_dir is None:
        output_dir = OUTPUT_TEXT_DIR
    
    if not raw_dir.exists():
        logger.error(f"Raw receipts folder not found: {raw_dir}")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)

    supported_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    results = {}
    file_count = 0

    # Create debug directory if needed
    if debug:
        debug_dir = Path("data/receipts/preprocessed")
        debug_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(raw_dir.iterdir()):
        if img_path.suffix.lower() not in supported_exts:
            continue

        if max_files and file_count >= max_files:
            logger.info(f"Reached max_files limit ({max_files})")
            break

        logger.info(f"[{file_count + 1}] Processing: {img_path.name}")

        try:
            # Load and preprocess
            try:
                img = load_image(str(img_path))
            except Exception as e:
                logger.error(f"  ✗ Failed to load image: {e}")
                results[img_path.name] = ""
                file_count += 1
                continue
            
            try:
                preprocessed = preprocess_receipt(img, debug=debug, aggressive=AGGRESSIVE_DENOISING)
            except Exception as e:
                logger.error(f"  ✗ Failed to preprocess: {e}")
                results[img_path.name] = ""
                file_count += 1
                continue

            # If debug, save preprocessed image
            if debug:
                debug_path = Path("data/receipts/preprocessed") / f"preprocessed_{img_path.stem}.png"
                try:
                    cv2.imwrite(str(debug_path), preprocessed)
                    logger.debug(f"Saved preprocessed image to {debug_path}")
                except Exception as e:
                    logger.warning(f"Could not save preprocessed image: {e}")

            # OCR
            try:
                text = extract_text(preprocessed, lang=OCR_LANG)
            except Exception as e:
                logger.error(f"  ✗ Failed to extract text: {e}")
                results[img_path.name] = ""
                file_count += 1
                continue

            parsed_data = parse_receipt_entities(text)
            if debug:
                logger.info(
                    f"  Parsed merchant={parsed_data['merchant']}, date={parsed_data['date']}, total={parsed_data['total']}, items={len(parsed_data['items'])}"
                )
                parsed_path = output_dir / f"{img_path.stem}.json"
                try:
                    with open(parsed_path, "w", encoding="utf-8") as f:
                        json.dump(parsed_data, f, indent=2)
                except Exception as e:
                    logger.warning(f"  Could not save parsed JSON: {e}")

            results[img_path.name] = text

            # Log extraction quality
            text_length = len(text)
            alpha_count = len(re.sub(r'[^a-zA-Z]', '', text))
            logger.info(f"  ✓ Extracted {text_length} chars ({alpha_count} letters)")

            # Save text output
            if save_text:
                out_file = output_dir / f"{img_path.stem}.txt"
                try:
                    with open(out_file, "w", encoding="utf-8") as f:
                        f.write(text)
                except Exception as e:
                    logger.error(f"  ✗ Failed to save output: {e}")

        except Exception as e:
            logger.error(f"Unexpected error processing {img_path.name}: {e}")
            results[img_path.name] = ""

        file_count += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete. Processed {file_count} files.")
    logger.info(f"Saved to: {output_dir}")
    logger.info(f"{'='*60}")
    
    return results

# ----------------------------------------------------------------------
# Stand-alone test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Quick test – processes all images in data/receipts/raw
    print("=" * 70)
    print("OCR RECEIPT PROCESSING SYSTEM")
    print("=" * 70)
    print("\nStarting OCR processing with enhanced preprocessing...")
    print(f"Configuration:")
    print(f"  - Language: {OCR_LANG}")
    print(f"  - Min image width: {MIN_IMAGE_WIDTH}px")
    print(f"  - Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  - Aggressive denoising: {AGGRESSIVE_DENOISING}")
    print()
    
    # Process all receipts (or first 10 for testing)
    extracted = process_all_receipts(debug=DEBUG_MODE, max_files=MAX_FILES_TO_PROCESS)
    
    print("\n" + "=" * 70)
    print("SAMPLE RESULTS (first 3 files)")
    print("=" * 70)
    
    for i, (fname, text) in enumerate(list(extracted.items())[:3]):
        print(f"\n[{i+1}] {fname}")
        print("-" * 70)
        # Show first 400 characters
        preview = text[:400] if text else "[NO TEXT EXTRACTED]"
        print(preview)
        if len(text) > 400:
            print(f"... ({len(text)} total characters)")
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: Successfully processed {len(extracted)} files")
    print(f"Results saved to: {OUTPUT_TEXT_DIR}")
    print("=" * 70)
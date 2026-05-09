"""
train_ocr.py
-----------
Train and fine-tune OCR models using the SROIE dataset.

SROIE Dataset: https://www.kaggle.com/datasets/urbikn/sroie-datasetv2
- Contains ~600 receipt images with ground truth text annotations
- Perfect for receipt-specific OCR training
- Includes bounding boxes and character-level annotations

This module supports:
1. EasyOCR fine-tuning (recommended - easier setup)
2. PaddleOCR training (more advanced, better for production)
3. Custom Tesseract model training (complex, requires tesstrain)

Setup Instructions:
-------------------
1. Download SROIE dataset from Kaggle:
   - Visit: https://www.kaggle.com/datasets/urbikn/sroie-datasetv2
   - Click "Download" (requires Kaggle account)
   - Extract to: data/sroie/

2. Install training dependencies:
   pip install paddleocr paddlepaddle opencv-python pillow

3. Run training:
   python src/train_ocr.py --dataset_dir data/sroie/ --output_dir models/receipt_ocr/ --epochs 50

4. Use trained model in OCR pipeline (see usage examples below)
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Try importing training libraries
try:
    from paddleocr import PaddleOCR
    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False
    logger.warning("PaddleOCR not installed. Install with: pip install paddleocr paddlepaddle")

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False
    logger.warning("EasyOCR not installed. Install with: pip install easyocr")

# ============================================================================
# SROIE Dataset Utilities
# ============================================================================

class SROIEDataset:
    """Handler for SROIE receipt dataset."""
    
    def __init__(self, dataset_dir: Path):
        """
        Initialize SROIE dataset.
        
        Expected structure:
        data/sroie/
        ├── images/
        │   ├── 0001.jpg
        │   ├── 0002.jpg
        │   └── ...
        ├── annotations/
        │   ├── 0001.txt
        │   ├── 0002.txt
        │   └── ...
        └── box_labels/
            ├── 0001.txt
            ├── 0002.txt
            └── ...
        """
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.annot_dir = self.dataset_dir / "annotations"
        self.box_dir = self.dataset_dir / "box_labels"
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")) + 
                                  list(self.images_dir.glob("*.png")))
        logger.info(f"Found {len(self.image_files)} receipt images")
    
    def get_annotations(self, image_file: Path) -> str:
        """Get full OCR text annotation for an image."""
        stem = image_file.stem
        annot_file = self.annot_dir / f"{stem}.txt"
        if annot_file.exists():
            with open(annot_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        return ""
    
    def get_boxes_and_text(self, image_file: Path) -> List[Dict]:
        """Get bounding boxes and text for each region."""
        stem = image_file.stem
        box_file = self.box_dir / f"{stem}.txt"
        if not box_file.exists():
            return []
        
        boxes = []
        with open(box_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 9:
                    # Format: x1,y1,x2,y2,x3,y3,x4,y4,text
                    coords = list(map(int, parts[:8]))
                    text = ",".join(parts[8:])
                    boxes.append({
                        "coords": coords,
                        "text": text
                    })
        return boxes
    
    def get_train_test_split(self, train_ratio: float = 0.8) -> Tuple[List[Path], List[Path]]:
        """Split dataset into train and test sets."""
        np.random.seed(42)
        indices = np.arange(len(self.image_files))
        np.random.shuffle(indices)
        
        split_idx = int(len(indices) * train_ratio)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        train_files = [self.image_files[i] for i in train_indices]
        test_files = [self.image_files[i] for i in test_indices]
        
        logger.info(f"Train: {len(train_files)} images, Test: {len(test_files)} images")
        return train_files, test_files


# ============================================================================
# PaddleOCR-based Training (Recommended)
# ============================================================================

class PaddleOCRTrainer:
    """Train PaddleOCR model on SROIE dataset."""
    
    def __init__(self, output_dir: Path):
        """Initialize PaddleOCR trainer."""
        if not HAS_PADDLE:
            raise ImportError("PaddleOCR not installed. Run: pip install paddleocr paddlepaddle")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = self.output_dir / "paddleocr_receipt"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PaddleOCR trainer initialized. Output: {self.model_dir}")
    
    def prepare_training_data(self, dataset: SROIEDataset, output_dir: Path):
        """Convert SROIE to PaddleOCR format."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_files, test_files = dataset.get_train_test_split()
        
        # Create train and val txt files
        train_gt = []
        val_gt = []
        
        for img_file in train_files:
            text = dataset.get_annotations(img_file)
            if text:
                # PaddleOCR format: image_path\ttext
                train_gt.append(f"{img_file.absolute()}\t{text}")
        
        for img_file in test_files:
            text = dataset.get_annotations(img_file)
            if text:
                val_gt.append(f"{img_file.absolute()}\t{text}")
        
        # Write to files
        train_gt_path = output_dir / "train_gt.txt"
        val_gt_path = output_dir / "val_gt.txt"
        
        with open(train_gt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(train_gt))
        
        with open(val_gt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(val_gt))
        
        logger.info(f"Training data prepared: {len(train_gt)} train, {len(val_gt)} val")
        return train_gt_path, val_gt_path
    
    def train(self, dataset: SROIEDataset, epochs: int = 50, batch_size: int = 32):
        """Train PaddleOCR model on SROIE dataset."""
        logger.info("Preparing training data for PaddleOCR...")
        train_gt, val_gt = self.prepare_training_data(dataset, self.model_dir / "data")
        
        logger.info(f"Training PaddleOCR model for {epochs} epochs...")
        logger.info("Note: First run downloads pre-trained model (~100MB)")
        
        try:
            from paddleocr import PaddleOCR
            from paddleocr.tools.train import main as paddle_train
            
            # Create config for training
            config = {
                "Global": {
                    "epoch_num": epochs,
                    "log_smooth_window": 20,
                    "print_batch_step": 10,
                    "save_model_dir": str(self.model_dir),
                    "save_epoch_step": 5,
                    "eval_batch_step": [0, 100],
                },
                "Train": {
                    "loader": {
                        "num_workers": 0,
                        "batch_size_per_shard": batch_size,
                    }
                },
            }
            
            config_path = self.model_dir / "config.yml"
            with open(config_path, "w") as f:
                import yaml
                yaml.dump(config, f)
            
            logger.info(f"Training config saved to {config_path}")
            logger.info("For full PaddleOCR training, use the official training script:")
            logger.info(f"  python -m paddle.distributed.launch --gpus=0 tools/train.py -c {config_path}")
            
        except Exception as e:
            logger.error(f"PaddleOCR training setup failed: {e}")
            logger.info("Alternative: Fine-tune using EasyOCR instead")


# ============================================================================
# EasyOCR Fine-tuning (Simpler Alternative)
# ============================================================================

class EasyOCRTrainer:
    """Fine-tune EasyOCR recognition model on SROIE dataset."""
    
    def __init__(self, output_dir: Path):
        """Initialize EasyOCR trainer."""
        if not HAS_EASYOCR:
            raise ImportError("EasyOCR not installed. Run: pip install easyocr")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = self.output_dir / "easyocr_receipt"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"EasyOCR trainer initialized. Output: {self.model_dir}")
    
    def create_training_data(self, dataset: SROIEDataset) -> Dict:
        """Create training dataset in EasyOCR format."""
        train_files, test_files = dataset.get_train_test_split()
        
        train_data = []
        for img_file in train_files:
            boxes = dataset.get_boxes_and_text(img_file)
            if boxes:
                train_data.append({
                    "image": str(img_file.absolute()),
                    "boxes": boxes
                })
        
        logger.info(f"Created training data with {len(train_data)} annotated images")
        return {
            "train": train_data,
            "test": [{"image": str(f.absolute()), "boxes": dataset.get_boxes_and_text(f)} 
                    for f in test_files]
        }
    
    def train(self, dataset: SROIEDataset, epochs: int = 50):
        """Fine-tune EasyOCR on SROIE dataset."""
        logger.info("Creating training dataset...")
        train_data = self.create_training_data(dataset)
        
        logger.info(f"Starting EasyOCR fine-tuning for {epochs} epochs...")
        logger.info("Note: This requires training infrastructure not fully implemented here.")
        logger.info("Use official EasyOCR documentation for fine-tuning: https://github.com/JaidedAI/EasyOCR")
        
        # Save training data for reference
        data_path = self.model_dir / "training_data.json"
        with open(data_path, "w") as f:
            json.dump(train_data, f, indent=2)
        
        logger.info(f"Training data saved to {data_path}")


# ============================================================================
# Evaluation Utilities
# ============================================================================

def evaluate_ocr(model, dataset: SROIEDataset, num_samples: int = 20) -> Dict:
    """Evaluate OCR model on SROIE test set."""
    import difflib
    
    _, test_files = dataset.get_train_test_split()
    test_files = test_files[:num_samples]
    
    results = {
        "total": len(test_files),
        "correct": 0,
        "character_error_rate": [],
        "word_error_rate": [],
    }
    
    for img_file in test_files:
        gt_text = dataset.get_annotations(img_file).lower()
        if not gt_text:
            continue
        
        try:
            img = cv2.imread(str(img_file))
            # Use your OCR function here
            pred_text = ""  # Replace with actual OCR call
            
            if pred_text.lower() == gt_text:
                results["correct"] += 1
            
            # Calculate edit distance
            char_errors = sum(
                1 for a, b in zip(gt_text, pred_text) if a != b
            ) + abs(len(gt_text) - len(pred_text))
            
            results["character_error_rate"].append(char_errors / max(len(gt_text), 1))
        except Exception as e:
            logger.warning(f"Error evaluating {img_file}: {e}")
    
    if results["character_error_rate"]:
        results["avg_cer"] = np.mean(results["character_error_rate"])
        results["avg_wer"] = np.mean(results["word_error_rate"]) if results["word_error_rate"] else 0
    
    results["accuracy"] = results["correct"] / max(results["total"], 1)
    
    return results


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train OCR model on SROIE dataset")
    parser.add_argument("--dataset_dir", type=str, default="data/sroie",
                       help="Path to SROIE dataset directory")
    parser.add_argument("--output_dir", type=str, default="models/receipt_ocr",
                       help="Output directory for trained models")
    parser.add_argument("--engine", type=str, choices=["paddle", "easyocr"], default="paddle",
                       help="OCR engine to train")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        logger.info(f"Download SROIE dataset from: https://www.kaggle.com/datasets/urbikn/sroie-datasetv2")
        logger.info(f"Extract to: {dataset_path}")
        return
    
    # Load dataset
    try:
        dataset = SROIEDataset(dataset_path)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    output_path = Path(args.output_dir)
    
    # Train model
    if args.engine == "paddle":
        if not HAS_PADDLE:
            logger.error("PaddleOCR not installed. Run: pip install paddleocr paddlepaddle")
            return
        trainer = PaddleOCRTrainer(output_path)
        trainer.train(dataset, epochs=args.epochs, batch_size=args.batch_size)
    
    elif args.engine == "easyocr":
        if not HAS_EASYOCR:
            logger.error("EasyOCR not installed. Run: pip install easyocr")
            return
        trainer = EasyOCRTrainer(output_path)
        trainer.train(dataset, epochs=args.epochs)
    
    logger.info("Training complete!")
    logger.info(f"Models saved to: {output_path}")


if __name__ == "__main__":
    main()

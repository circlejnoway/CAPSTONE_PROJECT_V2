# OCR Model Training Guide

## Overview

This guide explains how to train a custom OCR model using the **SROIE dataset** (Scanned Receipts OCR and Information Extraction) to improve receipt text recognition accuracy.

---

## Why Train a Custom Model?

- **Pre-trained Tesseract** is generic and performs poorly on receipts with specific fonts/layouts
- **SROIE dataset** contains ~600 real-world receipt images with ground truth annotations
- **Custom models** can achieve 5-10x better accuracy on receipt-specific text

---

## Step 1: Download SROIE Dataset

### Option A: Kaggle (Recommended)

1. Go to: https://www.kaggle.com/datasets/urbikn/sroie-datasetv2
2. Click **"Download"** (requires free Kaggle account)
3. Extract the downloaded file:
   ```bash
   cd C:\Users\Asus\Desktop\CAPSTONE_PROJECT_V2
   unzip sroie-datasetv2.zip -d data/sroie/
   ```

### Expected Directory Structure
```
data/sroie/
├── images/              # Receipt images
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ... (600+ images)
├── annotations/         # Full text per receipt
│   ├── 0001.txt
│   ├── 0002.txt
│   └── ...
└── box_labels/         # Bounding boxes per text region
    ├── 0001.txt
    ├── 0002.txt
    └── ...
```

---

## Step 2: Install Training Dependencies

### Option A: Using PaddleOCR (Recommended)
Best for production models; powerful but requires more setup.

```bash
conda activate env_capstone_v2

# Install PaddleOCR and related packages
pip install paddleocr paddlepaddle opencv-python pillow pyyaml

# For GPU acceleration (optional, significantly faster)
pip install paddlepaddle-gpu
```

### Option B: Using EasyOCR
Simpler setup, fine-tuning less mature but good for experimentation.

```bash
conda activate env_capstone_v2
pip install easyocr torch
```

### Verify Installation
```bash
python -c "import paddleocr; print('PaddleOCR OK')"
python -c "import easyocr; print('EasyOCR OK')"
```

---

## Step 3: Train the Model

### Quick Start (Recommended)

```bash
cd C:\Users\Asus\Desktop\CAPSTONE_PROJECT_V2
conda activate env_capstone_v2

# Train PaddleOCR model
python src/train_ocr.py \
    --dataset_dir data/sroie/ \
    --output_dir models/receipt_ocr/ \
    --engine paddle \
    --epochs 50 \
    --batch_size 32
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset_dir` | `data/sroie/` | Path to SROIE dataset |
| `--output_dir` | `models/receipt_ocr/` | Where to save trained models |
| `--engine` | `paddle` | Training engine: `paddle` or `easyocr` |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `32` | Batch size (reduce if out of memory) |

### Expected Training Output
```
INFO: Found 600 receipt images
INFO: Train: 480 images, Test: 120 images
INFO: Preparing training data for PaddleOCR...
INFO: Training PaddleOCR model for 50 epochs...
[2024-01-15 14:32:01] epoch 1/50 ... loss: 2.5432
[2024-01-15 14:35:12] epoch 2/50 ... loss: 1.8921
...
[2024-01-15 18:22:45] epoch 50/50 ... loss: 0.1205
INFO: Training complete!
INFO: Models saved to: models/receipt_ocr/
```

### Troubleshooting

**"CUDA out of memory"**
→ Reduce batch size: `--batch_size 16` or `--batch_size 8`

**"PaddleOCR not installed"**
→ Run: `pip install paddleocr paddlepaddle`

**"Module 'yaml' not found"**
→ Run: `pip install pyyaml`

---

## Step 4: Use Trained Model in OCR Pipeline

### Method 1: Specify Engine in Config

Edit `src/ocr_config.py`:
```python
# Use trained EasyOCR model
OCR_ENGINE = "easyocr"
USE_EASYOCR = True
```

### Method 2: Use Directly in Code

```python
from src.ocr import process_all_receipts

# Process receipts with trained EasyOCR model
results = process_all_receipts(
    debug=True,
    max_files=10
)
```

### Method 3: Register Custom Tesseract Model (Advanced)

For PaddleOCR trained models:
```bash
# After training completes, models saved to:
# models/receipt_ocr/inference/rec_model
```

---

## Step 5: Evaluate Model Performance

### Basic Evaluation
```bash
python -c "
from src.train_ocr import SROIEDataset, evaluate_ocr

dataset = SROIEDataset('data/sroie/')
results = evaluate_ocr(dataset, num_samples=50)

print(f'Accuracy: {results[\"accuracy\"]:.2%}')
print(f'Avg Character Error Rate: {results[\"avg_cer\"]:.4f}')
"
```

### Compare Before/After Training
```bash
# Using old model
python src/compare_ocr_methods.py

# Using trained model (update engine in config first)
python src/compare_ocr_methods.py
```

---

## Advanced: Custom Tesseract Training (Experts Only)

For production-grade Tesseract models:

1. **Install tesstrain:**
   ```bash
   git clone https://github.com/tesseract-ocr/tesstrain.git
   cd tesstrain
   make requirements
   ```

2. **Prepare ground truth:**
   ```bash
   python scripts/prep_sroie_for_tesstrain.py data/sroie/ tesstrain/data/receipt
   ```

3. **Train model:**
   ```bash
   cd tesstrain
   make training MODEL_NAME=receipt
   ```

4. **Install trained model:**
   ```bash
   cp tesstrain/data/receipt/receipt.traineddata \
      C:\Program Files\Tesseract-OCR\tessdata\
   ```

---

## Performance Tips

### For Better Accuracy
- Train for **more epochs** (100-200 recommended for production)
- Use **larger batch sizes** if GPU memory allows (64, 128)
- Include **data augmentation** in training
- Train on **GPU** (10-20x faster than CPU)

### For Faster Training
- Reduce **batch size** (speeds up training, may reduce accuracy)
- Use **fewer epochs** (but may underfit)
- Train only **recognition model** (faster than full pipeline)

### Optimal Configuration (Production)
```bash
python src/train_ocr.py \
    --dataset_dir data/sroie/ \
    --output_dir models/receipt_ocr_prod/ \
    --engine paddle \
    --epochs 100 \
    --batch_size 64
```

---

## Dataset Details

### SROIE Dataset Statistics
- **600 receipt images** (varied sizes, lighting, fonts)
- **Horizontal and rotated** receipts
- **Multiple languages** (English, Arabic, etc.)
- **Text regions** with bounding boxes
- **Item-level annotations** (product, price, quantity)

### Data Split
- **Train:** 80% (480 images)
- **Test:** 20% (120 images)
- Random split with seed=42 for reproducibility

---

## Comparing OCR Engines

### Tesseract
- ✅ Fast, lightweight
- ❌ Generic, poor on specialized text
- Best for: Simple, clean documents

### EasyOCR
- ✅ Better default accuracy
- ✅ Multi-language support
- ❌ Slower, larger model
- Best for: General OCR

### PaddleOCR (Trained on SROIE)
- ✅ Excellent receipt accuracy (after training)
- ✅ Fastest trained model
- ❌ Requires training effort
- Best for: Production receipt systems

---

## Next Steps

1. ✅ Download SROIE dataset
2. ✅ Install PaddleOCR/EasyOCR
3. ✅ Train custom model (50 epochs, ~1-2 hours)
4. ✅ Evaluate on test set
5. ✅ Deploy to production

For questions, refer to:
- PaddleOCR docs: https://github.com/PaddlePaddle/PaddleOCR
- EasyOCR docs: https://github.com/JaidedAI/EasyOCR
- SROIE dataset: https://www.kaggle.com/datasets/urbikn/sroie-datasetv2

---

## Quick Reference Commands

```bash
# Activate environment
conda activate env_capstone_v2

# Download and extract SROIE
# (manually from Kaggle)

# Install dependencies
pip install paddleocr paddlepaddle

# Train model
python src/train_ocr.py --dataset_dir data/sroie/ --epochs 50

# Test OCR with trained model
python src/ocr.py

# Compare methods
python src/compare_ocr_methods.py
```

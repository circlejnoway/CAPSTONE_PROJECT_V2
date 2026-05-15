# Google Colab Setup Guide for PaddleOCR Training

## Step 1: Prepare Your Data for Google Drive

### On your local machine:

1. **Create a zip file of your training data:**
   - Navigate to your project folder: `C:\Users\Asus\Desktop\CAPSTONE_PROJECT_V2\`
   - Create a zip file containing:
     - `models/receipt_ocr/paddleocr_receipt/data/train_gt.txt`
     - `models/receipt_ocr/paddleocr_receipt/data/val_gt.txt`
     - `models/receipt_ocr/paddleocr_receipt/data/ppocr_keys_v1.txt`
     - `data/sroie/SROIE2019/train/img/` (all receipt images)
   
   Name it: `receipt_ocr_data.zip`

2. **Upload to Google Drive:**
   - Go to https://drive.google.com
   - Create a new folder named `capstone_ocr`
   - Upload `receipt_ocr_data.zip` to that folder

---

## Step 2: Create Google Colab Notebook

1. Go to https://colab.research.google.com
2. Click **"New notebook"**
3. Rename it: `PaddleOCR_Receipt_Training`
4. Copy and paste the code cells below in order

---

## Step 3: Colab Notebook Code Cells

### Cell 1: Enable GPU
```python
# Check GPU availability
import tensorflow as tf
print("GPU available:", tf.config.list_physical_devices('GPU'))
```

### Cell 2: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
- Click the link and authorize
- Copy the authorization code and paste it

### Cell 3: Extract and Setup Data
```python
import os
import zipfile

# Extract data
zip_path = '/content/drive/My Drive/capstone_ocr/receipt_ocr_data.zip'
extract_path = '/content/receipt_ocr_data'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction complete!")
print("Contents:", os.listdir(extract_path))
```

### Cell 4: Install PaddleOCR and Dependencies
```python
# Install PaddleOCR
!pip install paddleocr paddlepaddle -q
!pip install pyyaml -q

# Check installation
import paddle
print("PaddleOCR installed successfully!")
print("Paddle version:", paddle.__version__)
```

### Cell 5: Clone PaddleOCR Repository
```python
import os
os.chdir('/content')

# Clone if not already there
!git clone https://github.com/PaddlePaddle/PaddleOCR.git -q

os.chdir('/content/PaddleOCR')
print("Working directory:", os.getcwd())
```

### Cell 6: Create Training Config
```python
import os

# Create config directory
config_dir = '/content/receipt_ocr_config'
os.makedirs(config_dir, exist_ok=True)

# Create config.yml
config_content = """Global:
  cal_metric_during_train: true
  character_dict_path: /content/receipt_ocr_data/ppocr_keys_v1.txt
  checkpoints: null
  distributed: false
  epoch_num: 50
  eval_batch_step: [0, 100]
  infer_mode: false
  log_smooth_window: 20
  max_text_length: 100
  pretrained_model: null
  print_batch_step: 10
  save_epoch_step: 5
  save_model_dir: /content/output/rec/receipt_ocr/
  use_gpu: true
  use_space_char: true

Train:
  dataset:
    data_dir: /content/receipt_ocr_data/
    label_file_list: 
      - /content/receipt_ocr_data/train_gt.txt
    name: SimpleDataSet
    transforms:
      - DecodeImage:
          channel_first: false
          img_mode: BGR
      - RecAug: null
      - MultiLabelEncode: null
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: ['image', 'label_ctc', 'label_sar', 'length', 'valid_ratio']
  loader:
    batch_size_per_card: 64
    drop_last: true
    num_workers: 4
    shuffle: true

Eval:
  dataset:
    data_dir: /content/receipt_ocr_data/
    label_file_list: 
      - /content/receipt_ocr_data/val_gt.txt
    name: SimpleDataSet
    transforms:
      - DecodeImage:
          channel_first: false
          img_mode: BGR
      - MultiLabelEncode: null
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: ['image', 'label_ctc', 'label_sar', 'length', 'valid_ratio']
  loader:
    batch_size_per_card: 64
    drop_last: false
    num_workers: 4
    shuffle: false

Architecture:
  algorithm: SVTR_LCNet
  model_type: rec
  Transform: null
  Backbone:
    name: MobileNetV1Enhance
    scale: 0.5
    last_conv_stride: [1, 2]
    last_pool_type: avg
  Neck:
    name: SequenceEncoder
    encoder_type: rnn
  Head:
    name: MultiHead
    head_list:
      - CTCHead:
          Head:
            fc_decay: 1.0e-05
          Neck:
            name: rnn
            depth: 2
            dims: 64
            hidden_dims: 120
      - SARHead:
          enc_dim: 512
          max_text_length: 100

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss: null
    - SARLoss: null

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.001
    warmup_epoch: 5
  regularizer:
    name: L2
    factor: 3.0e-05

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc
"""

config_path = os.path.join(config_dir, 'config.yml')
with open(config_path, 'w') as f:
    f.write(config_content)

print(f"Config created at: {config_path}")
```

### Cell 7: Start Training
```python
import os
os.chdir('/content/PaddleOCR')

# Run training
!python tools/train.py -c /content/receipt_ocr_config/config.yml
```

---

## Step 4: Expected Results

**With GPU (Colab T4):**
- ~0.5-1 hour for 50 epochs (instead of 10 hours)
- **Speed improvement: 10-20x faster**

**Estimated timeline:**
- Cell 1-4: ~2 minutes
- Cell 5: ~1 minute
- Cell 6: ~30 seconds
- Cell 7 (Training): ~45 minutes to 1 hour

---

## Step 5: Download Results (After Training Completes)

### Cell 8: Save Model to Drive
```python
import shutil

# Copy trained model to Google Drive
source = '/content/output/rec/receipt_ocr/'
dest = '/content/drive/My Drive/capstone_ocr/trained_model/'

shutil.copytree(source, dest, dirs_exist_ok=True)
print("Model saved to Google Drive!")
```

---

## Important Notes

⚠️ **Colab Session Limits:**
- Free tier: 12 hours maximum per session
- After 12 hours: notebook disconnects and you lose unsaved work
- **Solutions:**
  - Save checkpoints to Google Drive frequently
  - Use paid Colab+ for longer sessions ($9.99/month, 24 hours)

💡 **Key Optimizations for Colab:**
- Increased batch_size to 64 (from 16) → faster training
- GPU enabled (use_gpu: true)
- No pretrained model (faster initialization)

📊 **Monitor Training:**
- Watch the loss values decrease
- Check validation accuracy (acc) increasing

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No such file or directory" | Check paths use `/content/` not Windows paths |
| Out of memory | Reduce `batch_size_per_card` to 32 or 16 |
| Training too slow | Ensure GPU is enabled (Cell 1 should show GPU) |
| Data not found | Verify zip file uploaded to Google Drive correctly |

---

## Next Steps After Training

1. Download model from Google Drive to your local machine
2. Test it with your OCR inference script
3. Fine-tune if accuracy is not satisfactory

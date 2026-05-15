@echo off
cd /d "C:\Users\Asus\Desktop\CAPSTONE_PROJECT_V2\PaddleOCR"
python tools/train.py -c "../models/receipt_ocr/paddleocr_receipt/config.yml"
pause
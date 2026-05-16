"""
pipeline.py
-----------
Core pipeline orchestration for the receipt OCR and nutrition workflow.
"""

from .ocr_engine import load_image
from .parser import parse_receipt


def run_pipeline(image_path: str):
    img = load_image(image_path)
    # TODO: add pipeline orchestration logic
    return {"status": "pending", "image_path": image_path}

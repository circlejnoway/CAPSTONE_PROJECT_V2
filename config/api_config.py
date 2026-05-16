from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

API_CONFIG = {
    "usda_api_key": os.getenv("USDA_API_KEY", ""),
    "ninjas_api_key": os.getenv("NINJAS_API_KEY", ""),
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
    "ppocr_keys_file": os.getenv("PPOCR_KEYS_FILE", "ppocr_keys_v1.txt"),
}

# CAPSTONE_PROJECT_V2 - Receipt OCR

OCR system for processing and analyzing receipt documents using multiple OCR methods including PaddleOCR.

## Quick Setup (for collaborators)

### Prerequisites
- **Anaconda** or **Miniconda** installed ([download here](https://www.anaconda.com/download))
- **Git** installed

### Step 1: Clone the Repository
```bash
git clone https://github.com/circlejnoway/CAPSTONE_PROJECT_V2.git
cd CAPSTONE_PROJECT_V2
```

### Step 2: Create Conda Environment
```bash
conda env create -f environment.yml
conda activate capstone-ocr
```

### Step 3: Set Up Environment Variables
```bash
# Copy the template file
cp .env.example .env

# Edit .env with your API keys
# Windows:
notepad .env
# Mac/Linux:
nano .env
```

Add your API keys to `.env` (ask the project maintainer if you need them).

### Step 4: Verify Installation
```bash
python -c "import cv2, paddleocr; print('Setup successful!')"
```

## Project Structure
```
├── src/                    # Source code
├── models/                 # Trained models (git-ignored)
├── data/                   # Raw and processed data (git-ignored)
├── PaddleOCR/             # PaddleOCR dependencies (git-ignored)
├── app.py                 # Main application
├── compare_ocr_methods.py # OCR comparison script
├── ocr_quality_analyzer.py # Quality analysis tools
└── test_ocr_improvements.py # Test suite
```

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Creating branches for your changes
- Making commits
- Submitting pull requests
- Code standards

## API Keys
- **USDA_API_KEY**: For USDA data lookups
- **NINJAS_API_KEY**: For additional data sources
- **GEMINI_API_KEY**: For Google Gemini API

## Support
Contact the project maintainer for questions or issues.

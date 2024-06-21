# OCR Text Extraction Project

## Overview

This project extracts text from a set of 75 images using three different Optical Character Recognition (OCR) tools: **EasyOCR**, **PyTesseract**, and **TrOCR**. The results are then analyzed to identify discrepancies, and a consensus text is determined and corrected for common OCR errors. The final output is saved as a JSON file.

## Table of Contents

- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Error Handling](#error-handling)
- [Performance Optimization](#performance-optimization)
- [Output Quality](#output-quality)
- [Logging](#logging)
- [Future Improvements](#future-improvements)

## Project Structure

```
.
├── selected_images/            # Directory containing the 75 images
├── ocr_results.json            # Output JSON file with extracted and corrected text
├── ocr_processing.log          # Log file for tracking processing details and errors
├── main.py                     # Main script for executing OCR and text processing
└── README.md                   # This readme file
```

## Dependencies

The project requires the following libraries:

- `Pillow` (PIL)
- `easyocr`
- `pytesseract`
- `transformers` (for TrOCR)
- `torch`
- `concurrent.futures`
- `tqdm`

Ensure you have the required tools installed:

- Tesseract OCR: [Tesseract Installation Guide](https://github.com/tesseract-ocr/tesseract)
- Python 3.8 or higher

## Installation

1. **Clone the repository**:
   ```bash
   git clone [<repository-url>](https://github.com/yujansaya/raha_beach_ocr/)
   cd [<repository-directory>](https://github.com/yujansaya/raha_beach_ocr/)
   ```

2. **Install the Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure Tesseract OCR is installed**:
   - On Linux: `sudo apt-get install tesseract-ocr`
   - On macOS: `brew install tesseract`
   - On Windows: [Download the Tesseract installer](https://github.com/tesseract-ocr/tesseract/wiki)

## Usage

1. **Prepare the input images**:
   - Place the 75 images in the `selected_images/` directory.

2. **Run the main script**:
   ```bash
   python main.py
   ```

3. **View the results**:
   - The extracted and corrected text will be saved in `ocr_results.json`.
   - Logs can be found in `ocr_processing.log`.

### Example Command

```bash
python main.py
```

## Error Handling

The script includes comprehensive error handling:
- **Logging Errors**: Errors encountered during OCR processing are logged in `ocr_processing.log`.
- **Handling Missing Files**: Unsupported file formats or missing files are logged and skipped.

## Performance Optimization

To improve performance:
- **Parallel Processing**: Uses `ThreadPoolExecutor` for concurrent processing of images.
- **Batch Processing**: Processes images in batches for TrOCR to utilize GPU more efficiently.

## Output Quality

The `correct_text` function refines the OCR output by correcting common errors:
- **Symbol Confusion**: Handles confusion between `|`, `I`, `1`, `/`, and other symbols.

## Logging

The script logs detailed information about the processing steps and errors:
- **Log File**: `ocr_processing.log`
- **Log Levels**: Includes information and error levels.

## Future Improvements

- **Model Fine-Tuning**: Fine-tune OCR models on a dataset similar to the target images for better accuracy.
- **Additional OCR Tools**: Explore other OCR tools to enhance text extraction quality like Google Cloud Vision API, Amazon Textract or OpenCV.
- **Advanced Error Correction**: Implement machine learning models for context-based error correction.

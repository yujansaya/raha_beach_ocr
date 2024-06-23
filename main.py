import json
import os
import re
from PIL import Image
import easyocr
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
import logging
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from Levenshtein import distance as levenshtein_distance

# Paths
input_dir = "/selected_images"
output_file = "ocr_results.json"

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"Using device: {device}")

# Initialize OCR tools
easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())  # Enable GPU for EasyOCR if available
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update this path to your Tesseract installation
trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(device)  # Move TrOCR model to GPU if available

# Set up logging
logging.basicConfig(filename='ocr_processing.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def preprocess_image(image_path):
    # Load image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur to smooth edges
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply adaptive thresholding to binarize the image
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 7, 2
    )

    # Dilate to close gaps in text
    kernel = np.ones((3,3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)

    # Resize the image to enhance readability
    image = cv2.resize(image, (0,0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return image


# Helper function for TrOCR
def trocr_ocr(image_path):
    try:
        img = cv2.imread(image_path)
        pixel_values = trocr_processor(img, return_tensors="pt").pixel_values.to(device)  # Move input tensor to GPU if available
        generated_ids = trocr_model.generate(pixel_values)
        text_trocr = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text_trocr
    except Exception as e:
        logging.error(f"TrOCR failed for {image_path}: {e}")
        return ""


# Helper function for Easy OCR
def easyocr_ocr(image_path):
    try:
        img = cv2.imread(image_path)
        return " ".join(easyocr_reader.readtext(img, detail=0))
    except Exception as e:
        logging.error(f"EasyOCR failed for {image_path}: {e}")
        return ""


# Helper function for Tesseract
def pytesseract_ocr(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        text_pytesseract = pytesseract.image_to_string(img)
      
        # Tesseract is not very good in recognizing big thick letters and gives back empty string.
        # Let's tackle such cases with image pre-processing
        if text_pytesseract == "":
            text_pytesseract = pytesseract.image_to_string(preprocess_image(image_path))
        return text_pytesseract
    except Exception as e:
        logging.error(f"PyTesseract failed for {image_path}: {e}")
        return ""


# Function to compute the Levenshtein distance between strings
def similarity(a, b):
    return 1 - levenshtein_distance(a, b) / max(len(a), len(b))


# Function to apply majority voting and contextual analysis
def majority_vote(texts):
    if not texts:
        return ""

    # Tokenize each text into words and align lengths
    tokenized_texts = [text.split() for text in texts]
    max_len = max(len(tokens) for tokens in tokenized_texts)
    aligned_texts = [tokens + [''] * (max_len - len(tokens)) for tokens in tokenized_texts]

    # Combine tokens at each position using majority vote
    consensus_tokens = []
    for i in range(max_len):
        tokens_at_i = [tokens[i] for tokens in aligned_texts if tokens[i]]
        most_common_token = Counter(tokens_at_i).most_common(1)[0][0]
        consensus_tokens.append(most_common_token)

    return ' '.join(consensus_tokens).strip()


# Function to select the best OCR output
def select_best_text(texts):
    if not texts:
        return ""
    
    # Use Levenshtein distance to find the most consistent text
    best_text = max(texts, key=lambda text: sum(similarity(text, other) for other in texts if text != other))
    
    # Use majority vote as an additional check
    majority_text = majority_vote(texts)
    
    # If the best text and majority text are similar, return one of them
    if similarity(best_text, majority_text) > 0.9:
        return best_text
    
    # If there's a significant difference, log and choose the one with higher consensus
    logging.warning(f"Discrepancy found between best and majority texts: {best_text} vs {majority_text}")
    return best_text if similarity(best_text, majority_text) >= 0.5 else majority_text


# OCR processing function
def process_image(image_name):
    image_path = os.path.join(input_dir, image_name)
    if image_name.endswith('.jpg') or image_name.endswith('.png'):
        text_easyocr = easyocr_ocr(image_path)
        text_pytesseract = pytesseract_ocr(image_path)
        text_trocr = trocr_ocr(image_path)

        # Store raw texts
        raw_texts = [text_pytesseract,text_easyocr, text_trocr]
        print(raw_texts)
        
        # Determine consensus text using similarity comparison
        if any(raw_texts):
            best_text = select_best_text(raw_texts)
        else:
            best_text = ""

        # Log the selected text
        logging.info(f"Processed {image_name} with consensus text: {best_text}")
        
        return {
            "image_name": image_name,
            "text": best_text
        }
    else:
        logging.warning(f"Unsupported file format: {image_name}")
        return {
            "image_name": image_name,
            "text": ""
        }

# Main execution with ThreadPoolExecutor for performance
def main():
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, image_name) for image_name in os.listdir(input_dir)]
        for future in futures:
            result = future.result()
            if result:
                results.append(result)

    # Validate and sanitize JSON output
    for result in results:
        result["text"] = result["text"].strip()  # Trim leading/trailing whitespace
        print(result)
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    logging.info(f"Results saved to {output_file}")



if __name__ == "__main__":
    main()

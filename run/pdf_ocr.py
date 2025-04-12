#!/usr/bin/env python3
"""
Simple PDF OCR for Filipino Stories

This script extracts text from PDF files containing Filipino stories
using Optical Character Recognition (OCR).
Includes image preprocessing to improve accuracy.
"""

import os
import sys
import argparse
from pathlib import Path

# Check for required packages
try:
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    import cv2
    import numpy as np
except ImportError:
    print("Required packages not installed. Please install with:")
    print("pip install pdf2image pytesseract pillow opencv-python numpy")
    print("Note: You also need to install Tesseract OCR on your system.")
    print("For Filipino text, ensure the Filipino language pack is installed.")
    sys.exit(1)

# Define default directories
PDF_DIR = Path("pdf")
OUTPUT_DIR = Path("txt/uncleaned")

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def preprocess_image(image, enhance_contrast=1.5, sharpen=1.5, denoise=True, binarize=False):
    """
    Preprocess the image to improve OCR accuracy.
    
    Args:
        image: PIL Image object
        enhance_contrast: Contrast enhancement factor (1.0 = original)
        sharpen: Sharpness enhancement factor (1.0 = original)
        denoise: Whether to apply denoising
        binarize: Whether to convert to binary image (black and white)
        
    Returns:
        Processed PIL Image
    """
    # Convert PIL image to OpenCV format for advanced processing
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Denoise image if requested
    if denoise:
        img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)
    
    # Convert back to PIL for further processing
    processed_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    # Enhance contrast
    if enhance_contrast != 1.0:
        enhancer = ImageEnhance.Contrast(processed_img)
        processed_img = enhancer.enhance(enhance_contrast)
    
    # Sharpen the image
    if sharpen != 1.0:
        enhancer = ImageEnhance.Sharpness(processed_img)
        processed_img = enhancer.enhance(sharpen)
    
    # Binarize (convert to black and white) if requested
    if binarize:
        processed_img = processed_img.convert('L')  # Convert to grayscale
        # Use adaptive thresholding
        img_cv = cv2.cvtColor(np.array(processed_img), cv2.COLOR_RGB2GRAY)
        img_cv = cv2.adaptiveThreshold(
            img_cv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        processed_img = Image.fromarray(img_cv)
    
    return processed_img

def extract_text_from_pdf(pdf_path, output_path=None, dpi=300, 
                          preprocess=False, enhance_contrast=1.5, 
                          sharpen=1.5, denoise=True, binarize=False,
                          tesseract_config=''):
    """
    Extract text from a PDF file containing Filipino text using OCR.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path to save the extracted text (default: same name with .txt extension)
        dpi: DPI for image conversion (higher values give better quality but slower processing)
        preprocess: Whether to apply image preprocessing
        enhance_contrast: Contrast enhancement factor
        sharpen: Sharpness enhancement factor
        denoise: Whether to apply denoising
        binarize: Whether to convert to binary image
        tesseract_config: Additional Tesseract configuration
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        return
    
    # Set default output path if not provided
    if output_path is None:
        output_path = str(Path(pdf_path).with_suffix('.txt'))
    
    print(f"Processing PDF: {pdf_path}")
    
    # Convert PDF to images
    try:
        print("Converting PDF to images...")
        images = convert_from_path(pdf_path, dpi=dpi)
        print(f"PDF converted to {len(images)} images")
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return
    
    # Process each page to extract text
    extracted_text = []
    
    for page_num, image in enumerate(images):
        print(f"Processing page {page_num+1}/{len(images)}")
        
        # Preprocess the image if requested
        if preprocess:
            print(f"  Applying image preprocessing...")
            image = preprocess_image(
                image, 
                enhance_contrast=enhance_contrast,
                sharpen=sharpen,
                denoise=denoise,
                binarize=binarize
            )
        
        # Process the entire page with Filipino language settings
        try:
            # Configure Tesseract for best accuracy with Filipino text
            config = f'-l fil --psm 6 {tesseract_config}'
            text = pytesseract.image_to_string(image, config=config)
            
            if text.strip():
                extracted_text.append(text)
            else:
                print(f"  No text found on page {page_num+1}")
        except Exception as e:
            print(f"  Error performing OCR on page {page_num+1}: {e}")
    
    # Combine text from all pages and save
    if extracted_text:
        full_text = "\n\n".join(extracted_text)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        print(f"Extracted text saved to: {output_path}")
        print("Note: OCR is not 100% accurate. You may need to manually correct some text.")
        return full_text
    else:
        print("No text was extracted from the document")
        return ""

def main():
    parser = argparse.ArgumentParser(
        description="Extract text from PDF files containing Filipino stories with enhanced accuracy"
    )
    parser.add_argument(
        "input",
        help="PDF file to process (if only filename is provided, will look in pdf/ directory)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output text file (default: txt/uncleaned/<input_name>.txt)"
    )
    parser.add_argument(
        "--dpi",
        type=int, default=300,
        help="DPI for image conversion (higher values give better quality but slower processing)"
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Apply image preprocessing to improve OCR accuracy"
    )
    parser.add_argument(
        "--contrast",
        type=float, default=1.5,
        help="Contrast enhancement factor (1.0 = original)"
    )
    parser.add_argument(
        "--sharpen",
        type=float, default=1.5,
        help="Sharpness enhancement factor (1.0 = original)"
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable image denoising"
    )
    parser.add_argument(
        "--binarize",
        action="store_true",
        help="Convert image to black and white before OCR"
    )
    parser.add_argument(
        "--tesseract-config",
        default="",
        help="Additional Tesseract configuration options"
    )
    
    args = parser.parse_args()
    
    # Handle input path
    input_path = Path(args.input)
    if not input_path.is_absolute() and not str(input_path).startswith(('/', './')):
        # If just a filename is provided, look in the pdf directory
        input_path = PDF_DIR / input_path
    
    # Handle output path
    output_path = args.output
    if output_path is None:
        # Use default output directory with same name as input
        output_path = OUTPUT_DIR / input_path.name.replace('.pdf', '.txt')
    
    # Process the PDF file
    extract_text_from_pdf(
        str(input_path),
        str(output_path),
        dpi=args.dpi,
        preprocess=args.preprocess,
        enhance_contrast=args.contrast,
        sharpen=args.sharpen,
        denoise=not args.no_denoise,
        binarize=args.binarize,
        tesseract_config=args.tesseract_config
    )

if __name__ == "__main__":
    main() 
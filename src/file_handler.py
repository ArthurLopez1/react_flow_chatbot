import os
import re
from PyPDF2 import PdfReader
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import logging
import camelot
import unittest
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

# ---------------------------
# Constants and Configuration
# ---------------------------

METADATA_FILE = "processed_files_metadata.json"
CHUNKS_DIR = "./data/chunks"
FIGURES_DIR = "./data/figures"
LOG_DIR = "./log"
LOG_FILE = os.path.join(LOG_DIR, "file_handler.log")

# Ensure the log directory exists
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# ---------------------------
# Helper Functions
# ---------------------------

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_image(image):
    grayscale = image.convert('L')
    return grayscale.point(lambda x: 0 if x < 140 else 255, '1')

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_metadata(metadata):
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)

# ---------------------------
# Parsing Functions
# ---------------------------

def parse_embedded_text(pdf_path):
    text_data = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                page_number = page_index + 1
                tables = page.find_tables()
                table_bboxes = [table.bbox for table in tables]
                all_words = page.extract_words()
                filtered_words = [word for word in all_words if not any(
                    word['x0'] >= bbox[0] and word['x1'] <= bbox[2] and
                    word['top'] >= bbox[1] and word['bottom'] <= bbox[3]
                    for bbox in table_bboxes
                )]
                if not filtered_words:
                    continue
                lines = {}
                for word in filtered_words:
                    assigned = False
                    for line_top in lines:
                        if abs(word['top'] - line_top) <= 3:
                            lines[line_top].append((word['x0'], word['text']))
                            assigned = True
                            break
                    if not assigned:
                        lines[word['top']] = [(word['x0'], word['text'])]
                combined_text = " ".join(
                    clean_text(' '.join(w[1] for w in sorted(words, key=lambda x: x[0])))
                    for _, words in sorted(lines.items(), key=lambda x: x[0])
                )
                if combined_text:
                    text_data.append({
                        "source": os.path.basename(pdf_path),
                        "content": combined_text.strip(),
                        "page_number": page_number
                    })
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
    return text_data

def parse_tables_pdfplumber(pdf_path):
    all_tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                page_number = page_index + 1
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        all_tables.append({
                            "source": os.path.basename(pdf_path),
                            "content": table,
                            "page_number": page_number
                        })
    except Exception as e:
        logging.error(f"Error extracting tables from {pdf_path}: {e}")
    return all_tables

def parse_tables_camelot(pdf_path):
    tables_data = []
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
        if not tables:
            logging.warning(f"No tables found in {pdf_path} using Camelot.")
            return tables_data
        for table in tables:
            df = table.df
            headers = df.iloc[0].tolist()
            table_rows = [
                {header: cell for header, cell in zip(headers, row.tolist())}
                for _, row in df.iloc[1:].iterrows()
            ]
            tables_data.append({
                "source": os.path.basename(pdf_path),
                "content": table_rows,
                "page_number": table.page
            })
    except Exception as e:
        logging.error(f"Error extracting tables with Camelot from {pdf_path}: {e}")
    return tables_data

def perform_ocr(pdf_path):
    text_data = []
    try:
        images = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        logging.error(f"Error converting {pdf_path} to images: {e}")
        return text_data
    for page_number, image in enumerate(images, start=1):
        try:
            preprocessed_image = preprocess_image(image)
            ocr_text = pytesseract.image_to_string(preprocessed_image, lang='eng')
            cleaned_text = clean_text(ocr_text)
            if cleaned_text:
                text_data.append({
                    "source": os.path.basename(pdf_path),
                    "content": cleaned_text,
                    "page_number": page_number
                })
        except Exception as e:
            logging.error(f"OCR failed on page {page_number} of {pdf_path}: {e}")
    return text_data

def extract_figures(pdf_path):
    figures_data = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                page_number = page_index + 1
                images = page.images
                for image_index, image in enumerate(images):
                    x0, y0, x1, y1 = image["x0"], image["top"], image["x1"], image["bottom"]
                    # Check if the bounding box is within the page's bounding box
                    if x0 < 0 or y0 < 0 or x1 > page.width or y1 > page.height:
                        logging.warning(f"Skipping image with bounding box {image['x0'], image['top'], image['x1'], image['bottom']} on page {page_number} of {pdf_path} due to invalid coordinates.")
                        continue
                    figure = page.within_bbox((x0, y0, x1, y1)).to_image()
                    figure_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page{page_number}_figure{image_index}.png"
                    figure_filepath = os.path.join(FIGURES_DIR, figure_filename)
                    figure.save(figure_filepath, format="PNG")
                    figure_metadata = {
                        "source": os.path.basename(pdf_path),
                        "page_number": page_number,
                        "figure_file": figure_filepath
                    }
                    metadata_filename = f"{os.path.splitext(figure_filename)[0]}.json"
                    metadata_filepath = os.path.join(FIGURES_DIR, metadata_filename)
                    with open(metadata_filepath, 'w', encoding='utf-8') as metadata_file:
                        json.dump(figure_metadata, metadata_file, ensure_ascii=False, indent=4)
                    figures_data.append(figure_metadata)
    except Exception as e:
        logging.error(f"Error extracting figures from {pdf_path}: {e}")
    return figures_data

def parse_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        logging.error(f"File not found: {pdf_path}")
        return None, None, None
    embedded_text = parse_embedded_text(pdf_path)
    tables = parse_tables_pdfplumber(pdf_path)
    figures = extract_figures(pdf_path)
    if not tables:
        logging.info(f"No tables found with pdfplumber in {pdf_path}. Trying Camelot.")
        tables = parse_tables_camelot(pdf_path)
    if not any(page["content"] for page in embedded_text):
        logging.info(f"No embedded text found in {pdf_path}. Performing OCR.")
        return perform_ocr(pdf_path), tables, figures
    return embedded_text, tables, figures

def process_pdfs_in_folder(folder_path):
    all_parsed_data = []
    metadata = load_metadata()

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            last_modified = os.path.getmtime(pdf_path)

            # Check if the file has already been processed and if it has been modified since then
            if filename in metadata and metadata[filename] == last_modified:
                logging.info(f"Skipping {pdf_path} as it has already been processed.")
                continue

            logging.info(f"Processing {pdf_path}...")
            parsed_text, parsed_tables, parsed_figures = parse_pdf(pdf_path)

            if not parsed_text and not parsed_tables and not parsed_figures:
                logging.warning(f"Skipping {filename} due to no extracted data.")
                continue

            all_parsed_data.append({
                "pdf_file": filename,
                "text": parsed_text,
                "tables": parsed_tables,
                "figures": parsed_figures
            })
            logging.info(f"Extracted data from {filename}.")

            # Update metadata
            metadata[filename] = last_modified

    # Save metadata after processing all files
    save_metadata(metadata)

    return all_parsed_data

def get_data_from_specific_page(parsed_data, target_page):
    specific_page_data = []
    for data in parsed_data:
        pdf_file = data['pdf_file']
        texts = [text for text in data['text'] if text['page_number'] == target_page]
        tables = [table for table in data['tables'] if table['page_number'] == target_page]
        figures = [figure for figure in data['figures'] if figure['page_number'] == target_page]
        if texts or tables or figures:
            specific_page_data.append({
                "pdf_file": pdf_file,
                "text": texts,
                "tables": tables,
                "figures": figures
            })
    return specific_page_data

def split_text_into_chunks(parsed_data, chunk_size=1000, chunk_overlap=100):
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    for pdf_data in parsed_data:
        pdf_file = pdf_data['pdf_file']
        text_entries = pdf_data.get('text', [])
        for text_entry in text_entries:
            page_number = text_entry.get('page_number')
            source = text_entry.get('source')
            content = text_entry.get('content', '')
            if content:
                splits = text_splitter.split_text(content)
                for split_index, split in enumerate(splits):
                    chunk_filename = f"{os.path.splitext(pdf_file)[0]}_page{page_number}_chunk{split_index}.json"
                    chunk_filepath = os.path.join(CHUNKS_DIR, chunk_filename)
                    chunk_data = {
                        'chunk_content': split,
                        'page_number': page_number,
                        'source': source,
                        'chunk_file': chunk_filepath
                    }
                    with open(chunk_filepath, 'w', encoding='utf-8') as chunk_file:
                        json.dump(chunk_data, chunk_file, ensure_ascii=False, indent=4)
                    all_chunks.append(chunk_data)
    return all_chunks

# ---------------------------
# Main Execution Block
# ---------------------------

if __name__ == "__main__":
    folder_path = "./data"
    
    # Create chunks and figures directories if they don't exist
    if not os.path.exists(CHUNKS_DIR):
        os.makedirs(CHUNKS_DIR)
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)
    
    parsed_data = process_pdfs_in_folder(folder_path)
    chunked_data = split_text_into_chunks(parsed_data)
    
    # Print a sample chunk for inspection
    if chunked_data:
        sample_chunk = chunked_data[0]
        print("Sample Chunk:")
        print(f"Content: {sample_chunk['chunk_content']}\n")
        print(f"Page Number: {sample_chunk['page_number']}")
        print(f"Source: {sample_chunk['source']}")
        print(f"Chunk File: {sample_chunk['chunk_file']}")
    else:
        print("No chunks were generated.")
import logging
import re
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import pytesseract
from pdf2image import convert_from_path
import fitz
import camelot
from typing import List

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean text by removing HTML tags, excessive whitespace, and formatting artifacts.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    # Replace multiple spaces with a single space
    text = re.sub(r' {2,}', ' ', text)
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n{2,}', '\n', text)
    # Remove leading and trailing whitespace on each line
    text = '\n'.join(line.strip() for line in text.splitlines())
    # Remove lines that are empty or contain only non-alphanumeric characters
    text = '\n'.join(line for line in text.splitlines() if line and re.search(r'\w', line))
    # Normalize spaces around punctuation
    text = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', text)
    
    return text

def parse_pdf_with_pymupdf(pdf_path: str) -> List[Document]:
    """
    Extracts embedded text from a PDF using PyMuPDF and identifies tables.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        List[Document]: A list of extracted documents with metadata.
    """
    logger = logging.getLogger(__name__)
    documents = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Error reading PDF file {pdf_path}: {e}")
        return documents

    for page_num in range(len(doc)):
        try:
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text:
                cleaned_text = clean_text(text)
                documents.append(Document(
                    page_content=cleaned_text, 
                    metadata={
                        "page_number": page_num + 1,
                        "source": "embedded_text"
                    }
                ))
                logger.info(f"Extracted and cleaned text from page {page_num + 1} using PyMuPDF.")
            else:
                logger.warning(f"No embedded text found on page {page_num + 1}. Skipping.")
            
            # Check for tables on the page
            tables = page.get_text("dict")["blocks"]
            for block in tables:
                if block["type"] == 0 and "lines" in block:
                    table_text = "\n".join([" ".join(line["spans"][0]["text"] for line in block["lines"] if "spans" in line) for block in tables if block["type"] == 0])
                    documents.append(Document(
                        page_content=table_text,
                        metadata={
                            "page_number": page_num + 1,
                            "source": "table"
                        }
                    ))
                    logger.info(f"Extracted table from page {page_num + 1} using PyMuPDF.")
        except Exception as e:
            logger.error(f"Error extracting text from page {page_num + 1} in {pdf_path}: {e}")

    logger.info(f"Total embedded documents extracted from {pdf_path}: {len(documents)}")
    return documents

def split_document(document: Document, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split a document into chunks for processing, using LangChain's RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents([document])
    return chunks

def parse_pdf_with_ocr(pdf_path: str, page_num: int) -> List[Document]:
    """
    Extracts text from a PDF using OCR and converts each page to a LangChain Document.
    """
    images = convert_from_path(pdf_path, dpi=300)
    documents = []

    for page_num, image in enumerate(images):
        ocr_text = pytesseract.image_to_string(image, lang='swe+eng')
        if ocr_text:
            try:
                cleaned_text = clean_text(ocr_text)
                documents.append(Document(
                    page_content=ocr_text, 
                    metadata={
                        "page_number": page_num,
                        "source": "ocr"
                    }
                ))
                logger.info(f"Extracted and cleaned text from page {page_num + 1} using OCR.")
            except ValueError:
                logger.warning(f"OCR on page {page_num + 1} resulted in empty text after cleaning. Skipping.")
        else:
            logger.warning(f"No text found on page {page_num + 1} after OCR. Skipping.")

    logger.info(f"Total documents extracted from {pdf_path} using OCR: {len(documents)}")
    return documents

def parse_tables(pdf_path: str) -> List[Document]:
    """
    Extracts tables from a PDF and converts them to LangChain Documents.
    """
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')  # Use 'stream' or 'lattice' based on table structure
    except Exception as e:
        logger.error(f"Error extracting tables: {e}")
        return []

    table_documents = []

    for i, table in enumerate(tables):
        table_text = table.df.to_string(index=False)
        table_documents.append(Document(
            page_content=table_text,
            metadata={
                "table_number": i + 1,
                "source": "table"
            }
        ))
        logger.info(f"Extracted table {i + 1} from PDF.")

    logger.info(f"Total tables extracted: {len(table_documents)}")
    return table_documents


def deduplicate_documents(documents: List[Document]) -> List[Document]:
    seen = {}
    unique_docs = []
    for doc in documents:
        doc_hash = hash(doc.page_content)
        if doc_hash not in seen:
            seen[doc_hash] = doc
            unique_docs.append(doc)
        else:
            # Optional: Compare sources and decide whether to replace or keep the existing document
            existing_doc = seen[doc_hash]
            if existing_doc.metadata.get("source") == "ocr" and doc.metadata.get("source") == "embedded":
                seen[doc_hash] = doc
                unique_docs[-1] = doc
    return unique_docs

def load_all_content(pdf_path: str) -> List[Document]:
    """
    Extracts text, tables, and performs OCR on a PDF.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        List[Document]: A combined list of all extracted documents after deduplication.
    """
    text_docs = parse_pdf_with_pymupdf(pdf_path)
    table_docs = parse_tables(pdf_path)
    
    # Determine pages without embedded text for OCR
    embedded_pages = {doc.metadata["page_number"] for doc in text_docs}
    total_pages = len(text_docs)  # Assuming one document per page
    ocr_pages = set(range(1, total_pages + 1)) - embedded_pages
    
    ocr_docs = []
    for page in ocr_pages:
        ocr_docs.extend(parse_pdf_with_ocr(pdf_path, page))
    
    all_documents = text_docs + table_docs + ocr_docs
    all_documents = deduplicate_documents(all_documents)
    
    logger.info(f"Total combined documents after deduplication: {len(all_documents)}")
    return all_documents

def load_document(file_path: str) -> List[Document]:
    """
    Load a document from a file path. Supports PDF files with text and tables extraction.
    """
    documents = []

    if file_path.lower().endswith('.pdf'):
        documents.extend(parse_pdf_with_pymupdf(file_path))
        documents.extend(parse_tables(file_path))

    return documents

def test_load_and_print_document(file_path: str):
    """
    Test function to load and print a document after processing.
    """
    documents = load_document(file_path)
    for doc in documents:
        content_source = doc.metadata.get('source', 'unknown')
        if content_source in ['embedded_text', 'ocr']:
            # Apply cleaning to embedded text and OCR outputs
            cleaned_content = clean_text(doc.page_content)
            print(f"Metadata: {doc.metadata}")
            print(f"Content:\n{cleaned_content}\n")
        else:
            # For tables and other content, do not clean
            print(f"Metadata: {doc.metadata}")
            print(f"Content:\n{doc.page_content}\n")

if __name__ == "__main__":
    pdf_path = "./data/ersattningsmodell_vaders_2019.pdf"
    
    print("Testing parse_pdf_with_pymupdf:")
    test_load_and_print_document(pdf_path)
    
    #print("\nTesting parse_tables:")
    #test_load_and_print_document(pdf_path)
import ssl
import urllib.request

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context


import logging
import re
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
from typing import List
from docling.document_converter import DocumentConverter


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

def parse_pdf(pdf_path: str) -> List[Document]:
    """
    Extracts text from a PDF using Docling and converts each page to a LangChain Document.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        List[Document]: A list of extracted documents with metadata.
    """
    logger = logging.getLogger(__name__)
    documents = []

    try:
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
    except Exception as e:
        logger.error(f"Error reading PDF file {pdf_path}: {e}")
        return documents

    for page_num, page in enumerate(result.document.pages, start=1):
        try:
            # Extract text
            text = page.text
            if text:
                cleaned_text = clean_text(text)
                document = Document(
                    page_content=cleaned_text, 
                    metadata={
                        "page_number": page_num,
                        "source": "embedded_text"
                    }
                )
                documents.extend(split_document(document))
                logger.info(f"Extracted, cleaned, and split text from page {page_num} using Docling.")
            else:
                logger.warning(f"No embedded text found on page {page_num}. Skipping.")
        except Exception as e:
            logger.error(f"Error extracting content from page {page_num} in {pdf_path}: {e}")

    logger.info(f"Total documents extracted from {pdf_path}: {len(documents)}")
    return documents

def split_document(document: Document, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split a document into smaller chunks using RecursiveCharacterTextSplitter.
    
    Args:
        document (Document): The document to split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The overlap between chunks.
    
    Returns:
        List[Document]: A list of split documents.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(document.page_content)
    return [Document(page_content=chunk, metadata=document.metadata) for chunk in chunks]

def load_all_content(pdf_path: str) -> List[Document]:
    """
    Extracts text from a PDF.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        List[Document]: A list of extracted documents.
    """
    text_docs = parse_pdf(pdf_path)
    # Removed OCR processing and deduplication
    all_documents = text_docs
    logger.info(f"Total documents extracted from {pdf_path}: {len(all_documents)}")
    return all_documents

def load_document(file_path: str) -> List[Document]:
    """
    Load a document from a file path. Supports PDF files with text extraction.
    """
    documents = []

    if file_path.lower().endswith('.pdf'):
        documents.extend(parse_pdf(file_path))

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

def test_parse_pdf_markdown():
    """
    Test function to parse Vägklimatologi.pdf and print its markdown representation.
    """
    pdf_path = "./data/Vägklimatologi.pdf"
    documents = parse_pdf(pdf_path)
    for doc in documents:
        # Assuming DocumentConverter can export to markdown
        markdown_output = DocumentConverter().convert_markdown(doc.page_content)
        print(f"Metadata: {doc.metadata}")
        print(f"Markdown Content:\n{markdown_output}\n")

if __name__ == "__main__":
    pdf_path = "./data/Vägklimatologi.pdf"
    
    source = pdf_path  # PDF path or URL
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    print(result.document.export_to_markdown())  #
    # Uncomment the following line to run the markdown test
    # test_parse_pdf_markdown()
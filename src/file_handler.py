import ssl
import urllib.request

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

import logging
import re
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

def split_document(document: Document, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split a markdown document into smaller chunks while retaining metadata,
    ensuring that each chunk is associated with its respective page number.

    Args:
        document (Document): The document to split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The overlap between chunks.

    Returns:
        List[Document]: A list of split documents.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(document.page_content)
    logger = logging.getLogger(__name__)

    split_documents = [
        Document(
            page_content=chunk,
            metadata={
                **document.metadata,
                "chunk_index": idx,
                "source": "markdown_split",
                "page_number": document.metadata.get("page_number", "unknown")
            }
        )
        for idx, chunk in enumerate(chunks)
    ]

    logger.info(f"Split document from page {document.metadata.get('page_number', 'unknown')} into {len(split_documents)} chunks.")
    return split_documents

def parse_pdf(pdf_path: str) -> List[Document]:
    """
    Parses a PDF file using Docling, exports the entire document to markdown, and converts
    it to LangChain Documents with metadata.
    
    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        List[Document]: List of structured documents.
    """
    logger = logging.getLogger(__name__)
    documents = []

    try:
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        docling_doc = result.document
        
        for page_num, page in enumerate(docling_doc.pages, start=1):
            markdown_content = page.export_to_markdown()
            document = Document(
                page_content=markdown_content,
                metadata={"source": "markdown", "page_number": page_num}
            )
            documents.extend(split_document(document))
        logger.info(f"Successfully processed {pdf_path} into {len(documents)} chunks.")
    except Exception as e:
        logger.error(f"Error parsing PDF {pdf_path}: {e}")

    return documents

def load_all_content(pdf_path: str) -> List[Document]:
    """
    Extracts text from a PDF.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        List[Document]: A list of extracted documents.
    """
    text_docs = parse_pdf(pdf_path)
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

def test_parse_pdf_markdown():
    """
    Test function to parse Vägklimatologi.pdf and print its markdown representation.
    """
    pdf_path = "./data/Vägklimatologi.pdf"
    documents = parse_pdf(pdf_path)
    for doc in documents:
        print(f"Metadata: {doc.metadata}")
        print(f"Content:\n{doc.page_content}\n")

if __name__ == "__main__":
    pdf_path = "./data/Vägklimatologi.pdf"
    
    # Parse the PDF and print the resulting documents
    documents = parse_pdf(pdf_path)
    for doc in documents:
        print(f"Metadata: {doc.metadata}")
        print(f"Content:\n{doc.page_content}\n")
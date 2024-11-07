import re
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

def clean_text(text: str) -> str:
    """
    Clean text by removing excessive whitespace and newlines.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    # Remove excessive newlines and leading/trailing whitespace
    text = re.sub(r'\n+', '\n', text).strip()
    # Replace multiple spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)

    return text

def parse_pdf_with_pypdf(pdf_path: str):
    """
    Extracts text from a PDF and converts each page to a LangChain Document.
    """
    reader = PdfReader(pdf_path)
    documents = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            cleaned_text = clean_text(text)  # Clean the text immediately after extraction
            documents.append(Document(page_content=cleaned_text, metadata={"page_number": page_num + 1}))

    return documents

def split_document(document: Document, chunk_size: int = 1500, chunk_overlap: int = 300):
    """
    Split a document into chunks for processing, using LangChain's RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents([document])

def load_document(file_path: str):
    """
    Load a document from a file path. Currently supports PDF files.
    """
    if file_path.lower().endswith('.pdf'):
        return parse_pdf_with_pypdf(file_path)
    else:
        raise ValueError("Unsupported file format. Only PDF files are supported.")

# Example usage
if __name__ == "__main__":
    pdf_path = "./data/ersattningsmodell_vaders_2019.pdf"
    docs_list = parse_pdf_with_pypdf(pdf_path)
    print(f"Extracted {len(docs_list)} pages from PDF.")

    # Split each document into chunks
    doc_splits = []
    for doc in docs_list:
        doc_splits.extend(split_document(doc))

    print(f"Split documents into {len(doc_splits)} chunks.")

#import ssl
#import urllib.request

# Disable SSL verification
#sl._create_default_https_context = ssl._create_unverified_context

import re
import json
import logging
import time
from pathlib import Path
from typing import List
import faulthandler

from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.tesseract_ocr_cli_model import TesseractCliOcrOptions
from docling.models.tesseract_ocr_model import TesseractOcrOptions

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from fuzzysearch import find_near_matches

logger = logging.getLogger(__name__)

input_doc_path = Path("./data/Vägklimatologi.pdf")

#def clean_text(text: str) -> str:
#    """
#    Clean text by removing HTML tags, excessive whitespace, and formatting artifacts.
#    """
#    if not isinstance(text, str):
#        raise ValueError("Input must be a string.")
#    
#    # Remove HTML tags
#    text = re.sub(r'<[^>]+>', '', text)
#    # Replace tabs with spaces
#    text = text.replace('\t', ' ')
#    # Replace multiple spaces with a single space
#    text = re.sub(r' {2,}', ' ', text)
#    # Replace multiple newlines with a single newline
#    text = re.sub(r'\n{2,}', '\n', text)
#    # Remove leading and trailing whitespace on each line
#    text = '\n'.join(line.strip() for line in text.splitlines())
#    # Remove lines that are empty or contain only non-alphanumeric characters
#    text = '\n'.join(line for line in text.splitlines() if line and re.search(r'\w', line))
#    # Normalize spaces around punctuation
#    text = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', text)
#    
#    return text

def split_document(document: Document, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split a markdown document into smaller chunks while retaining metadata.
    
    Args:
        document (Document): The document to split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The overlap between chunks.
    
    Returns:
        List[Document]: A list of split documents.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(document.page_content)

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

    logging.info(f"Split document into {len(split_documents)} chunks.")
    return split_documents

def parse_pdf():
    """ Parse a PDF file into Markdown and JSON formats. """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    input_doc_path = Path("./data/ersattningsmodell_vaders_2019.pdf")
    if not input_doc_path.exists():
        logger.error(f"Input PDF file does not exist: {input_doc_path}")
        return

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    start_time = time.time()
    conv_result = doc_converter.convert(input_doc_path)
    end_time = time.time() - start_time

    logger.info(f"Document converted in {end_time:.2f} seconds.")

    # Export results
    output_dir = Path("./data/cleaned_docs")
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = conv_result.input.file.stem

    # Export document JSON format
    json_output_path = output_dir / f"{doc_filename}.json"
    with json_output_path.open("w", encoding="utf-8") as fp:
        json_content = conv_result.document.export_to_dict()
        fp.write(json.dumps(json_content, ensure_ascii=False, indent=2))
    logger.info(f"Exported JSON to {json_output_path}.")

    # Export Markdown format
    markdown_output_path = output_dir / f"{doc_filename}.md"
    with markdown_output_path.open("w", encoding="utf-8") as fp:
        markdown_content = conv_result.document.export_to_markdown()
        fp.write(markdown_content)
    logger.info(f"Exported Markdown to {markdown_output_path}.")

    logger.info("PDF parsing completed successfully.")





def split_markdown_into_chunks(markdown_content: str, chunk_size: int = 1500, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(markdown_content)
    logging.info(f"Markdown content split into {len(chunks)} chunks.")
    return chunks

def get_page_texts_from_json(json_data):
    print("json_data keys:", json_data.keys())
    print("json_data type:", type(json_data))
    
    texts = json_data.get('texts', [])
    print("Type of json_data['texts']:", type(texts))
    if texts:
        print("First item in json_data['texts']:", texts[0])
    
    page_texts = {}
    for text_item in texts:
        text_content = text_item.get('text', '')
        prov_list = text_item.get('prov', [])
        for prov in prov_list:
            page_number = prov.get('page_no')
            if page_number is not None:
                # Accumulate text content for the page
                if page_number in page_texts:
                    page_texts[page_number] += ' ' + text_content
                else:
                    page_texts[page_number] = text_content
    return page_texts

def build_full_text_and_page_ranges(page_texts_ordered):
    full_text = ''
    page_ranges = {}
    current_pos = 0
    for page_number, page_text in page_texts_ordered:
        start_pos = current_pos
        full_text += page_text
        current_pos += len(page_text)
        end_pos = current_pos
        page_ranges[page_number] = (start_pos, end_pos)
    return full_text, page_ranges

def assign_metadata_to_chunks_by_position(chunks, full_text, page_ranges, doc_name):
    chunks_with_metadata = []
    current_pos = 0

    for chunk_index, chunk_text in enumerate(chunks):
        # Attempt exact match first
        chunk_start = full_text.find(chunk_text, current_pos)
        if chunk_start == -1:
            # Try fuzzy matching
            matches = find_near_matches(chunk_text, full_text, max_l_dist=10)
            if matches:
                match = matches[0]
                chunk_start = match.start
                chunk_end = match.end
                # Update current position
                current_pos = chunk_end
            else:
                # Chunk not found
                chunk_start = None
        else:
            chunk_end = chunk_start + len(chunk_text)
            # Update current position
            current_pos = chunk_end

        if chunk_start is not None:
            # Find overlapping pages
            overlapping_pages = []
            for page_number, (start_pos, end_pos) in page_ranges.items():
                if chunk_start < end_pos and chunk_end > start_pos:
                    overlapping_pages.append(page_number)
            page_numbers = overlapping_pages if overlapping_pages else None
        else:
            page_numbers = None
            logging.warning(f"Chunk {chunk_index} not found in full_text.")

        chunk_metadata = {
            'doc_name': doc_name,
            'chunk_index': chunk_index,
            'page_numbers': page_numbers,
            'text': chunk_text
        }
        chunks_with_metadata.append(chunk_metadata)

    return chunks_with_metadata

    return chunks_with_metadata

def process_and_assign_metadata():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Paths to the JSON file generated earlier
    doc_filename = "ersattningsmodell_vaders_2019"
    output_dir = Path("./data/cleaned_docs")
    json_path = output_dir / f"{doc_filename}.json"

    # Read the JSON content
    if not json_path.exists():
        logger.error(f"JSON file does not exist: {json_path}")
        return

    with json_path.open("r", encoding="utf-8") as json_fp:
        json_data = json.load(json_fp)

    # Extract page texts
    page_texts = get_page_texts_from_json(json_data)

    # Verify extracted page texts
    for page_number, text in page_texts.items():
        print(f"Page {page_number} Text Preview: {text[:100]}...")

    # Ensure pages are ordered by page number
    page_texts_ordered = sorted(page_texts.items())

    # Build full text and page ranges
    full_text, page_ranges = build_full_text_and_page_ranges(page_texts_ordered)

    # Split the full text into chunks
    chunks = split_markdown_into_chunks(full_text)

    print("\nFull Text Preview:")
    print(full_text[:500])

    # Assign metadata to chunks
    chunks_with_metadata = assign_metadata_to_chunks_by_position(
        chunks, full_text, page_ranges, doc_filename + ".pdf"
    )

    # Output results
    # Write the chunks with metadata to a JSON file
    chunks_output_path = output_dir / f"{doc_filename}_chunks_with_metadata.json"
    with chunks_output_path.open("w", encoding="utf-8") as fp:
        json.dump(chunks_with_metadata, fp, ensure_ascii=False, indent=2)
    logger.info(f"Exported chunks with metadata to {chunks_output_path}.")

    # Optionally, write the chunks to a Markdown file
    chunks_markdown_path = output_dir / f"{doc_filename}_chunks.md"
    with chunks_markdown_path.open("w", encoding="utf-8") as md_fp:
        for chunk in chunks_with_metadata:
            pages = chunk['page_numbers']
            md_fp.write(
                f"\n\n---\n\n### Chunk {chunk['chunk_index']} (Pages {pages})\n\n{chunk['text']}\n"
            )
    logger.info(f"Exported chunks to Markdown file {chunks_markdown_path}.")

if __name__ == "__main__":
    process_and_assign_metadata()
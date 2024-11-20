import logging
import time
from pathlib import Path
from typing import List
import os

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)

def parse_pdfs_in_data_folder():
    """Parse all PDF files in the data folder into Markdown and export chunks."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    data_folder = Path("./data")
    output_dir = Path("./data/cleaned_docs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get a list of all PDF files in the data folder
    pdf_files = list(data_folder.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in the data folder.")
        return

    for pdf_file in pdf_files:
        pdf_filename = pdf_file.stem
        markdown_output_path = output_dir / f"{pdf_filename}.md"

        # Check if the markdown file already exists
        if (markdown_output_path.exists()):
            logger.info(f"Markdown file already exists for {pdf_file.name}, skipping parsing.")
            continue

        logger.info(f"Processing PDF file: {pdf_file.name}")

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = False

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        start_time = time.time()
        try:
            conv_result = doc_converter.convert(pdf_file)
            end_time = time.time() - start_time

            logger.info(f"Document {pdf_file.name} converted in {end_time:.2f} seconds.")

            # Export Markdown format
            with markdown_output_path.open("w", encoding="utf-8") as fp:
                markdown_content = conv_result.document.export_to_markdown()
                fp.write(markdown_content)
            logger.info(f"Exported Markdown to {markdown_output_path}.")
        except Exception as e:
            logger.error(f"Failed to convert {pdf_file.name}: {e}", exc_info=True)

    logger.info("Finished processing all PDF files in data folder.")

def split_markdown_files(chunk_size=900, chunk_overlap=150) -> List[Document]:
    """
    Split all markdown files in the cleaned_docs directory into smaller chunks while preserving the structure.
    
    Returns:
        List[Document]: List of split documents with metadata.
    """
    output_dir = Path("./data/cleaned_docs")
    markdown_files = list(output_dir.glob("*.md"))

    if not markdown_files:
        logger.warning(f"No Markdown files found in {output_dir}")
        return []

    all_splits = []

    for md_file in markdown_files:
        logger.info(f"Processing Markdown file: {md_file.name}")

        # Read the content of the Markdown file
        markdown_document = md_file.read_text(encoding="utf-8")

        # Define headers to split on
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]

        # Split based on headers
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(markdown_document)

        # Further split based on character count
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(md_header_splits)

        # Add metadata
        for doc in splits:
            doc.metadata['source_file'] = md_file.name
            # Retain existing metadata from header splits
            # doc.metadata already contains metadata from md_header_splits

        all_splits.extend(splits)
        logger.info(f"Split Markdown file {md_file.name} into {len(splits)} chunks.")

    logger.info(f"Total chunks from all Markdown files: {len(all_splits)}")
    return all_splits

def load_documents():
    """
    Load and split documents from the data folder.

    Returns:
        List[Document]: A list of split documents ready for processing.
    """
    # First, parse PDFs and export to Markdown if not already done
    parse_pdfs_in_data_folder()

    # Then, split all Markdown files into chunks
    documents = split_markdown_files()

    return documents

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    documents = load_documents()
    logger.info(f"Loaded a total of {len(documents)} documents ready for embedding.")

    # Now you can pass `documents` to vectorstore or further processing
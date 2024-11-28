import logging
from pathlib import Path
from file_handler import process_pdfs_in_folder, split_text_into_chunks
from vectorstore import VectorStoreManager
from langchain.schema import Document

# Configure logging to display INFO level messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_on_documents(data_folder):
    vector_store = VectorStoreManager()
    documents = []

    # Process PDFs in the data folder using file_handler.py
    data_folder = Path(data_folder)
    pdf_files = list(data_folder.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {data_folder}.")
        return

    # Step 1: Process PDFs and get parsed data
    parsed_data = process_pdfs_in_folder(str(data_folder))

    # Step 2: Split text into chunks
    chunked_data = split_text_into_chunks(parsed_data)

    if not chunked_data:
        logger.warning("No chunks were generated from the PDFs.")
        return

    # Convert chunked data to Document objects
    for chunk in chunked_data:
        doc = Document(
            page_content=chunk["chunk_content"],
            metadata={
                "page_number": chunk["page_number"],
                "source": chunk["source"],
                "chunk_file": chunk["chunk_file"]
            }
        )
        documents.append(doc)

    num_docs = len(documents)
    logger.info(f"Loaded {num_docs} documents from {data_folder}.")

    if documents:
        # Add documents to the vector store
        vector_store.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to the vector store.")
    else:
        logger.warning(f"No documents extracted from {data_folder}.")

    # Summary of training
    logger.info("Training completed.")
    logger.info(f"Total documents loaded: {num_docs}")
    logger.info(f"Total documents added to vector store: {len(documents)}")

# Add a main guard to execute the training when the script is run directly
if __name__ == "__main__":
    data_folder = "./data"  # Adjust the path to your data folder if necessary
    train_on_documents(data_folder)
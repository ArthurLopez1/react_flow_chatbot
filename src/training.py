import logging
import os  
from file_handler import load_document
from vectorstore import VectorStoreManager

# Configure logging to display INFO level messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_on_documents(data_folder):
    vector_store = VectorStoreManager()

    # Initialize counters
    all_files = os.listdir(data_folder)
    total_files = len(all_files)
    files_processed = 0
    files_skipped = 0
    total_docs_loaded = 0
    total_docs_added = 0

    # Iterate over all files in the data folder
    for idx, file_name in enumerate(all_files, start=1):
        file_path = os.path.join(data_folder, file_name)
        logger.info(f"Processing file {idx}/{total_files}: {file_name}")

        # Check if the file is a PDF
        if file_path.lower().endswith('.pdf'):
            files_processed += 1
            try:
                # Load documents
                documents = load_document(file_path)
                num_docs = len(documents)
                total_docs_loaded += num_docs
                logger.info(f"Loaded {num_docs} documents from {file_path}.")

                # Check if there are any documents
                if documents:
                    # Add all documents to the vector store
                    vector_store.add_documents(documents)
                    total_docs_added += len(documents)
                else:
                    logger.warning(f"No documents extracted from {file_path}. Skipping.")

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}", exc_info=True)
        else:
            logger.info(f"Skipping non-PDF file: {file_name}")
            files_skipped += 1

    # Summary of training
    logger.info("Training completed.")
    logger.info(f"Total files processed: {files_processed}")
    logger.info(f"Total files skipped: {files_skipped}")
    logger.info(f"Total documents loaded: {total_docs_loaded}")
    logger.info(f"Total documents added to vector store: {total_docs_added}")

# Add a main guard to execute the training when the script is run directly
if __name__ == "__main__":
    # Specify the path to your data folder containing PDF files
    data_folder_path = "./data" 
    
    # Check if the provided data folder exists
    if not os.path.isdir(data_folder_path):
        logger.error(f"The specified data folder does not exist: {data_folder_path}")
    else:
        train_on_documents(data_folder_path)

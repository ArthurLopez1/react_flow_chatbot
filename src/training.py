import logging
import os  
from .file_handler import load_document, split_document
from .vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)

def train_on_documents(data_folder):
    vector_store = VectorStoreManager()

    # Iterate over all files in the data folder
    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)

        # Check if the file is a PDF
        if file_path.lower().endswith('.pdf'):
            logger.info(f"Processing document: {file_path}")

            try:
                # Load documents
                documents = load_document(file_path)
                logger.info(f"Loaded {len(documents)} documents from {file_path}.")

                # Check if there are any documents
                if documents:
                    # Add all documents to the vector store
                    vector_store.add_documents(documents)
                else:
                    logger.warning(f"No documents extracted from {file_path}. Skipping.")

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}", exc_info=True)

    logger.info("Training completed for all documents in the data folder.")

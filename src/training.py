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

                # Split each document into chunks
                document_chunks = []
                for document in documents:
                    chunks = split_document(document)
                    document_chunks.extend(chunks)
                    logger.info(f"Split document into {len(chunks)} chunks.")

                # Check if there are any document chunks
                if document_chunks:
                    # Embed and add all document chunks at once to the vector store
                    vector_store.add_documents(document_chunks)  # Updated function call
                else:
                    logger.warning(f"No document chunks found for {file_path}. Skipping.")

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}", exc_info=True)

    logger.info("Training completed for all documents in the data folder.")
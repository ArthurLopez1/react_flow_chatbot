import logging
import os  
from src.file_handler import load_documents
from src.vectorstore import VectorStoreManager
from langchain.schema import Document

# Configure logging to display INFO level messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_on_documents(data_folder):
    vector_store = VectorStoreManager()

    # Load documents
    documents = load_documents()
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
    # Load and process documents
    documents = load_documents()
    
    # Initialize the vector store manager
    vector_store = VectorStoreManager()
    
    # Add documents to the vector store
    vector_store.add_documents(documents)
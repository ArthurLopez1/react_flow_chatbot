import logging
from pathlib import Path
import sys
from settings import Config
from src.vectorstore import VectorStoreManager
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreSingleton:
    """
    Singleton class for VectorStoreManager.
    Ensures only one instance exists throughout the application.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreSingleton, cls).__new__(cls)
            vector_store_path = Path("vectorstore/vector_store.index")
            doc_mapping_path = Path("vectorstore/doc_mapping.pkl")
            cls._instance.vector_store_manager = VectorStoreManager(vector_store_path, doc_mapping_path)
            cls._instance.vector_store_manager.initialize()
        return cls._instance

    def __getattr__(self, name):
        return getattr(self.vector_store_manager, name)

# Singleton instance
vector_store_instance = VectorStoreSingleton()
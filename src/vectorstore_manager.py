import logging
from pathlib import Path
import sys
from settings import Config
from .vectorstore import VectorStoreManager
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
            config = Config()
            cls._instance = super(VectorStoreSingleton, cls).__new__(cls)
            cls._instance.vector_store_path = Path(config.VECTOR_STORE_PATH) / "vectorstore.index"
            cls._instance.doc_mapping_path = Path(config.VECTOR_STORE_PATH) / "doc_mapping.pkl"
            cls._instance.index = None
            cls._instance.doc_mapping = {}
            cls._instance.logger = logging.getLogger(__name__)
            logging.basicConfig(level=logging.INFO)
        return cls._instance

# singleton instance
vector_store_instance = VectorStoreSingleton()

import logging
from pathlib import Path
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
            config = Config()
            try:
                cls._instance = VectorStoreManager(
                    vector_store_path=Path(config.VECTOR_STORE_PATH) / "vectorstore.index",
                    doc_mapping_path=Path(config.VECTOR_STORE_PATH) / "doc_mapping.pkl",
                    dimension=384  # Ensure this matches your embedding dimension
                )
                cls._instance.initialize()
                logger.info("VectorStoreManager singleton instance created and initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize VectorStoreManager singleton: {e}")
                raise
        return cls._instance

# singleton instance
vector_store_instance = VectorStoreSingleton()
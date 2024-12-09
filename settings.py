import os
import logging
import streamlit as st

class Config:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading environment variables.")
        
        # Load Streamlit secrets
        secrets = st.secrets
        
        # Load environment variables
        self.TAVILY_API_KEY = secrets.get("TAVILY_API_KEY")
        self.LANGCHAIN_API_KEY = secrets.get("LANGCHAIN_API_KEY")
        
        # Azure configurations
        self.AZURE_STORAGE_ACCOUNT = secrets["azure"].get("account_name")
        self.AZURE_CONTAINER_NAME = secrets["azure"].get("container_name")
        self.AZURE_SAS_TOKEN = secrets["azure"].get("sas_token")
        
        # Set default paths
        self.DATA_PATH = os.path.join("..", "data")
        self.MODEL_PATH = os.getenv("MODEL_PATH", "path/to/model")
        self.VECTOR_STORE_PATH = "vectorstore"  # Ensure this attribute is defined
        
        # Set other configurations
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
        self.SEARCH_RESULTS = int(os.getenv("SEARCH_RESULTS", 3))
        
        # Set environment variables
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "local_rag_prototype"
        
        # Ensure environment variables are set for local model usage
        self.LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.2:3b-instruct-fp16")
        self.TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0))
        self.FORMAT = os.getenv("LLM_FORMAT", None)
        
        # Check for missing environment variables
        self._check_env_vars()

    def _check_env_vars(self):
        """Check for required environment variables and raise an error if any are missing."""
        required_vars = ["TAVILY_API_KEY", "LANGCHAIN_API_KEY", "AZURE_SAS_TOKEN", "AZURE_STORAGE_ACCOUNT", "AZURE_CONTAINER_NAME"]
        for var in required_vars:
            if getattr(self, var) is None:
                self.logger.error(f"Missing required environment variable: {var}")
                raise EnvironmentError(f"Missing required environment variable: {var}")

# Example usage
if __name__ == "__main__":
    config = Config()
    print(f"TAVILY_API_KEY: {config.TAVILY_API_KEY}")
    print(f"LANGCHAIN_API_KEY: {config.LANGCHAIN_API_KEY}")
    print(f"AZURE_SAS_TOKEN: {config.AZURE_SAS_TOKEN}")
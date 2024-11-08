import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv()
        
        # Load environment variables
        self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
        self.LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
        
        # Set default paths
        self.DATA_PATH = os.path.join("..", "data")
        self.MODEL_PATH = os.getenv("MODEL_PATH", "path/to/model")
        self.VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "path/to/vectorstore")
        
        # Set other configurations
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
        self.SEARCH_RESULTS = int(os.getenv("SEARCH_RESULTS", 3))
        
        # Set environment variables
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "local_rag_prototype"
        
        # Check for missing environment variables
        self._check_env_vars()

    def _check_env_vars(self):
        """Check for required environment variables and raise an error if any are missing."""
        required_vars = ["TAVILY_API_KEY", "LANGCHAIN_API_KEY"]
        for var in required_vars:
            if getattr(self, var) is None:
                raise EnvironmentError(f"Missing required environment variable: {var}")

# Example usage
if __name__ == "__main__":
    config = Config()
    print(f"TAVILY_API_KEY: {config.TAVILY_API_KEY}")
    print(f"LANGCHAIN_API_KEY: {config.LANGCHAIN_API_KEY}")
import os
from langchain_ollama import ChatOllama
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model configuration from environment variables or use default values
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.2:3b-instruct-fp16")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0))
FORMAT = os.getenv("LLM_FORMAT", None)

class LLM:
    _instance = None
    _json_instance = None

    def __new__(cls, format=None):
        if format == "json":
            if cls._json_instance is None:
                cls._json_instance = super(LLM, cls).__new__(cls)
                cls._json_instance._initialize(format)
            return cls._json_instance
        else:
            if cls._instance is None:
                cls._instance = super(LLM, cls).__new__(cls)
                cls._instance._initialize(format)
            return cls._instance

    def _initialize(self, format):
        try:
            if format:
                self.llm = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE, format=format)
            else:
                self.llm = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE)
            logger.info(f"Initialized LLM with model: {MODEL_NAME}, temperature: {TEMPERATURE}, format: {format}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def invoke(self, messages):
        return self.llm.invoke(messages)
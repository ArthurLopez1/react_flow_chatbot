import os
from dotenv import load_dotenv  # Add this import
from langchain_community.chat_models import ChatOllama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  

MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.2:1b-instruct-fp16")  
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0))
FORMAT = os.getenv("LLM_FORMAT", None)
LLM_HOST = os.getenv("LLM_HOST", "localhost")
LLM_PORT = int(os.getenv("LLM_PORT", 11434))
BASE_URL = f"http://{LLM_HOST}:{LLM_PORT}"

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
                self.llm = ChatOllama(
                    model=MODEL_NAME,
                    temperature=TEMPERATURE,
                    format=format,
                    base_url=BASE_URL
                )
            else:
                self.llm = ChatOllama(
                    model=MODEL_NAME,
                    temperature=TEMPERATURE,
                    base_url=BASE_URL
                )
            logger.info(
                f"Initialized LLM with model: {MODEL_NAME}, temperature: {TEMPERATURE}, format: {format}, base_url: {BASE_URL}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
            raise

    def invoke(self, messages):
        try:
            response = self.llm.invoke(messages)
            return response
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}", exc_info=True)
            raise

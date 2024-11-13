import os
import logging
import faulthandler  # New import
from src.llm_models import LLM
from src.vectorstore import VectorStoreManager
from src.training import train_on_document
from settings import Config
from src.react_workflow import run_simple_workflow

# Enable faulthandler at the very beginning
faulthandler.enable()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Force retraining by deleting vector store files
        pdf_path = "data/ersattningsmodell_vaders_2019.pdf"
        logger.info(f"Training on document: {pdf_path}")
        train_on_document(pdf_path)
        logger.info("Training completed successfully.")

        # Initialize the vector store for querying
        logger.info("Initializing vector store.")
        vector_store = VectorStoreManager()
        logger.info("Vector store initialized.")
        
        # Initialize models
        logger.info("Initializing LLM models.")
        llm = LLM()
        llm_json_mode = LLM(format="json")
        logger.info("LLM models initialized.")

        # Define initial state and configuration
        state = {
            "question": "Vilka två system används för att samla in väderdata i VädErs-modellen?",
            "config": {
                "top_k": 10  # Set top_k to match the value in vectorstore.py
            },
            "max_retries": 2
        }
        config = Config()  # Ensure config is initialized correctly
        logger.info("Configuration initialized.")

        # Run the simple workflow with intent detection and question splitting
        logger.info("Running simple workflow.")
        events = run_simple_workflow(state)
        logger.info("Workflow completed.")

        # Output the final generated answer
        final_state = events[-1] if events else {}
        generated_answer = final_state.get("generation", "No answer generated.")
        print("Final Generated Answer:")
        print(generated_answer)
        logger.info(f"Final state: {final_state}")

    except Exception as e:
        logger.error(f"An error occurred during main execution: {e}", exc_info=True)
    finally:
        # Ensure any necessary cleanup here
        logger.info("Execution completed.")

if __name__ == "__main__":
    main()
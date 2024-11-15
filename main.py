import os
import logging
import faulthandler
from src.llm_models import LLM
from src.vectorstore import VectorStoreManager
from src.training import train_on_documents  
from settings import Config
from src.react_workflow import run_simple_workflow


faulthandler.enable()

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Force retraining by deleting vector store files
        data_folder = "data"  # Define the data folder
        logger.info(f"Training on documents in folder: {data_folder}")
        train_on_documents(data_folder)  # Updated function call
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
                "top_k": 5  # Set top_k to match the value in vectorstore.py
            },
        }
        config = Config()  
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
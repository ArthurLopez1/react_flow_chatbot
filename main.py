import os
import logging
from venv import logger
from src.llm_models import LLM
from src.vectorstore import VectorStoreManager
from src.components import run_workflow, initialize_graph
from src.training import train_on_document
from settings import Config

def main():
    # Train with a sample document
    pdf_path = "data/ersattningsmodell_vaders_2019.pdf"
    train_on_document(pdf_path)

    # Initialize the vector store for querying
    vector_store = VectorStoreManager()
    
    # Initialize models
    llm = LLM()
    llm_json_mode = LLM(format="json")

    # Define initial state and configuration
    state = {
        "question": "Hur ofta analyseras väderdata i VädErs-modellen?",
        "max_retries": 2
    }
    config = Config()  # Ensure config is initialized correctly

    # Run the workflow
    events = run_workflow(state, config)

    # Output the final generated answer
    final_state = events[-1] if events else {}
    generated_answer = final_state.get("generation", "No answer generated.")
    print("Final Generated Answer:")
    print(generated_answer)
    logger.info(f"Final state: {final_state}")

if __name__ == "__main__":
    main()
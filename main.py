import os
import logging
from src.llm_models import LLM
from src.vectorstore import VectorStoreManager
from src.training import train_on_document
from settings import Config
from src.react_workflow import run_simple_workflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        "question": "Hi!",
        "max_retries": 2
    }
    config = Config()  # Ensure config is initialized correctly

    # Run the simple workflow with intent detection and question splitting
    events = run_simple_workflow(state)

    # Output the final generated answer
    final_state = events[-1] if events else {}
    generated_answer = final_state.get("generation", "No answer generated.")
    print("Final Generated Answer:")
    print(generated_answer)
    logger.info(f"Final state: {final_state}")

if __name__ == "__main__":
    main()
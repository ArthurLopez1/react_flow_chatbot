import os
from pathlib import Path
import logging  # Updated import for logger
import streamlit as st
from src.vectorstore import VectorStoreManager
from src.training import train_on_document
from settings import Config
from src.llm_models import LLM
from src.react_workflow import run_simple_workflow  
# Initialize the configuration
config = Config()

# Initialize the vector store manager
vec = VectorStoreManager()

# Initialize models
llm = LLM()
llm_json_mode = LLM(format="json")

# Load custom CSS
def read_css():
    css_path = Path(__file__).parent / "frontend" / "style.css"
    with open(css_path) as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

read_css()  # Call CSS function at the start of the script

# Streamlit app title
st.title("Chat with Klimator")  # Main title

# Sidebar for training the model
st.sidebar.title("Training")
pdf_path = st.sidebar.text_input("PDF Path", "data/ersattningsmodell_vaders_2019.pdf")

if st.sidebar.button("Train Model"):
    train_on_document(pdf_path)
    st.sidebar.success("Model trained successfully!")

# Main interface for asking questions
st.header("Ask a question about the document:")
st.markdown('<div data-testid="question-label">Enter your question here:</div>', unsafe_allow_html=True)
question = st.text_input("Question", key="question_input", label_visibility="collapsed")

if st.button("Get Answer"):
    # Define initial state and configuration
    state = {
        "question": question,
        "messages": [  # Add messages to mimic chat history
            {"content": question}  # Example message structure
        ]
    }

    # Run the simple workflow
    events = run_simple_workflow(state)
    final_state = events[-1] if events else {}
    generated_answer = final_state.get("generation", "No answer generated.")
    
    if "Error generating answer." in generated_answer:
        st.error(generated_answer)
    else:
        st.markdown(f'<div class="generated-answer">{generated_answer}</div>', unsafe_allow_html=True)
    
    logging.info(f"Final state: {final_state}")  # Ensure correct logger is used
    print(f"Final state: {final_state}")  # Ensure final state is printed to console

if __name__ == "__main__":
    # Define initial state
    state = {
        "question": "Hur ofta analyseras väderdata i VädErs-modellen?"
    }

    # Run the simple workflow
    events = run_simple_workflow(state)

    # Output the final generated answer
    final_state = events[-1] if events else {}
    generated_answer = final_state.get("generation", "No answer generated.")
    print("Final Generated Answer:")
    print(generated_answer)
    logging.info(f"Final state: {final_state}")

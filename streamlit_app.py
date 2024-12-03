import os
from pathlib import Path
import logging
import streamlit as st
from src.vectorstore import VectorStoreManager
from src.training import train_on_documents
from settings import Config
from src.llm_models import LLM
from src.react_workflow import run_simple_workflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the configuration
config = Config()

# Initialize the vector store manager
vec = VectorStoreManager()

# Initialize models
llm = LLM()
llm_json_mode = LLM(format="json")

def load_css():
    css_path = Path(__file__).parent / "frontend" / "style.css"
    if (css_path.exists()):
        with open(css_path) as css_file:
            st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)
    else:
        logger.warning(f"CSS file not found at {css_path}.")

load_css()

# Streamlit app title
st.title("Chat with Klimator")  # Main title

# Sidebar for training the model
st.sidebar.title("Training")
pdf_path = st.sidebar.text_input("PDF Path", "data/ersattningsmodell_vaders_2019.pdf")

if st.sidebar.button("Train Model"):
    data_folder = os.path.dirname(pdf_path)  # Derive data_folder from pdf_path
    train_on_documents(data_folder)  # Updated function call
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


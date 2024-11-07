from pathlib import Path
from venv import logger
import streamlit as st
from src.components import run_workflow
from src.vectorstore import VectorStoreManager
from src.training import train_on_document
from settings import Config
from src.llm_models import LLM
from react_workflow import run_simple_workflow  

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
st.markdown('<div data-testid="max-retries-label">Max Retries</div>', unsafe_allow_html=True)
max_retries = st.number_input("Max Retries", min_value=1, max_value=10, value=3, key="max_retries_input", label_visibility="collapsed")

if st.button("Get Answer"):
    # Define initial state and configuration
    state = {
        "question": question,
        "max_retries": max_retries
    }


    # simple workflow
    events = run_simple_workflow(state)
    final_state = events[-1] if events else {}
    generated_answer = final_state.get("generation", "No answer generated.")
    st.markdown(f'<div class="generated-answer">{generated_answer}</div>', unsafe_allow_html=True)
    logger.info(f"Final state: {final_state}")
    print(f"Final state: {final_state}")  # Ensure final state is printed to console

    # main workflow
    # events = run_workflow(state, config)
    # final_state = events[-1] if events else {}
    # generated_answer = final_state.get("generation", "No answer generated.")
    # st.write("Final Generated Answer:")
    # st.write(generated_answer)
    # logger.info(f"Final state: {final_state}")
    # print(f"Final state: {final_state}")  # Ensure final state is printed to console

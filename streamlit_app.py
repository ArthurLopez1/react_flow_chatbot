import streamlit as st
from pathlib import Path
import logging
import uuid
import os
from settings import Config
from src.vectorstore import VectorStoreManager
from src.training import train_on_documents
from src.llm_models import LLM
from src.react_workflow import run_simple_workflow
import base64

# Custom CSS to style the send button and sidebar
st.markdown(
    """
    <style>
    /* Remove default Streamlit button styling */
    .css-1cpxqw2.edgvbvh3 {
        padding: 0;
        background-color: transparent;
        border: none;
        cursor: pointer;
    }
    /* Style the emoji button on hover */
    .css-1cpxqw2.edgvbvh3:hover {
        background-color: #f0f0f0;
        border-radius: 5px;
    }

    /* Style for the Trained on section in sidebar */
    .trained-on-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .trained-on-item {
        margin-bottom: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the configuration
config = Config()

# Initialize the vector store manager
vec = VectorStoreManager()

# Initialize models
llm = LLM()

def get_image_as_base64(image_path):
    """
    Reads an image file and returns its Base64 encoded string.
    """
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return ""

def load_css():
    css_path = Path(__file__).parent / "frontend" / "style.css"
    if css_path.exists():
        with open(css_path) as css_file:
            st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)
    else:
        logger.warning(f"CSS file not found at {css_path}.")

load_css()

st.markdown("<h3>chat with</h3>", unsafe_allow_html=True)

logo_path = Path(__file__).parent / "assets" / "klimator.png"

if logo_path.exists():
    encoded_logo = get_image_as_base64(logo_path)
    logo_html = f'<img src="data:image/png;base64,{encoded_logo}" width="50" style="margin-right: 5px;" alt="Klimator Logo"/>'
    title_html = f'<h1 style="margin: 0;">klimator</h1>'
    st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            {logo_html}
            {title_html}
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    logger.error(f"Logo file not found at {logo_path}.")
    st.markdown("<h1>klimator</h1>", unsafe_allow_html=True)

# Sidebar: Display "Trained on:" with list of PDF files as buttons
st.sidebar.header("Trained on:")
data_folder = Path("data")  # Ensure this path is correct relative to your app

def get_pdf_list(data_folder):
    """
    Retrieves a list of PDF files from the specified data folder.
    """
    if data_folder.exists() and data_folder.is_dir():
        return list(data_folder.glob("*.pdf"))
    else:
        logger.error(f"The data folder '{data_folder}' does not exist or is not a directory.")
        return []

pdf_files = get_pdf_list(data_folder)

if pdf_files:
    for pdf_file in pdf_files:
        with st.sidebar.container():
            try:
                # Read and encode the PDF file
                with open(pdf_file, "rb") as f:
                    data = f.read()
                encoded_pdf = base64.b64encode(data).decode()

                # Create a download link styled as a button
                href = f"data:application/pdf;base64,{encoded_pdf}"
                download_link = f'<a href="{href}" download="{pdf_file.name}" class="file-button">{pdf_file.name}</a>'
                st.markdown(download_link, unsafe_allow_html=True)
            except Exception as e:
                logger.error(f"Error reading PDF file {pdf_file.name}: {e}")
                st.sidebar.error(f"Unable to read {pdf_file.name}.")
else:
    st.sidebar.info("No PDF files found in the data folder.")


# Create columns for input and send button
col1, col2 = st.columns([5, 1])  # Adjust the ratio as needed

with col1:
    question = st.text_input(
        "Question",
        key="question_input",
        label_visibility="collapsed",
        placeholder="Type your question here..."
    )

with col2:
    send_button = st.button("⌲")  # Emoji as send button

# Handle the send button click
if send_button:
    if question.strip():
        with st.spinner('Generating answer...'):
            try:
                # Pass a dictionary with the question
                events = run_simple_workflow({"question": question})
                
                # Extract the final state from events
                final_state = events[-1] if events else {}
                generated_answer = final_state.get("generation", "No answer generated.")
                
                if "No documents found" in generated_answer or "Error generating answer" in generated_answer:
                    st.error(generated_answer)
                else:
                    st.markdown(f'<div class="generated-answer">{generated_answer}</div>', unsafe_allow_html=True)
                
                logger.info(f"Generated answer: {generated_answer}")
                print(f"Generated answer: {generated_answer}")  # Ensure final state is printed to console
            except Exception as e:
                logger.error(f"Error during workflow execution: {e}")
                st.error("Sorry, I encountered an error processing your request. Please try again.")
    else:
        st.warning("Please enter a question before sending.")


#[theme]
#base="light"
#primaryColor="#a8a69a"

from .file_handler import load_document, split_document
from .vectorstore import VectorStoreManager

def train_on_document(file_path):
    # Load documents
    documents = load_document(file_path)
    
    # Split each document into chunks
    document_chunks = []
    for document in documents:
        document_chunks.extend(split_document(document))
    
    # Extract the text content from each chunk
    chunk_texts = [chunk.page_content for chunk in document_chunks]

    # Initialize the vector store manager
    vector_store = VectorStoreManager()

    # Embed and add all document chunks at once to the vector store
    vector_store.add_documents(chunk_texts)

    print(f"Training completed for document: {file_path}")

import os
import faiss
import numpy as np
import pickle
from typing import List, Tuple
from sentence_transformers import SentenceTransformer  # Updated import
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document  # Add this import


class VectorStoreManager:
    def __init__(self, vector_store_path="vector_store.index", doc_mapping_path="doc_mapping.pkl"):
        self.vector_store_path = vector_store_path
        self.doc_mapping_path = doc_mapping_path
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Manually set the dimension for the 'sentence-transformers/all-MiniLM-L6-v2' model
        self.dimension = 384  # Adjust this value based on your specific model

        # Initialize or load the FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self._load_index()

        # Document mapping for retrieval
        self.doc_mapping = self._load_doc_mapping()

    def embed_text(self, text: str):
        """
        Generate an embedding vector locally using HuggingFaceEmbeddings.
        """
        embedding = self.embedding_model.embed_documents([text])[0]
        return np.array(embedding).astype("float32")

    def add_documents(self, documents: List[str]):
        """
        Embed and add multiple documents to the FAISS index.
        """
        embeddings = [self.embed_text(doc) for doc in documents]
        embeddings_array = np.vstack(embeddings)  # Ensure embeddings_array is a 2D array
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Map indices to documents
        start_idx = len(self.doc_mapping)
        for i, doc in enumerate(documents):
            self.doc_mapping[start_idx + i] = doc
        
        print(f"Added {len(documents)} documents to the vector store.")
        self._save_index()
        self._save_doc_mapping()

    def _rerank_documents(self, documents: List[str], query: str) -> List[Tuple[str, float]]:
        """
        Rerank documents based on cosine similarity with the query.
        """
        try:
            print(f"Reranking documents for query: {query}")  # Debugging print
            # Encode the query and documents
            query_embedding = self.embed_text(query)
            doc_embeddings = np.array([self.embed_text(doc) for doc in documents])

            # Compute cosine similarities
            cosine_scores = np.dot(doc_embeddings, query_embedding.T).flatten()

            # Pair each document with its score
            reranked_results = list(zip(documents, cosine_scores))

            # Sort documents by relevance score in descending order
            reranked_results.sort(key=lambda x: x[1], reverse=True)

            return reranked_results
        except Exception as e:
            print(f"Error during reranking documents: {e}")
            raise  # Re-raise exception after logging

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve the top_k most similar documents for a given query.
        """
        query_embedding = self.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1)  # Ensure query_embedding is a 2D array
        distances, indices = self.index.search(query_embedding, top_k)

        # Fetch documents from mapping and create Document objects
        results = [Document(page_content=self.doc_mapping[idx], metadata={"relevance_score": float(distances[0][i])}) for i, idx in enumerate(indices[0]) if idx in self.doc_mapping]
        print(f"Retrieved {len(results)} documents for query: '{query}'")
        return results

    def _save_index(self):
        """
        Persist the FAISS index to disk.
        """
        try:
            faiss.write_index(self.index, self.vector_store_path)
            print(f"FAISS index saved to {self.vector_store_path}.")
        except Exception as e:
            print(f"Error saving FAISS index to disk: {e}")
            raise

    def _save_index(self):
        """
        Persist the FAISS index to disk.
        """
        faiss.write_index(self.index, self.vector_store_path)
        print("Vector store saved to disk.")

    def _load_index(self):
        """
        Load the FAISS index from disk if available.
        """
        if os.path.exists(self.vector_store_path):
            self.index = faiss.read_index(self.vector_store_path)
            print("Vector store loaded from disk.")
        else:
            print("No existing vector store found. Creating a new one.")

    def _save_doc_mapping(self):
        """
        Save the document mapping to disk for retrieval.
        """
        with open(self.doc_mapping_path, "wb") as f:
            pickle.dump(self.doc_mapping, f)
        print("Document mapping saved to disk.")

    def _load_doc_mapping(self):
        """
        Load the document mapping from a file if it exists.
        """
        if os.path.exists(self.doc_mapping_path):
            with open(self.doc_mapping_path, 'rb') as f:
                return pickle.load(f)
        return {}
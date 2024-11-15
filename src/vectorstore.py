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

        self.nlist = 10  # Adjusted number of clusters based on dataset size
        self.nprobe = 10  # Number of clusters to search

        # Initialize or load the FAISS index
        self._load_or_create_index()

        # Document mapping for retrieval
        self.doc_mapping = self._load_doc_mapping()

    def embed_text(self, text: str):
        """
        Generate an embedding vector locally using HuggingFaceEmbeddings.
        """
        embedding = self.embedding_model.embed_documents([text])[0]
        return np.array(embedding).astype("float32")

    def add_documents(self, documents: List[Document]):
        """
        Embed and add multiple documents to the FAISS index.
        """
        if not documents:
            print("No documents to add. Skipping.")
            return
        
        embeddings = [self.embed_text(doc.page_content) for doc in documents]
        if not embeddings:
            print("No embeddings generated. Skipping.")
            return
        
        try:
            embeddings_array = np.vstack(embeddings)  # Ensure embeddings_array is a 2D array
        except ValueError as e:
            print(f"Error stacking embeddings: {e}")
            return
        
        if embeddings_array.size == 0:
            print("Embeddings array is empty. Skipping addition to index.")
            return

        # Check if there are enough training points
        if embeddings_array.shape[0] < self.nlist * 39:  # Ensure at least 39 points per cluster
            print(f"Not enough training points. Need at least {self.nlist * 39} points.")
            return

        # Train the index if it's not trained
        if not self.index.is_trained:
            print("Training FAISS index...")
            self.index.train(embeddings_array)
            print("FAISS index trained.")

        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Map indices to documents
        start_idx = len(self.doc_mapping)
        for i, doc in enumerate(documents):
            self.doc_mapping[start_idx + i] = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
        
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
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k * 2)  # Retrieve more candidates
        
        # Fetch documents from mapping
        candidates = []
        for idx in indices[0]:
            if idx in self.doc_mapping:
                doc_info = self.doc_mapping[idx]
                candidates.append(doc_info["page_content"])
        
        # Rerank the candidates using cosine similarity
        reranked_results = self._rerank_documents(candidates, query)
        
        # Select top_k documents after reranking
        top_docs = reranked_results[:top_k]
        
        # Convert to Document objects
        results = []
        for doc_text, score in top_docs:
            # Find the corresponding metadata
            for idx, doc_info in self.doc_mapping.items():
                if doc_info["page_content"] == doc_text:
                    results.append(Document(
                        page_content=doc_text,
                        metadata=doc_info["metadata"]
                    ))
                    break
        
        print(f"Retrieved {len(results)} documents for query: '{query}' after reranking.")
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

    def _create_index(self):
        """
        Create a FAISS IndexIVFFlat index for large-scale vector storage.
        """
        quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_L2)
        self.index.nprobe = self.nprobe
        print("Created a new IndexIVFFlat index.")

    def _load_or_create_index(self):
        """
        Load the FAISS index from disk if available, otherwise create a new one.
        """
        if os.path.exists(self.vector_store_path):
            try:
                self.index = faiss.read_index(self.vector_store_path)
                print("Vector store loaded from disk.")
            except Exception as e:
                print(f"Error loading index: {e}. Creating a new index.")
                self._create_index()
        else:
            print("No existing vector store found. Creating a new one.")
            self._create_index()

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
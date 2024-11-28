import os
import faiss
import numpy as np
import pickle
from typing import List, Tuple
from sentence_transformers import SentenceTransformer  
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

        # Use IndexFlatL2 for small datasets
        self.index = faiss.IndexFlatL2(self.dimension)
        # Remove or comment out nlist and nprobe settings
        # self.nlist = 10
        # self.nprobe = 10

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

        embeddings_array = np.vstack(embeddings).astype("float32")

        # Add embeddings to the index
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

    def _rerank_documents(self, candidates, query):
        """
        Rerank documents based on cosine similarity with the query.
        Each candidate is a tuple of (doc_text, metadata).
        """
        try:
            # Encode the query and documents
            query_embedding = self.embed_text(query)
            doc_embeddings = np.array([self.embed_text(doc_text) for doc_text, _ in candidates])

            if doc_embeddings.size == 0:
                print("Document embeddings are empty. Cannot perform reranking.")
                return []

            # Compute cosine similarities
            cosine_scores = np.dot(doc_embeddings, query_embedding.T).flatten()

            # Pair each document with its score and metadata
            reranked_results = [
                (candidates[i][0], candidates[i][1], cosine_scores[i])
                for i in range(len(candidates))
            ]

            # Sort documents by relevance score in descending order
            reranked_results.sort(key=lambda x: x[2], reverse=True)

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
                candidates.append((doc_info["page_content"], doc_info["metadata"]))

        # Rerank the candidates using cosine similarity
        reranked_results = self._rerank_documents(candidates, query)

        # Select top_k documents after reranking
        top_docs = reranked_results[:top_k]

        # Convert to Document objects
        results = []
        for doc_text, metadata, score in top_docs:
            results.append(Document(
                page_content=doc_text,
                metadata=metadata
            ))

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
        # No longer needed for IndexFlatL2
        pass

    def _load_or_create_index(self):
        if os.path.exists(self.vector_store_path):
            try:
                self.index = faiss.read_index(self.vector_store_path)
                print("Vector store loaded from disk.")
            except Exception as e:
                print(f"Error loading index: {e}. Creating a new index.")
                self.index = faiss.IndexFlatL2(self.dimension)
                print("Created a new IndexFlatL2 index.")
        else:
            print("No existing vector store found. Creating a new one.")
            self.index = faiss.IndexFlatL2(self.dimension)
            print("Created a new IndexFlatL2 index.")

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
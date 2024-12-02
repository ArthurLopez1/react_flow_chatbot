import logging
from typing import List, Dict, Any
from typing_extensions import TypedDict
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from src.vectorstore import VectorStoreManager
from src.llm_models import LLM
from langchain.schema import Document
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from sklearn.metrics.pairwise import cosine_similarity  
from langchain_core.runnables import RunnableSequence

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the LLM once
llm = LLM()

# Initialize the vector store manager
vector_store = VectorStoreManager()

# Test retrieval
query = "Test query relevant to your documents"
retrieved_docs = vector_store.retrieve_documents(query, top_k=1)

for doc in retrieved_docs:
    print(f"Retrieved Document Metadata: {doc.metadata}")
    print(f"Content: {doc.page_content}")

class SimpleGraphState(TypedDict):
    """Graph state containing information for each graph node."""
    question: str
    answers: List[str]
    generation: str
    documents: List[Document]  

def retrieve_relevant_documents(state: Dict[str, Any]):
    query = state.get("question", "")
    
    # If 'question' is a dict, extract the actual query string
    if isinstance(query, dict):
        query = query.get("text", "")
    
    if not isinstance(query, str):
        logger.error(f"Expected query to be a string, but got {type(query)}. Content: {query}")
        state["documents"] = []
        return state
    
    documents = vector_store.retrieve_documents(query, top_k=5)
    state["documents"] = documents
    return state

def retrieve_documents(state: Dict[str, Any]):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    documents = retrieve_relevant_documents(state)  
    state["documents"] = documents      
    return state

def transform_documents(state: Dict[str, Any]):
    documents = state.get("documents", [])
    if not documents:
        state["documents"] = []
        return state

    # Use the global vector_store instance to embed text
    try:
        embeddings = [vector_store.embed_text(doc.page_content) for doc in documents]
        logger.info("Embeddings generated for documents.")
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        state["documents"] = []
        return state

    # Proceed with filtering (optional)
    # Example: Remove duplicates based on content similarity
    unique_documents = []
    seen = set()
    for doc, embedding in zip(documents, embeddings):
        content_hash = hash(doc.page_content)
        if content_hash not in seen:
            seen.add(content_hash)
            unique_documents.append(doc)

    logger.info(f"Filtered documents count: {len(unique_documents)}")

    state["documents"] = unique_documents
    return state

rag_prompt = """You are tasked with answering a question based on the provided sources. Answer in the same language as the question.

You will be given two inputs:

<sources>
{SOURCES}
</sources>

<question>
{QUESTION}
</question>

Instructions:

1. Read and analyze all the provided sources carefully.

2. Use the information from the sources to answer the question comprehensively. 

3. As you write your answer, use superscript numbers for citations that correspond to the numbered sources below.

4. Format your answer in markdown, including a "Referencer" section at the end.

5. In the "Referencer" section, list each reference corresponding to the superscript numbers in your answer.

6. Use the following format for references:

   - In-text citation: Use superscript numbers like `¹`, `²`, `³`, etc.

   - Reference entry: Include the source document title and page number, e.g., `¹ [Ersättningsmodell VädErs 2019, Sidan 5]`.

7. If the answer is not present in the sources, state that you don't have enough information to answer.

8. Make sure your answer is clear, concise, and relevant to the question.

Always answer in the same language as the question.
"""

def _rag_answer_chain():
    system_message = SystemMessagePromptTemplate.from_template(
        "You are a knowledgeable assistant fluent in Swedish and English. Answer the question based on the provided sources."
    )
    human_message = HumanMessagePromptTemplate.from_template(rag_prompt)
    prompt_template = ChatPromptTemplate.from_messages([system_message, human_message])
    return RunnableSequence(prompt_template, llm.llm)

def rag_answer(state: Dict[str, Any]):
    if not state.get("documents"):
        logger.error('Unable to generate answer, no documents found')
        state["generation"] = "No documents found to generate an answer."
        return state

    try:
        question = state["question"]

        # Prepare sources with identifiers for citations
        sources = ""
        citation_info = {}
        for i, doc in enumerate(state["documents"], 1):
            content = doc.page_content
            source = doc.metadata.get('source', 'Unknown Document')
            page_number = doc.metadata.get('page_number', 'Unknown Page')
            # Format sources as a numbered list without "Source" prefix
            sources += f"{i}. {source}, Page {page_number}:\n{content}\n\n"
            citation_info[i] = f"{source}, Page {page_number}"

        # Store citations for reference section
        state['citations'] = citation_info

        final_answer_chain = _rag_answer_chain()

        # Invoke the chain with input variables
        inputs = {"SOURCES": sources, "QUESTION": question}
        response = final_answer_chain.invoke(inputs)
        # Extract the content from the response
        if isinstance(response, dict) and 'text' in response:
            generated_text = response['text']
        elif hasattr(response, 'content'):
            generated_text = response.content
        else:
            generated_text = str(response)

        logger.info(f"LLM response: {generated_text}")
        
        state["generation"] = generated_text

        return state
    except Exception as e:
        logger.error(f"Error in rag_answer: {e}", exc_info=True)
        state["generation"] = "Error generating answer."
        return state

def initialize_simple_graph():
    """Define a simple graph, nodes, and edges."""
    workflow = StateGraph(SimpleGraphState)
    
    # Add nodes
    workflow.add_node("retrieve_documents", retrieve_relevant_documents)
    workflow.add_node("transform_documents", transform_documents)
    workflow.add_node("rag_answer", rag_answer)

    # Build graph
    workflow.set_entry_point("retrieve_documents")
    workflow.add_edge("retrieve_documents", "transform_documents")
    workflow.add_edge("transform_documents", "rag_answer")
    workflow.add_edge("rag_answer", END)

    # Compile the graph
    return workflow.compile()

def run_simple_workflow(state: Dict[str, Any]):
    """Run the simplified workflow."""
    graph = initialize_simple_graph()
    events = []
    logger.info(f"Initial state: {state}")
    for event in graph.stream(state, stream_mode="values"):
        events.append(event)
        logger.info(f"Event: {event}")
    logger.info(f"Events: {events}")
    return events

if __name__ == "__main__":
    # Define initial state with the user's question
    state = {
        "question": "Hur ofta analyseras väderdata i VädErs-modellen?"
    }

    # Run the workflow
    events = run_simple_workflow(state)

    # Output the final generated answer
    final_state = events[-1] if events else {}
    generated_answer = final_state.get("generation", "No answer generated.")
    print("Final Generated Answer:")
    print(generated_answer)
    logger.info(f"Final state: {final_state}")
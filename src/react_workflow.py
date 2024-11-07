import os
import logging
from typing import List, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from .vectorstore import VectorStoreManager
from .llm_models import LLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the LLM once
llm = LLM()

class SimpleGraphState(TypedDict):
    """Graph state containing information for each graph node."""
    question: str
    generation: str
    documents: List[str]

# Nodes
def retrieve_documents(state: Dict[str, Any]):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE DOCUMENTS---")
    question = state["question"]

    # Write retrieved documents to documents key in state
    documents = VectorStoreManager().retrieve_documents(question)
    state["documents"] = documents  # Ensure documents are added to state
    print(f"Retrieved documents: {documents}")
    return state

def generate_answer(state: Dict[str, Any]):
    """
    Generate answer using retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]

    # Format documents for generation
    docs_txt = "\n\n".join([doc[0] for doc in documents])
    prompt = f"Here is the context to use to answer the question:\n\n{docs_txt}\n\nThink carefully about the above context.\n\nNow, review the user question:\n\n{question}\n\nProvide an answer to this question using only the above context. Use three sentences maximum and keep the answer concise.\n\nAnswer:"
    generation = llm.invoke([HumanMessage(content=prompt)])
    state["generation"] = generation.content
    print(f"Generated answer: {generation.content}")
    return state

def initialize_simple_graph():
    """Define a simple graph, nodes, and edges."""
    workflow = StateGraph(SimpleGraphState)

    # Add nodes
    workflow.add_node("retrieve_documents", retrieve_documents)  # retrieve documents
    workflow.add_node("generate_answer", generate_answer)  # generate answer

    # Build graph
    workflow.set_entry_point("retrieve_documents")
    workflow.add_edge("retrieve_documents", "generate_answer")
    workflow.add_edge("generate_answer", END)

    # Compile the graph
    graph = workflow.compile()

    return graph

def run_simple_workflow(state: Dict[str, Any]):
    """Run the simple workflow."""
    graph = initialize_simple_graph()
    events = []
    logger.info(f"Initial state: {state}")
    for event in graph.stream(state, stream_mode="values"):
        events.append(event)
        logger.info(f"Event: {event}")
    logger.info(f"Events: {events}")
    return events

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
    logger.info(f"Final state: {final_state}")

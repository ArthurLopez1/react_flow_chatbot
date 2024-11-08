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
    intent: str
    question_list: List[str]
    generation: str
    documents: List[str]

intent_prompt = """You are an AI assistant tasked with detecting the intent of an incoming message. Your goal is to categorize the message into one of three intents: greeting, specific_question, or metadata_query.
Here are the definitions for each intent:
1. Greeting: A simple salutation or farewell, such as "Hello," "Hi," "Good morning," "Goodbye," or "See you later."
2. Specific_question: A specific inquiry about a subject asked by the user. This could be any question that doesn't fall under the metadata_query category.
3. Metadata_query: Any request involving a count, sort, or retrieval of specific subsets of data. This includes questions about the latest documents, counting items, or sorting results.

<chat_history>
{chat_history}
<chat_history>

<question>
{question}
</question>

Carefully read and analyze the message to determine its intent based on the definitions provided above.
Guidelines for categorization:
- If the message is a simple greeting or farewell, classify it as "greeting"
- If the message asks for information about counts, sorting, or retrieving specific subsets of data, classify it as "metadata_query"
- For questions that require extra info or cannot be answer based on the existing chat history, classify it as "specific_question"
After analyzing the message, provide your reasoning for the classification in <reasoning> tags."""

def detect_intent(state: Dict[str, Any]):
    """
    Detect the intent of the user's input using the provided intent_prompt.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates 'intent' in the state
    """
    print("---DETECT INTENT---")
    question = state["question"]
    chat_history = state.get("chat_history", "")

    # Use LLM to classify intent
    prompt = intent_prompt.format(chat_history=chat_history, question=question)
    intent_response = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
    state["intent"] = intent_response
    print(f"Detected intent: {intent_response}")
    return state

def split_questions(state: Dict[str, Any]):
    """
    Split the user's input into multiple questions if necessary.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates 'question_list' in the state
    """
    print("---SPLIT QUESTIONS---")
    question = state["question"]

    # Use LLM to split questions
    prompt = f"Split the following input into individual questions:\n\n{question}\n\nQuestions:"
    split_response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    questions = split_response.split('\n')
    state["question_list"] = questions
    print(f"Split questions: {questions}")
    return state

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
    workflow.add_node("detect_intent", detect_intent)  # detect intent
    workflow.add_node("split_questions", split_questions)  # split questions
    workflow.add_node("retrieve_documents", retrieve_documents)  # retrieve documents
    workflow.add_node("generate_answer", generate_answer)  # generate answer

    # Build graph
    workflow.set_entry_point("detect_intent")
    workflow.add_edge("detect_intent", "split_questions")
    workflow.add_edge("split_questions", "retrieve_documents")
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

# src/components.py
import os
import json
import logging
from typing import List, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from .testers.file_handler_0 import load_document, split_document
from .vectorstore import VectorStoreManager
from .llm_models import LLM
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from IPython.display import Image, display

# Prompt and instructions setup
router_instructions = """You are an expert at Trafikverket (Swedish Transport Administration) and methereology.

The vectorstore contains documents related to weather data collection, weather situation analyses, compensation calculation, and related fields of study.

Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

Return JSON with a single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

doc_grader_instructions = """You are a grader assessing the relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

Assess whether the document contains at least some information that is relevant to the question.

Return JSON with a single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least one information that is relevant to the question."""

rag_prompt = """You are an assistant for question-answering tasks in Swedish. 

Here is the context to use to answer the question:

{context}

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this question using only the above context. 

Use three sentences maximum and keep the answer concise.

Answer:"""

hallucination_grader_instructions = """You are a teacher grading a quiz in Swedish. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Score:

A score of yes means that the student's answer meets most of the criteria. This is the highest (best) score. 

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

answer_grader_instructions = """You are a teacher grading a quiz in Swedish. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) The STUDENT ANSWER helps to answer the QUESTION

Score:

A score of yes means that the student's answer meets most of the criteria. This is the highest (best) score. 

A score of no means that the student's answer does not meet of the criteria at all. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""

# Initialize the LLM once
llm = LLM()
llm_json_mode = LLM(format="json")

# Web Search Tool
web_search_tool = TavilySearchResults(k=3)

class GraphState(TypedDict):
    """Graph state containing information for each graph node."""
    question: str
    generation: str
    web_search: str
    max_retries: int
    answers: int
    loop_step: int
    documents: List[str]

# nodes
def format_docs(docs):
    """
    Format documents for display by joining them with double newlines.
    
    Parameters:
        docs (list): List of documents, where each document is expected to be 
                     a tuple (doc, score).
    
    Returns:
        str: Formatted string of documents.
    """
    formatted_docs = []
    
    for item in docs:
        # Check if item has at least one element (the document text)
        if isinstance(item, tuple) and len(item) >= 1:
            doc = item[0]  # Get the document content
        else:
            # If the item is not properly formatted, convert it to a string
            doc = str(item)
        formatted_docs.append(doc)
    
    return "\n\n".join(formatted_docs)

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Write retrieved documents to documents key in state
    documents = VectorStoreManager().retrieve_documents(question)
    state["documents"] = documents  # Ensure documents are added to state
    print(f"Retrieved documents: {documents}")
    return state

def generate(state: Dict[str, Any], config: Dict[str, Any]):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    state["generation"] = generation.content
    state["loop_step"] = loop_step + 1
    print(f"Generated answer: {generation.content}")
    return state

def grade_documents(state):
    """Grade the relevance of retrieved documents to the question."""
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    relevant_doc_found = False
    for doc, score in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=doc, question=question
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append((doc, score))
            relevant_doc_found = True
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")

    web_search = "No" if relevant_doc_found else "Yes"
    state["documents"] = filtered_docs
    state["web_search"] = web_search
    print(f"Filtered documents: {filtered_docs}")
    return state

def web_search(state):
    """Web search based on the question."""
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    state["documents"] = documents  # Ensure documents are added to state
    print(f"Web search results: {documents}")
    return state


# edges
def route_question(state: Dict[str, Any], config: Dict[str, Any]):
    """Route the question to the appropriate datasource."""
    print("---ROUTE QUESTION---")
    question = state["question"]
    route_question_prompt_formatted = router_instructions.format(question=question)
    result = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=route_question_prompt_formatted)]
    )
    try:
        response_content = json.loads(result.content)
        print(f"Response content: {response_content}")  # Debugging line
        source = response_content["datasource"]
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Response content: {result.content}")
        raise
    except KeyError as e:
        print(f"KeyError: {e}")
        print(f"Response content: {response_content}")
        raise
    state["datasource"] = source
    return source  # Return the next node to call as a string
    
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state: Dict[str, Any], config: Dict[str, Any]):
    """Grade the generated answer against the retrieved documents and the question."""
    print("---GRADE GENERATION vs DOCUMENTS AND QUESTION---")
    question = state["question"]
    generation = state["generation"]
    documents = state["documents"]
    max_retries = state["max_retries"]
    loop_step = state.get("loop_step", 0)

    # Format the documents for grading
    docs_txt = format_docs(documents)
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=docs_txt, generation=generation
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)["binary_score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        # Test using question and generation from above
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generation=generation
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        elif loop_step <= max_retries:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            return "max retries"
    elif loop_step <= max_retries:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    else:
        print("---DECISION: MAX RETRIES REACHED---")
        return "max retries"

def initialize_graph(config):
    """Define graph, nodes, and edges as per the logic."""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("websearch", web_search)  # web search
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generate

    # Build graph
    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
            "max retries": END,
        },
    )

    # Compile the graph
    graph = workflow.compile()

    # Save the Mermaid graph image to the assets folder
    image_path = os.path.join("./assets", "workflow_graph.png")
    graph_image = graph.get_graph().draw_mermaid_png()
    with open(image_path, "wb") as f:
        f.write(graph_image)

    return graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_workflow(state: Dict[str, Any], config: Dict[str, Any]):
    """Run the main workflow."""
    graph = initialize_graph(config)
    events = []
    logger.info(f"Initial state: {state}")
    logger.info(f"Config: {config}")
    for event in graph.stream(state, stream_mode="values"):
        events.append(event)
        logger.info(f"Event: {event}")
    logger.info(f"Events: {events}")
    return events
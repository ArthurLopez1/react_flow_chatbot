import os
import logging
from typing import List, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from .vectorstore import VectorStoreManager
from .llm_models import LLM
import json
import re
import uuid
from langchain.schema import Document
from langchain_cohere.rerank import CohereRerank
from langchain_core.output_parsers import StrOutputParser, PydanticToolsParser
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the LLM once
llm = LLM()

class SimpleGraphState(TypedDict):
    """Graph state containing information for each graph node."""
    question: str
    intent: str
    answers: List[str]
    question_list: List[str]
    generation: str
    documents: List[str]

intent_prompt = """You are an AI assistant tasked with detecting the intent of an incoming message. Your goal is to categorize the message into one of two intents: greeting or metadata_query.
Here are the definitions for each intent:
1. Greeting: A simple salutation or farewell, such as "Hello," "Hi," "Good morning," "Goodbye," or "See you later."
2. Metadata_query: Any request involving a count, sort, or retrieval of specific subsets of data. This includes questions about the latest documents, counting items, or sorting results.

<chat_history>
{chat_history}
<chat_history>

<question>
{question}
</question>

Carefully read and analyze the message to determine its intent based on the definitions provided above.
Guidelines for categorization:
- If the message is a simple greeting or farewell, classify it as "greeting"
- If the message asks for information about documents, regulations, or retrieving specific subsets of data, classify it as "metadata_query"
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

    # Extract intent from the response
    if "greeting" in intent_response:
        state["intent"] = "greeting"
    elif "metadata_query" in intent_response:
        state["intent"] = "metadata_query"
    else:
        state["intent"] = "unknown"

    print(f"Detected intent: {state['intent']}")
    return state

def handle_greeting(state: Dict[str, Any]):
    """
    Handle greeting intent by generating an immediate response and a follow-up question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates 'generation' in the state
    """
    print("---HANDLE GREETING---")
    response = "Hello! How can I assist you today?"
    state["generation"] = response
    print(f"Greeting response: {response}")
    return state

split_prompt = """You are tasked with analyzing a given question and, if possible, splitting it into multiple separate queries. Your goal is to identify distinct questions within a single input and present them as a list.

Here is the question you need to analyze:
<question>
{QUESTION}
</question>

Carefully examine the question to determine if it contains multiple distinct queries. Look for:
1. Multiple question marks
2. Conjunctions like "and" or "or" that might link separate questions
3. Different topics or subjects being addressed

If the question can be split, separate it into individual queries. Each query should be able to stand alone as a complete question.

Remember to handle edge cases:
- If the question cannot be split, return it as a single question in the list.
- If the question is empty or nonsensical, return an empty list of questions.

Provide your final output as a valid JSON array within <answer> tags."""

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
    prompt = split_prompt
    split_response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    questions = split_response.split('\n')
    state["question_list"] = questions
    print(f"Split questions: {questions}")
    return state

# Nodes
def _rerank_docs(documents: List[Document], question: str) -> List[Dict[str, Any]]:
    reranker = CohereRerank(model="rerank-multilingual-v3.0")
    reranked_results = reranker.rerank(documents, query=question)
    return reranked_results

def retrieve_documents(state: Dict[str, Any]):
    """
    Retrieve documents from vectorstore, rerank them, and filter based on relevance.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates 'documents' with a list of relevant documents
    """
    print("---RETRIEVE DOCUMENTS---")
    question = state["question"]
    config = state.get('config', {})

    # Retrieve documents from vector store
    vec_manager = VectorStoreManager()
    results = vec_manager.retrieve_documents(question)
    documents = []

    # Create Document objects and add similarity scores to metadata
    for content, score in results:
        doc = Document(page_content=content, metadata={"score": score})
        documents.append(doc)

    # Filter documents based on a similarity threshold
    similarity_threshold = 0.5
    docs_above_threshold = [doc for doc in documents if doc.metadata.get('score', 0) > similarity_threshold]
    print(f"Retrieved {len(documents)} documents from vector search")

    # Optionally perform internet search
    internet_search = config.get('internet_search', False)
    if internet_search and len(docs_above_threshold) == 0:
        # Implement internet search logic here
        # Example:
        # web_search_tool = TavilySearchResults()
        # web_docs = web_search_tool.invoke({"query": question}, max_results=3)
        # for doc in web_docs:
        #     document = Document(
        #         page_content=doc["content"],
        #         metadata={
        #             "type": "search",
        #             "url": doc["url"],
        #             "uuid": str(uuid.uuid4()),
        #             "source": "web_search"
        #         }
        #     )
        #     documents.append(document)
        pass

    # Rerank documents
    reranked_results = _rerank_docs(documents, question)

    # Filter reranked documents based on relevance score
    relevance_threshold = 0.7
    filtered_docs = []
    for result in reranked_results:
        if result['relevance_score'] > relevance_threshold:
            doc = documents[result['index']]
            doc.metadata['relevance_score'] = result['relevance_score']
            filtered_docs.append(doc)

    state["documents"] = filtered_docs
    print(f"Final filtered documents: {filtered_docs}")
    return state

def _filter_duplicate_docs(documents: List[Document]) -> List[Document]:
    seen_uuids = set()
    unique_documents = []
    for doc in documents:
        uuid = doc.metadata.get('uuid')
        if uuid not in seen_uuids:
            seen_uuids.add(uuid)
            unique_documents.append(doc)
    return unique_documents

def transform_documents(state: Dict[str, Any]):
    """
    Transform documents by filtering duplicates and discarding irrelevant information.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates 'documents' with filtered documents
    """
    print("---TRANSFORM DOCUMENTS---")
    documents = state.get("documents", [])
    if not documents:
        state["documents"] = []
        return state

    filtered_docs = _filter_duplicate_docs(documents)
    state["documents"] = filtered_docs
    print(f"Filtered documents: {filtered_docs}")
    return state

rag_prompt = """You are tasked with answering a question based on a set of provided sources. Your goal is to provide a well-researched answer with proper citations.

You will be given two inputs:

<sources>

{SOURCES}

</sources>

<question>

{QUESTION}

</question>

Follow these steps to complete the task:

1. Carefully read and analyze all the provided sources.

2. Identify information from the sources that is relevant to answering the question.

3. Formulate a comprehensive answer to the question using only the information from the provided sources. Do not include any external knowledge or information not present in the sources.

4. As you write your answer, include citations for each piece of information you use from the sources. Use superscript numbers for citations, linking to the corresponding reference at the bottom of the document.

5. Format your answer as markdown, following this structure:

   - Main content with superscript citations

   - A "References" section at the end

6. For each citation, create a corresponding reference entry in the "References" section. Number the references in the order they appear in the text.

7. Use the following format for citations and references:

   - In-text citation: Use superscript like "¹" to refer to a reference

   - Reference entry: Quote the sentence you used to give the answer.

8. Ensure that all citations in the text have a corresponding reference entry, and vice versa.

9. If the answer is not present in the source, don't answer the question. And don't provide any sources or references.

Always return the answer in a structured message within the "answer" key!

Remember to use only the information provided in the sources to answer the question. Do not include any external knowledge or personal opinions. Ensure that your answer is comprehensive, well-structured, and properly cited. Always answer in the same language as the question.
"""

def _rag_answer_chain():
    prompt_template = rag_prompt
    return prompt_template | llm | StrOutputParser()

async def rag_answer(state: Dict[str, Any]):
    question = state["question"]
    sources = ""
    for i, doc in enumerate(state["documents"], 1):
        sources += f"{i}. {doc.page_content}\n\n"
    final_answer_chain = _rag_answer_chain()
    response = await final_answer_chain.ainvoke({"SOURCES": sources, "QUESTION": question})
    state["generation"] = response
    return state

class CitedSources(BaseModel):
    source: str = Field(description="The source of the information")
    url: str = Field(description="The URL associated with the source")
    source_type: str = Field(description="The type of source")

cite_sources_prompt = """Here is the question you need to answer: 

<question> 

{QUESTION} 

</question> 

You will be provided with a list of sources. Each source contains the following information: 

1. Source text 

2. URL of the source 

Your task is to: 

1. Read through each source carefully. 

2. Identify the most relevant snippet of text (up to 300 characters) that helps answer the question. 

3. Extract only one snippet per source. 

4. Format the output as a function call for each relevant snippet found. 

Important guidelines: 

- Ensure that each extracted snippet is directly relevant to the question. 

- Keep the extracted text to a maximum of 300 characters. 

- Include only one snippet per source. 

- If a source does not contain relevant information, do not include it in the output. 

- Do not modify or summarize the text; extract it verbatim from the source. 

- Ensure that the extracted text is coherent and can stand alone as a piece of information. 

Now, proceed to analyze the following sources and extract the relevant citations: 

<sources> 

{SOURCES} 

</sources>"""

def _setup_cite_sources_chain():
    llm_with_tools = llm.bind_tools([CitedSources])
    prompt = cite_sources_prompt
    return prompt | llm_with_tools | PydanticToolsParser(tools=[CitedSources])

async def cite_sources(state: Dict[str, Any]):
    question = state["question"]
    sources = ""
    if len(state["documents"]) == 0:
        logger.error('Unable to answer, no sources found')
        state["cited_sources"] = ""
        return state
    for i, doc in enumerate(state["documents"], 1):
        source = doc.metadata.get('source', 'Unknown')
        url = doc.metadata.get('url', 'No URL provided')
        sources += f"{i}. {doc.page_content}\n\nURL: {url}\nSOURCE: {source}\n"
    cite_sources_chain = _setup_cite_sources_chain()
    response = await cite_sources_chain.ainvoke({"SOURCES": sources, "QUESTION": question})
    state["cited_sources"] = response
    return state

def initialize_simple_graph():
    """Define a simple graph, nodes, and edges."""
    workflow = StateGraph(SimpleGraphState)

    # Add nodes
    workflow.add_node("detect_intent", detect_intent)
    workflow.add_node("handle_greeting", handle_greeting)
    workflow.add_node("split_questions", split_questions)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("transform_documents", transform_documents)  # Add the new node
    workflow.add_node("rag_answer", rag_answer)
    workflow.add_node("cite_sources", cite_sources)

    # Build graph
    workflow.set_entry_point("detect_intent")
    workflow.add_conditional_edges("detect_intent", lambda state: state["intent"], {
        "greeting": "handle_greeting",
        "metadata_query": "split_questions"
    })
    workflow.add_edge("split_questions", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "transform_documents")  # Add the new edge
    workflow.add_edge("transform_documents", "rag_answer")
    workflow.add_edge("rag_answer", "cite_sources")
    workflow.add_edge("cite_sources", END)
    workflow.add_edge("handle_greeting", END)

    # Compile the graph
    graph = workflow.compile()

    # Save the Mermaid graph image to the assets folder
    image_path = os.path.join("assets", "workflow_graph.png")
    graph_image = graph.get_graph().draw_mermaid_png()
    with open(image_path, "wb") as f:
        f.write(graph_image)

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

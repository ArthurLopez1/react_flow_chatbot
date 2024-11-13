import os
import logging
from typing import List, Dict, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langgraph.graph import StateGraph, END
from .vectorstore import VectorStoreManager
from .llm_models import LLM
from langchain.schema import Document, HumanMessage
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.runnables import Runnable 
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from sklearn.metrics.pairwise import cosine_similarity  # Add this import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the LLM once
llm = LLM()

class SimpleGraphState(TypedDict):
    """Graph state containing information for each graph node."""
    question: str
    answers: List[str]
    generation: str
    documents: List[Document]  # Corrected type hint

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
    state["documents"] = documents      
    print(f"Retrieved documents: {documents}")
    return state

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

    # Filter duplicates based on UUID if available
    seen_uuids = set()
    unique_documents = []
    for doc in documents:
        uuid = doc.metadata.get('uuid')
        if uuid not in seen_uuids:
            seen_uuids.add(uuid)
            unique_documents.append(doc)

    # Perform semantic similarity checks to remove near-duplicates
    if unique_documents:
        embeddings = [VectorStoreManager().embed_text(doc.page_content) for doc in unique_documents]
        similarity_matrix = cosine_similarity(embeddings)
        filtered_documents = []
        for i, doc in enumerate(unique_documents):
            if all(similarity_matrix[i][j] < 0.9 for j in range(len(unique_documents)) if i != j):
                filtered_documents.append(doc)
        unique_documents = filtered_documents

    state["documents"] = unique_documents
    print(f"Filtered documents: {unique_documents}")
    print(f"Number of documents after filtering: {len(unique_documents)}")  # New debugging print
    return state

rag_prompt = """You are tasked with answering a question based on the sources. Your goal is to provide a well-researched answer with proper citations. Answer in the same language as the question.

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

3. **Formulate a comprehensive answer to the question using the information from the provided sources. Write your answer exclusively in the same language as the question.

4. As you write your answer, include citations for each piece of information you use from the sources. Use superscript numbers for citations, linking to the corresponding reference at the bottom of the document.

5. Format your answer as markdown, following this structure:

   - Main content with your answer in the language of the question.
   
   - A "References" section at the end

6. For each citation, create a corresponding reference entry in the "References" section. Number the references in the order they appear in the text.

7. Use the following format for citations and references:

   - In-text citation: Use superscript like "¹" to refer to a reference
   
   - Reference entry: Include the sentence and the exact page number from the source.

8. Ensure that all citations in the text have a corresponding reference entry, and vice versa.

9. If the answer is not present in the source, don't answer the question. And don't provide any sources or references.

Always return the answer in a structured message within the "answer" key!

Remember to use the information provided in the sources to answer the question. Ensure that your answer is comprehensive, well-structured, and properly cited. **Always answer in the same language as the question.**
"""

def _rag_answer_chain():
    # Define system and human messages
    system_message = SystemMessagePromptTemplate.from_template(
        "You are fluent in Swedish and English. Answer the question based on the provided sources."
    )
    human_message = HumanMessagePromptTemplate.from_template(rag_prompt)
    # Create a chat prompt template with messages
    prompt_template = ChatPromptTemplate.from_messages([system_message, human_message])
    return LLMChain(prompt=prompt_template, llm=llm.llm)

def rag_answer(state: Dict[str, Any]):
    if not state.get("documents"):
        logger.error('Unable to generate answer, no documents found')
        state["generation"] = "No documents found to generate an answer."
        return state

    try:
        question = state["question"]

        sources = ""
        for i, doc in enumerate(state["documents"], 1):
            sources += f"{i}. {doc.page_content}\n\n"
        final_answer_chain = _rag_answer_chain()
        # Log the formatted prompt for debugging
        formatted_prompt = final_answer_chain.prompt.format(SOURCES=sources, QUESTION=question)
        logger.info(f"Formatted prompt:\n{formatted_prompt}")
        logger.info(f"Number of sources provided to LLM: {len(state['documents'])}")  # New debugging print
        # Invoke the chain with input variables
        response = final_answer_chain.invoke({"SOURCES": sources, "QUESTION": question})
        logger.info(f"LLM response: {response}")  # Log the response from the LLM
        # Ensure the response is properly formatted
        if isinstance(response, dict) and 'text' in response:
            formatted_response = f"{response['text']}\n\nSources:\n\n\"{sources}\""
            state["generation"] = formatted_response  # The response is a string
        else:
            logger.error(f"Unexpected response format: {response}")
            state["generation"] = "Error generating answer."
        return state
    except Exception as e:
        logger.error(f"Error in rag_answer: {e}", exc_info=True)
        state["generation"] = "Error generating answer."
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
    # Replace the raw prompt string with a PromptTemplate
    prompt_template = PromptTemplate(
        input_variables=["SOURCES", "QUESTION"],
        template=cite_sources_prompt
    )
    # Update the LLM instantiation using ChatOllama
    llm_with_tools = ChatOllama(
        model=llm.llm.model,
        temperature=llm.llm.temperature,
        functions=[CitedSources]
    )
    # Construct the chain using Runnables
    return prompt_template | llm_with_tools | PydanticToolsParser(tools=[CitedSources])

def cite_sources(state: Dict[str, Any]):
    if not state.get("documents"):
        logger.error('Unable to answer, no sources found')
        state["cited_sources"] = ""
        return state

    question = state["question"]
    sources = ""
    for i, doc in enumerate(state["documents"], 1):
        source = doc.metadata.get('source', 'Unknown')
        url = doc.metadata.get('url', 'No URL provided')
        document_name = doc.metadata.get('document_name', 'Unknown Document')  # Retrieve document name
        sources += f"{i}. {doc.page_content}\n\nURL: {url}\nSource: {source}\nReferens: {document_name}\n"
    cite_sources_chain = _setup_cite_sources_chain()
    response = cite_sources_chain.invoke({"SOURCES": sources, "QUESTION": question})
    state["cited_sources"] = response
    return state

def initialize_simple_graph():
    """Define a simple graph, nodes, and edges."""
    workflow = StateGraph(SimpleGraphState)
    
    # Add nodes
    # Removed "__start__" node as it is reserved
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("transform_documents", transform_documents)
    workflow.add_node("rag_answer", rag_answer)
    workflow.add_node("cite_sources", cite_sources)

    # Build graph
    workflow.set_entry_point("retrieve_documents")  # Set "retrieve_documents" as entry point
    # Removed edge from "__start__" to "retrieve_documents"
    workflow.add_edge("retrieve_documents", "transform_documents")
    workflow.add_edge("transform_documents", "rag_answer")
    workflow.add_edge("rag_answer", "cite_sources")
    workflow.add_edge("cite_sources", END)

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
    # Define initial state
    state = {
        "question": "Hur ofta analyseras väderdata i VädErs-modellen?"
    }

    # Run the simplified workflow
    events = run_simple_workflow(state)

    # Output the final generated answer
    final_state = events[-1] if events else {}
    generated_answer = final_state.get("generation", "No answer generated.")
    print("Final Generated Answer:")
    print(generated_answer)
    logger.info(f"Final state: {final_state}")
import time
from typing import TypedDict, List, Annotated, Literal, Optional, Dict, Any, Type, Tuple
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ConfigDict
from operator import add
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool, StructuredTool
from langgraph.prebuilt import ToolNode, InjectedState
import ast
import json
import asyncio
from typing import TypedDict, Dict, List, Any, Optional
from langgraph.graph import StateGraph, END
import uuid
import operator
import random
import os
from dotenv import load_dotenv

from sphinx.addnodes import desc_inline
from sympy.codegen.ast import continue_

from LLM import split_reply
from utils import log_entry
from main import *
from RAGSystem import *
from ParseInputPDF import *
from SOM_Justification import *
from QraftLite_Justification import *

load_dotenv()
MODEL1_URL = os.environ.get("MODEL1_URL")
MODEL1_NAME = os.environ.get("MODEL1_NAME")
MODEL1_API_KEY = os.environ.get("MODEL1_API_KEY")

local_llm = ChatOpenAI(
    base_url=MODEL1_URL,
    api_key=MODEL1_API_KEY,
    model=MODEL1_NAME,
    temperature=0.1,
)

MAX_STEPS = 20
RESET_MARKER = "<<RESET_TO_PHASE_SUMMARIES>>"


def clean_llm_output(msg: AIMessage) -> AIMessage:
    """Post-processes every LLM response to remove <think>...</think> blocks, preserving the AIMessage object."""
    if isinstance(msg, AIMessage) and isinstance(msg.content, str):
        cleaned = split_reply(msg.content)
        msg.content = cleaned if cleaned else msg.content
    return msg


def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}


def extract_usage_from_response(response, prompt_name: str, MODEL_NAME: str = "UTM") -> tuple[int, int, int] | None:
    """
    Extract (total, prompt, completion) tokens from LangChain AIMessage.
    Returns None if usage data isn't available.
    """
    # Try newer LangChain format
    usage = None
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        meta = response.usage_metadata
        usage = (
            meta.get("total_tokens", 0),
            meta.get("input_tokens", 0),
            meta.get("output_tokens", 0)
        )
    elif hasattr(response, "response_metadata"):
        meta = response.response_metadata.get("usage", {})
        if meta:
            usage = (
                meta.get("total_tokens", 0),
                meta.get("prompt_tokens", 0),
                meta.get("completion_tokens", 0)
            )

    if usage:
        total, input, output = usage
        return [{
            "total_tokens": total,
            "input_tokens": input,
            "output_tokens": output,
            "prompt_name": prompt_name,
            "model": MODEL_NAME
        }]
    else:
        return []


def add_messages(left: list[BaseMessage], right: list[BaseMessage]) -> list[BaseMessage]:
    """
    Custom reducer for messages.
    If a reset marker is present, clears old messages and uses only the new ones.
    Otherwise, appends/replaces messages as normal.
    """
    # Check if right contains a reset marker
    has_reset = any(
        isinstance(msg, SystemMessage) and RESET_MARKER in msg.content
        for msg in right
    )

    if has_reset:
        # Clear old messages, return only the new ones (excluding the marker)
        x = [msg for msg in right if not (isinstance(msg, SystemMessage) and RESET_MARKER in msg.content)]
        return x

    # Normal append/replace logic
    new_messages = left.copy()
    for msg in right:
        if isinstance(msg, ToolMessage) and msg.tool_call_id:
            # Replace existing tool message with same ID
            replaced = False
            for i, existing_msg in enumerate(new_messages):
                if isinstance(existing_msg, ToolMessage) and existing_msg.tool_call_id == msg.tool_call_id:
                    new_messages[i] = msg
                    replaced = True
                    break
            if not replaced:
                new_messages.append(msg)
        else:
            new_messages.append(msg)
    return new_messages


class EvaluateVerdictsSchema(BaseModel):
    model_config = ConfigDict(extra='ignore')  # Ignores hallucinated fields
    next_classification_type: Literal[
        "Naive_RAG", "Naive_RAG_Enhanced", "Quotes_RAG", "Quotes_RAG_Enhanced", "Vectorless_RAG", "FINALISE"] = Field(
        description="Must exactly match: Naive_RAG, Naive_RAG_Enhanced, Quotes_RAG, Quotes_RAG_Enhanced, Vectorless_RAG or FINALISE if no next classification is needed."
    )
    reclassify_mode: Literal["ALL", "INCONCLUSIVE_ONLY"] = Field(
        default="ALL",
        description="Choose 'INCONCLUSIVE_ONLY' to only re-classify facts that are currently INCONCLUSIVE (saves time/tokens). Choose 'ALL' to re-classify all facts from scratch."
    )
    reason: str = Field(description="Reason why you chose the next_classification_type")


class JustifyVerdictsSchema(BaseModel):
    model_config = ConfigDict(extra='ignore')
    next_justification_type: Literal["Basic", "QraftLite", "SOM", "END"] = Field(
        description="Must exactly match: Basic, QraftLite, SOM, or END if no next justfication is needed.")
    reason: str = Field(description="Reason why you chose the next_justification_type")


class SingleAgentStateGraph(TypedDict):
    # Input Parameters
    filename: str
    reference_pdfs_path: str
    output_path: str
    semaphore: Any
    report: Any
    start_time: Any
    usage_logs: Annotated[List[Dict[str, Any]], operator.add]

    full_text: str
    papers: dict
    reference_dict: dict
    titles: dict
    citation_list: list
    facts: dict
    problems: list
    groups: dict
    references: dict
    citation_to_filename: dict
    verdicts: Dict

    # Classification & RAG config
    classification_type: Literal[
        "Naive_RAG", "Naive_RAG_Enhanced", "Quotes_RAG", "Quotes_RAG_Enhanced", "Vectorless_RAG"]
    max_number_of_chunks: int
    max_number_of_enhancement_chunks: int
    reclassify_mode: Optional[str]

    justification_type: Literal["Basic", "QraftLite", "SOM"]

    verdict_snapshots: Dict
    retry_count: int
    should_retry: bool
    best_verdicts: Dict
    next_classification_type: Optional[str]
    next_justification_type: Optional[str]

    messages: Annotated[list[BaseMessage], add_messages]
    active_phase: str
    phase_summaries: Annotated[List[str], operator.add]


faiss_indexes = None
chunk_stores = None


# ========= EXTRAS ========= #

def strip_think_tags(msg: AIMessage) -> AIMessage:
    """Post-processes every LLM response to remove <think>...</think> blocks."""
    if isinstance(msg, AIMessage) and isinstance(msg.content, str):
        # Apply your existing function
        cleaned = split_reply(msg.content)
        msg.content = cleaned if cleaned else msg.content  # Fallback if empty
    return msg


# ========= PREPROCESSIG TOOLS ========= #

@tool
async def process_texts(state: Annotated[dict, InjectedState]) -> dict:
    """Process the input PDF and Reference PDFs. Extract citation list."""
    print("[TOOL CALL] process_texts")
    try:
        filename = state["filename"]
        reference_pdfs_path = state["reference_pdfs_path"]
        semaphore = state["semaphore"]
        full_text, _ = await asyncio.to_thread(extract_text, filename=filename, find_reference_strings=False)
        handle_reference_paper_task = asyncio.create_task(extract_reference_paper_text(reference_pdfs_path))
        handle_paragraph_task = asyncio.create_task(
            process_paragraphs(filename=filename, semaphore=semaphore))
        handle_sentence_task = asyncio.create_task(
            process_sentences(full_text=full_text, semaphore=semaphore))

        paragraph_citation_list, (citation_list, reference_string), papers = await asyncio.gather(
            handle_paragraph_task,
            handle_sentence_task,
            handle_reference_paper_task
        )
        citation_list.extend(paragraph_citation_list)
        references_dict, invalid_references, titles = old_extract_references(reference_string=reference_string, filename=state["filename"])

        # Return a compact summary + raw data for state storage
        return {
            "summary": f"Extracted {len(citation_list)} citations, {len(papers)} papers, {len(titles)} titles. Next step is 'extract_facts_tool'.",
            "full_text": full_text,
            "citation_list": citation_list,
            "papers": papers,
            "reference_dict": references_dict,
            "titles": titles
        }
    except Exception as e:
        return {"summary": "Error!".format(e)}


@tool
async def extract_facts_tool(state: Annotated[dict, InjectedState]) -> dict:
    """Extract facts - Currently don't parse facts"""
    print("[TOOL CALL] extract_facts_tool")
    try:
        facts, problems = extract_facts(citations=state["citation_list"])
        return {
            "summary": f"Found {len(facts)} facts and {len(problems)} problems. Next step is 'group_and_match', or (optionally) 'get_facts_keys_tool'.",
            "facts": facts,
            "problems": problems
        }
    except Exception as e:
        return {"summary": f"Error! Ensure that 'process_texts' tool runs before using this tool! {e}"}


@tool
async def get_citation_keys_tool(facts_key: str, state: Annotated[dict, InjectedState]) -> dict:
    """(Optional) Returns a list of all citation keys currently in the facts dictionary."""
    print("[TOOL CALL] get_facts_keys_tool")
    try:
        facts_data = state.get("facts", {})
        keys = list(facts_data.keys()) if facts_data else []
        # Return a short summary for the LLM's context window
        return {
            "summary": f"Found {len(keys)} keys. List: {keys}\n\nNext step is 'remove_facts_key_tool' or 'group_and_match'."}
    except Exception as e:
        return {"summary": f"Error! Ensure that 'extract_facts_tool' tool runs before using this tool! {e}"}


@tool
async def remove_facts_key_tool(facts_key: str, key_to_remove: str, state: Annotated[dict, InjectedState]) -> dict:
    """(Optional) Removes a specific citation key from the facts dictionary. Use only for duplicate/multi-citation keys like '[1-3]' or '(A et al.; B et al.)'."""
    print("[TOOL CALL] remove_facts_key_tool")
    try:
        facts_data = state.get("facts", {}).copy()

        if facts_data and key_to_remove in facts_data:
            del facts_data[key_to_remove]
            # Return the UPDATED facts dict so LangGraph merges it into state,
            # and a summary for the LLM to read.
            return {
                "summary": f"Successfully removed key '{key_to_remove}'. Next step is 'group_and_match', or (optionally) 'get_facts_keys_tool', or (optionally) 'remove_facts_key_tool'",
                "facts": facts_data
            }
        else:
            return {"summary": f"Key '{key_to_remove}' not found or facts dict is empty. No action taken."}
    except Exception as e:
        return {"summary": f"Error! Ensure that 'extract_facts_tool' tool runs before using this tool! {e}"}


@tool
async def group_and_match(state: Annotated[dict, InjectedState]) -> dict:
    """Group facts by in-text citation & match them to reference papers"""
    print("[TOOL CALL] group_and_match")
    try:
        groups, group_problems = group_facts(facts=state["facts"], references=state["reference_dict"])
        references, citation_to_filename, unmatched = await match_input_ref_papers(papers=state["papers"],
                                                                                   reference_list=state[
                                                                                       "reference_dict"],
                                                                                   titles=state["titles"],
                                                                                   semaphore=state["semaphore"])
        state["report"].write(f"Number of unmatched references: {len(unmatched)}")
        return {
            "summary": f"Grouped into {len(groups)} clusters. Matched {len(references)} papers. {len(unmatched)} unmatched. Next step is 'create_vector_database'",
            "groups": groups,
            "references": references,
            "citation_to_filename": citation_to_filename,
        }
    except Exception as e:
        return {"summary": f"Error! Ensure that 'extract_facts_tool' tool runs before using this tool! {e}"}


@tool
async def create_vector_database(state: Annotated[dict, InjectedState]) -> dict:
    """Creates a FAISS vector database."""
    global faiss_indexes, chunk_stores
    print("[TOOL CALL] create_vector_database")
    try:
        faiss_indexes, chunk_stores = create_FAISS_indices(references=state["references"])
        return {
            "summary": f"Built {len(faiss_indexes)} FAISS indexes and {len(chunk_stores)} chunk stores. This is the last preproecssing step.",
            # "faiss_indexes": faiss_indexes,
            # "chunk_stores": chunk_stores
        }
    except Exception as e:
        return {"summary": f"Error! Ensure that 'group_and_match' tool runs before using this tool! {e}"}


# ========= PREPROCESSIG NODES ========= #

async def handle_preprocessing(state: SingleAgentStateGraph):
    """Calls the LLM to decide which preprocessing tools to run."""
    print("[NODE] handle_preprocessing")
    prompt = f"""Persona: You are an agent specialising in verifying citations within research papers. 

CRITICAL RULES:
- There is no human user in this interaction.
- All required inputs are already available in the state.
- NEVER ask questions.
- NEVER request clarification.
- NEVER refer to a user.
- NEVER ask for guidance, you will NOT get any aside from this prompt.
- ALWAYS use English.
- You MUST write 1-2 sentences reasoning about your next action.

TASK: Preprocess the input research paper PDF and its reference PDFs by calling the tools listed below. You MUST create the vector databases to finish the preprocessing.
ONLY USE THESE TOOLS: Call 'process_texts', 'extract_facts_tool', 'get_facts_keys_tool', 'remove_facts_key_tool', 'group_and_match', 'create_vector_database'.

Valid Pipeline:
1. 'process_texts'
2. 'extract_facts_tool'
4. 'group_and_match'
5. 'create_vector_database'

Tools: 'get_facts_keys_tool' and 'remove_facts_key_tool' MUST be ran before 'group_and_match'

INPUTS:
filename: {state['filename']}
reference_pdfs_path: {state['reference_pdfs_path']}
output_path: {state['output_path']}

You MUST follow the valid pipeline and you MUST create the vector databases using 'create_vector_database'.
"""

    messages = [SystemMessage(content=prompt)] + state.get("messages", []) + [SystemMessage(content=prompt)]

    response = await llm_preprocessing.ainvoke(messages)

    usage_log = extract_usage_from_response(response, prompt_name="SingleAgent_preprocessing")

    if hasattr(response, "tool_calls") and response.tool_calls:
        return {
            "messages": [response],
            "usage_logs": usage_log,
            "active_phase": "preprocess"
        }

    print(f"Attempting to finish preprocessing!")

    return {
        "messages": [response],
        "usage_logs": usage_log,
        "active_phase": "preprocess"
    }


def should_continue_preprocessing(state: SingleAgentStateGraph) -> str:
    """Router: If the LLM called tools, run them. Otherwise, preprocessing is done."""
    messages = state.get("messages", [])
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    if faiss_indexes is None:
        return "preprocess"
    return "end_preprocessing"


# ========= VERDICT HELPER FUNCTIONS ========= #

def count_verdicts(verdicts: dict[tuple[str, str, str, str]]) -> dict[str, int]:
    counts = {"TRUE": 0, "FALSE": 0, "INCONCLUSIVE": 0}
    for row in verdicts.values():
        try:
            counts[row[3]] += 1
        except KeyError:
            continue
    return counts


def separate_inconclusives(verdicts: dict[int, tuple[str, str, str, str]]):
    no_inconclusive, inconclusives = {}, {}
    for fact_id in verdicts.keys():
        row = verdicts[fact_id]
        if row[3] == "INCONCLUSIVE":
            inconclusives[fact_id] = row
        else:
            no_inconclusive[fact_id] = row
    return no_inconclusive, inconclusives


# ========= VERDICT TOOLS ========= #

@tool
def get_classification_descriptions_and_metrics():
    """Gets classification types, descriptions and metrics."""
    print("[TOOL CALL] get_classification_descriptions_and_metrics")
    x = """Below are the names, descriptions and metrics of the classification techniques:
    Naive_RAG: Basic Retrival Augmented Generation - uses context to check the veracity of a claim. Mean accuracy: 0.713. Mean token usage: Low.
    Naive_RAG_Enhanced: Basic Retrival Augmented Generation - reranks context, then uses highest quality context to check the veracity of a claim. Mean accuracy: 0.803. Mean token usage: Very High.
    Quotes_RAG: Retrieval Augmented Generation - first selects some relevent quotes form the provided context, then only uses the quotes to check th everacity of a claim. Mean accuracy: 0.68. Mean token usage: Low to Moderate.
    Quotes_RAG_Enhanced: Retrieval Augmented Generation - First reranks context, then selects some relevent quotes form the reranked context, then only uses the quotes to check th everacity of a claim. Mean accuracy: 0.68. Mean token usage: Moderate to Very High.
    Vectorless_RAG: Uses an agentic AI system to build a table of centents and traverse pages to check the veracity of a claim. Mean accuracy: 0.59. Mean token usage: Low to High.
    """
    return x


@tool
async def run_classification(classification_type: Literal[
    "Naive_RAG", "Naive_RAG_Enhanced", "Quotes_RAG", "Quotes_RAG_Enhanced", "Vectorless_RAG"],
                             max_number_of_chunks: int, max_number_of_enhancement_chunks: int,
                             state: Annotated[dict, InjectedState]):
    """Runs the chosen classification method."""
    # classification_type = state["classification_type"]
    # max_number_of_chunks = state["max_number_of_chunks"]
    # max_number_of_enhancement_chunks = state["max_number_of_enhancement_chunks"]
    print("[TOOL CALL] run_classification")
    print(f"Classification type: {classification_type}")

    if max_number_of_chunks > 20:
        max_number_of_chunks = 20
    elif max_number_of_chunks < 1:
        max_number_of_chunks = 1
    if max_number_of_enhancement_chunks > 10:
        max_number_of_enhancement_chunks = 10
    elif max_number_of_enhancement_chunks < 0:
        max_number_of_enhancement_chunks = 0

    if classification_type == 'Naive_RAG':
        verdicts = await naive_RAG_query(faiss_indexes=faiss_indexes, chunk_stores=chunk_stores,
                                         groups=state["groups"],
                                         facts=state["facts"], semaphore=state["semaphore"])
    elif classification_type == 'Naive_RAG_Enhanced':
        verdicts = await naive_RAG_query(faiss_indexes=faiss_indexes, chunk_stores=chunk_stores,
                                         groups=state["groups"],
                                         facts=state["facts"], semaphore=state["semaphore"], chunk_evaluation=True,
                                         max_chunks=max_number_of_chunks,
                                         enhancement_chunks=max_number_of_enhancement_chunks)
    elif classification_type == "Quotes_RAG":
        verdicts = await quotes_check_fact(faiss_indexes=faiss_indexes, chunk_stores=chunk_stores,
                                           groups=state["groups"],
                                           facts=state["facts"], semaphore=state["semaphore"])
    elif classification_type == "Quotes_RAG_Enhanced":
        verdicts = await quotes_check_fact(faiss_indexes=faiss_indexes, chunk_stores=chunk_stores,
                                           groups=state["groups"],
                                           facts=state["facts"], semaphore=state["semaphore"],
                                           chunk_evaluation=True,
                                           max_chunks=max_number_of_chunks,
                                           enhancement_chunks=max_number_of_enhancement_chunks)
    else:
        verdicts = await vectorless_rag_verify_facts(groups=state["groups"], facts=state["facts"],
                                                     citation_to_paper_names=state["citation_to_filename"],
                                                     reference_pdf_path=state["reference_pdfs_path"],
                                                     semaphore=state["semaphore"])
    verdicts = {k: v for k,v in verdicts.items() if v[3] == "TRUE" or v[3] == "FALSE" or v[3] == "INCONCLUSIVE"}
    with open(f"{state['output_path']}/{classification_type}_verdicts.json", "w") as f:
        json.dump(verdicts, f)
    counts = count_verdicts(verdicts)
    return {"verdicts": verdicts,
            "summary": f"Classification using {classification_type} successful!\nNumber of 'TRUE' verdicts: {counts['TRUE']}\nNumber of 'INCONCLUSIVE' verdicts: {counts['INCONCLUSIVE']}\nNumber of 'FALSE' verdicts: {counts['FALSE']}",
            "classification_type": classification_type
    }


@tool
async def reclassify_inconclusive(classification_type: Literal["Naive_RAG", "Naive_RAG_Enhanced", "Quotes_RAG", "Quotes_RAG_Enhanced", "Vectorless_RAG"],max_number_of_chunks: int, max_number_of_enhancement_chunks: int,state: Annotated[dict, InjectedState]):
    """Re-runs classification ONLY on facts that currently have an INCONCLUSIVE verdict, and merges the results back into the main verdicts."""
    print("[TOOL CALL] reclassify_inconclusive")

    current_verdicts = state.get("verdicts", {})
    inconclusive_fact_ids = set()

    for fact_id, row in current_verdicts.items():
        if isinstance(row, (tuple, list)) and len(row) > 3 and row[3] == "INCONCLUSIVE":
            inconclusive_fact_ids.add(fact_id)

    if not inconclusive_fact_ids:
        return {"summary": "No INCONCLUSIVE verdicts found to reclassify."}

    filtered_groups = {}
    for paper_id, fact_ids in state.get("groups", {}).items():
        filtered_ids = [fid for fid in fact_ids if fid in inconclusive_fact_ids]
        if filtered_ids:
            filtered_groups[paper_id] = filtered_ids

    filtered_facts = {fid: state["facts"][fid] for fid in inconclusive_fact_ids if fid in state.get("facts", {})}

    if max_number_of_chunks > 20:
        max_number_of_chunks = 20
    elif max_number_of_chunks < 1:
        max_number_of_chunks = 1
    if max_number_of_enhancement_chunks > 10:
        max_number_of_enhancement_chunks = 10
    elif max_number_of_enhancement_chunks < 0:
        max_number_of_enhancement_chunks = 0

    if classification_type == 'Naive_RAG':
        new_verdicts = await naive_RAG_query(faiss_indexes=faiss_indexes, chunk_stores=chunk_stores,
                                             groups=filtered_groups, facts=filtered_facts, semaphore=state["semaphore"])
    elif classification_type == 'Naive_RAG_Enhanced':
        new_verdicts = await naive_RAG_query(faiss_indexes=faiss_indexes, chunk_stores=chunk_stores,
                                             groups=filtered_groups, facts=filtered_facts, semaphore=state["semaphore"],
                                             chunk_evaluation=True, max_chunks=max_number_of_chunks,
                                             enhancement_chunks=max_number_of_enhancement_chunks)
    elif classification_type == "Quotes_RAG":
        new_verdicts = await quotes_check_fact(faiss_indexes=faiss_indexes, chunk_stores=chunk_stores,
                                               groups=filtered_groups, facts=filtered_facts,
                                               semaphore=state["semaphore"])
    elif classification_type == "Quotes_RAG_Enhanced":
        new_verdicts = await quotes_check_fact(faiss_indexes=faiss_indexes, chunk_stores=chunk_stores,
                                               groups=filtered_groups, facts=filtered_facts,
                                               semaphore=state["semaphore"],
                                               chunk_evaluation=True, max_chunks=max_number_of_chunks,
                                               enhancement_chunks=max_number_of_enhancement_chunks)
    else:  # Vectorless_RAG
        ref_path = state.get("reference_pdfs_path") or state.get("reference_pdfs")
        new_verdicts = await vectorless_rag_verify_facts(groups=filtered_groups, facts=filtered_facts,
                                                         citation_to_paper_names=state["citation_to_filename"],
                                                         reference_pdf_path=ref_path,
                                                         semaphore=state["semaphore"])

    # Filter out invalid verdicts just in case
    new_verdicts = {k: v for k, v in new_verdicts.items() if
                    isinstance(v, (tuple, list)) and len(v) > 3 and v[3] in ["TRUE", "FALSE", "INCONCLUSIVE"]}

    merged_verdicts = current_verdicts.copy()
    merged_verdicts.update(new_verdicts)

    counts = count_verdicts(merged_verdicts)
    return {
        "verdicts": merged_verdicts,
        "summary": f"Re-classified {len(new_verdicts)} INCONCLUSIVE verdicts using {classification_type}.\nNew counts - TRUE: {counts['TRUE']}, FALSE: {counts['FALSE']}, INCONCLUSIVE: {counts['INCONCLUSIVE']}",
        "classification_type": classification_type
    }


# ========= VERDICT NODES ========= #

async def choose_classification(state: SingleAgentStateGraph):
    """Initially choose a classification technique"""
    print("[NODE] choose_classification")
    prompt = f"""Persona: You are a specialist in verifying citations within research papers. 

CRITICAL RULES:
- There is no human user in this interaction.
- All required inputs are already available in the state.
- NEVER ask questions.
- NEVER request clarification.
- NEVER refer to a user.
- NEVER ask for guidance, you will NOT get any aside from this prompt.
- ALWAYS use English.
- You MUST write 1-2 sentences reasoning about your next action.

TASK: Now that the preprocessing is finished, classify the facts using a classification technique.
TOOLS:
- Call 'get_classification_descriptions_and_metrics' to get a list of names and descriptions of each type of classification technique.
- Call 'run_classification' with the chosen classification type and max_number_of_chunks to retrieve and max_number_of_enhancement_chunks to run the classification.
- Call 'reclassify_inconclusive' to ONLY re-classify facts that currently have an INCONCLUSIVE verdict.

RECLASSIFY MODE: {state.get('reclassify_mode', 'ALL')}
PREVIOUS CLASSIFICATION TYPE: {state.get('classification_type', 'None')}
"""
    messages = [SystemMessage(content=prompt)] + state.get("messages", []) + [SystemMessage(content=prompt)]

    response = await llm_classification.ainvoke(messages)

    usage_log = extract_usage_from_response(response, prompt_name="SingleAgent_choose_classification")

    if hasattr(response, "tool_calls") and response.tool_calls:
        return {
            "messages": [response],
            "usage_logs": usage_log,
            "active_phase": "choose_classification"
        }

    print(f"Attempting to finish choose_classification!")

    return {
        "messages": [response],
        "usage_logs": usage_log,
        "active_phase": "choose_classification"
    }


def route_after_choose_classification(state: SingleAgentStateGraph) -> str:
    """Route after choose_classification."""
    messages = state.get("messages", [])
    last_message = messages[-1]

    # If the LLM requested tools, go execute them globally
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "run_tools"
    if state.get("verdicts") is None:
        return "choose_classification"

    # If no tool calls, we successfully parsed the JSON and are done with this phase
    return "end_classification"


async def evaluate_verdicts(state: SingleAgentStateGraph) -> Dict[str, Any]:
    """Evaluate the verdicts and determine if """
    print("[NODE] evaluate_verdicts")
    verdicts = state.get("verdicts", {})
    retry_count = state.get("retry_count", 0)

    counts = count_verdicts(verdicts=verdicts)
    metrics = counts.copy()
    metrics["TOTAL"] = metrics["TRUE"] + metrics["INCONCLUSIVE"] + metrics["FALSE"]
    metrics["INCONCLUSIVE_RATE"] = round(metrics["INCONCLUSIVE"] / metrics["TOTAL"], 2)

    attempt_key = f"attempt_{retry_count + 1}"

    prompt = f"""Persona: You are a specialist in verifying citations within research papers. 

CRITICAL RULES:
- There is no human user in this interaction.
- All required inputs are already available in the state.
- NEVER ask questions.
- NEVER request clarification.
- NEVER refer to a user.
- NEVER ask for guidance, you will NOT get any aside from this prompt.
- ALWAYS use English.
- You MUST write 1-2 sentences reasoning about your next action.

TASK: Evaluate current verdict distribution and decide next step.
CURRENT ATTEMPT: {attempt_key}
CLASSIFICATION USED: {state.get('classification_type')}
METRICS: {metrics}
PREVIOUS ATTEMPT: {list(state.get('verdict_snapshots', {}).keys())[:-1]}
AVAILABLE METHODS: Naive_RAG, Naive_RAG_Enhanced, Quotes_RAG, Quotes_RAG_Enhanced, Vectorless_RAG
TOOLS: Call 'get_classification_descriptions_and_metrics' to get a list of names and descriptions of each type of classification technique.
DO NOT CALL 'run_classification' EVER.
RULES:
- If inconclusive_rate > 0.15, retry with a DIFFERENT method.
- If retry_count >= {state.get('max_retries', 2)}, return 'FINALISE'.
- For reclassify_mode: If you want to save time/tokens and only try to resolve the INCONCLUSIVE verdicts, choose 'INCONCLUSIVE_ONLY'. If you think the entire classification needs a fresh start, choose 'ALL'.
Output ONLY valid JSON matching EvaluateVerdictsSchema schema. 
FOR EXAMPLE:
{{"next_classification_type": "FINALISE", "reclassify_mode": "ALL", "reason":"inconclusive_rate is below 0.3"}}
Output ONLY valid JSON matching EvaluateVerdictsSchema schema. 
"""
    messages = [SystemMessage(content=prompt)] + state.get("messages", []) + [SystemMessage(content=prompt)]

    response = await llm_classification.ainvoke(messages)

    usage_log = extract_usage_from_response(response, prompt_name="SingleAgent_evaluate_verdicts")

    if hasattr(response, "tool_calls") and response.tool_calls:
        return {
            "messages": [response],
            "usage_logs": usage_log,
            "active_phase": "evaluate_verdicts"
        }

    try:
        # Clean markdown code blocks just in case the LLM added them
        content = re.sub(r"```(?:json)?\n?(.*?)\n?```", r"\1", response.content, flags=re.DOTALL).strip()
        results = EvaluateVerdictsSchema.model_validate_json(content)

        next_type = results.next_classification_type
        reason = results.reason

        should_retry = (next_type != "FINALISE" and retry_count < state.get("max_retries", 2))

        updated_snapshots = state.get("verdict_snapshots", {}).copy()
        updated_snapshots[attempt_key] = {
            "classification_type": state.get("classification_type"),
            "verdicts": verdicts,
            "metrics": metrics
        }

        # Return the parsed state updates
        return {
            "messages": [response],
            "usage_logs": usage_log,
            "verdict_snapshots": updated_snapshots,
            "retry_count": retry_count + 1 if should_retry else retry_count,
            "should_retry": should_retry,
            "next_classification_type": next_type if should_retry else None,
            "reclassify_mode": results.reclassify_mode if should_retry else "ALL",  # <--- SAVE MODE TO STATE
            "best_verdicts": verdicts,
            "active_phase": "evaluate_verdicts"
        }
    except Exception as e:
        print(f"[Evaluate] Tool loop failed to parse JSON: {e}")
        return {
            "messages": [response, SystemMessage(
                content=f"Error: Your output was not valid JSON. Error: {e}. Please output ONLY valid JSON matching EvaluateVerdictsSchema.")],
            "usage_logs": usage_log,
            "active_phase": "evaluate_verdicts"
        }


def route_after_evaluate_verdicts(state: SingleAgentStateGraph) -> str:
    """Router for the evaluate_verdicts phase."""
    messages = state.get("messages", [])
    last_message = messages[-1]

    # If the LLM requested tools, go execute them globally
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "run_tools"

    # If JSON parsing failed, the node appended an error message. Loop back to try again.
    if isinstance(last_message, SystemMessage) and "Error: Your output was not valid JSON" in last_message.content:
        return "evaluate_verdicts"

    # Otherwise, we have a valid decision. Check the state to see where to go next.
    if state.get("should_retry"):
        return "retry_classification"  # This should route back to choose_classification
    else:
        return "finalise_verdicts"  # This should route to choose_justification_type


# ========= JUSTIFICATION TOOLS ========= #

@tool
def get_justification_descriptions_and_metrics():
    """Gets the justification types, descriptions and metrics."""
    print("[TOOL CALL] get_justification_descriptions_and_metrics")
    return """Basic: Use the justification provided when creating the verdicts. Cost: None. Detail: Low
    QraftLite: Agentic AI system that creates justifications by gathering evidence, creating a draft and has a singly evaluation pass and improvemtn phase. Cost: Medium. Detail: High
    SOM: Agentic AI system with three agents, all of which gathering evidence, creating individual drafts and providing feedback to each other. Cost: Very high. Detail: Very High.
    """


@tool
async def run_justification(justification_type: Literal["Basic", "QraftLite", "SOM"],
                            state: Annotated[dict, InjectedState]):
    """Runs the chosen justification type."""
    print("[TOOL CALL] run_justification")
    print(f"Justification type: {justification_type}")
    global faiss_indexes
    global chunk_stores
    state["report"].write(f"Justification type: {justification_type}")
    initial_verdicts = state["verdicts"]
    groups = state["groups"]
    semaphore = state["semaphore"]

    outer_semaphore = asyncio.Semaphore(8)
    if justification_type != "SOM":
        outer_semaphore = asyncio.Semaphore(24)

    async def inner_run_justification(justifier_object, claim, initial_verdict, fact_id, row):
        async with outer_semaphore:
            try:
                result = await justifier_object.verify_citation(claim=claim, initial_verdict=initial_verdict)
                return (fact_id, row, result)
            except Exception as e:
                print(f"[TASK ERROR] {fact_id}: {e}")
                raise e

    async def run_basic_justification(context, fact, fact_id, row, initial_verdict, client, model):
        async with outer_semaphore:
            try:
                result = await naive_RAG_justification(context, fact, initial_verdict, client, model,
                                                       prompt_name="naive_RAG")
                result = json.loads(result)
                if "justification" in result.keys():
                    return (fact_id, row, result)
                else:
                    raise Exception("No justification found!")
            except Exception as e:
                print(f"[TASK ERROR] {fact}: {e}")
                raise e

    tasks = []
    for paper_id in faiss_indexes.keys():
        fact_group = groups.get(paper_id)
        if fact_group is None:
            continue
        for fact_id in fact_group:
            if justification_type == "Basic":
                row = initial_verdicts.get(fact_id)
                if row is not None:
                    claim = row[2]
                    intial_verdict = row[3]
                    chunks = query_faiss_index(query=claim, faiss_index=faiss_indexes, chunk_store=chunk_stores)
                    client, model = async_client()
                    tasks.append(asyncio.create_task(
                        run_basic_justification(chunks, claim, fact_id, row, intial_verdict, client, model)))
                    continue
            elif justification_type == "SOM":
                justifier = SOM_Justifier(faiss=faiss_indexes, chunks=chunk_stores, paper_id=paper_id,
                                          semaphore=semaphore)
            else:
                justifier = QraftLite(faiss=faiss_indexes, chunks=chunk_stores, paper_id=paper_id, semaphore=semaphore)
            justifier.set_paper_id(paper_id=paper_id)
            row = initial_verdicts.get(fact_id)
            if row is not None:
                tasks.append(asyncio.create_task(
                    inner_run_justification(justifier_object=justifier, claim=row[2], initial_verdict=row[3],
                                            fact_id=fact_id,
                                            row=row)))

    results = await asyncio.gather(*tasks)
    verdicts = {}
    # Process results
    for tup in results:
        fact_id = tup[0]
        row = tup[1]
        justification = tup[2]
        if isinstance(justification, Exception):
            print(f"[ERROR] Failed to verify {fact_id}: {justification}")
            continue
        new_row = (justification.get("verdict", row[0]), row[1], row[2], row[3], justification.get('justification'))
        verdicts[fact_id] = new_row

    return {"verdicts": verdicts, "summary": f"Justifcation of type {justification_type} successful!"}


@tool
def get_evaluation_criteria() -> str:
    """Get the rubric for evaluating justifications."""
    print("[TOOL CALL] get_evaluation_criteria")
    return """EVALUATION RUBRIC:
1. Completeness: The justification must be valid in full contextuality.
2. Coherence: Ensure the faithfulness/consistency between the veracity prediction and justification.
3. Interactivity: Put into consideration the users’ feedback - users will likely want to have a full explanation on why the verdict is TRUE/INCONCLUSIVE/FALSE.
4. Actionability: Provide the user with the needed suggestions for modifying the claim to change it from INCONCLUSIVE/FALSE to TRUE. **VERY IMPORTANT**
5. Novelty: Ensure the justification offers new information for the user to use to improve their citation if needed. 
6. Impartial justification: The justification must use the provided context and not an AI's own knowledge.
"""


@tool
def get_justifications_sample(state: Annotated[dict, InjectedState], sample_number: Annotated[
    int, "Number of sample to retrieve. Minimum=1, Maximum=10, Default=5."] = 5) -> str:
    """Get a sample of justifications to review."""
    print("[TOOL CALL] get_justifications_sample")
    verdict_keys = list(state.get("verdicts", {}).keys())
    if not verdict_keys: return "No verdicts found."

    x = random.sample(verdict_keys, min(sample_number, len(verdict_keys)))
    justifications = "######\n"
    for i in x:
        # FIX: Access the tuple from the state dictionary
        justifications += f"{state['verdicts'][i][4]}\n"
    justifications += "######"
    return justifications


# ========= JUSTIFICATION NODES ========= #

async def choose_justification_type(state: SingleAgentStateGraph):
    """Choose a justification method for the verdicts."""
    print("[NODE] choose_justification_type")
    prompt = f"""Persona: You are a specialist in verifying citations within research papers. 

CRITICAL RULES:
- There is no human user in this interaction.
- All required inputs are already available in the state.
- NEVER ask questions.
- NEVER request clarification.
- NEVER refer to a user.
- NEVER ask for guidance, you will NOT get any aside from this prompt.
- ALWAYS use English.
- You MUST write 1-2 sentences reasoning about your next action.

TASK: Choose a justificaiton method for the verdicts.
Call 'get_justification_descriptions_and_metrics' to get a list of names and descriptions of each type of justification technique.
Call 'run_justification' with the chosen justification type to create the justifications.
    """
    messages = [SystemMessage(content=prompt)] + state.get("messages", []) + [SystemMessage(content=prompt)]

    response = await llm_justification.ainvoke(messages)

    usage_log = extract_usage_from_response(response, prompt_name="SingleAgent_choose_justification_type")

    if hasattr(response, "tool_calls") and response.tool_calls:
        return {
            "messages": [response],
            "usage_logs": usage_log,
            "active_phase": "choose_justification_type"
        }

    return {
        "messages": [response],
        "usage_logs": usage_log,
        "active_phase": "choose_justification_type"
    }


async def evaluate_justifications(state: SingleAgentStateGraph):
    """Evaluate a sample of justifications and determine whether or not to use a different method."""
    print("[NODE] evaluate_justifications")
    prompt = f"""Persona: You are a specialist in verifying citations within research papers. 

CRITICAL RULES:
- There is no human user in this interaction.
- All required inputs are already available in the state.
- NEVER ask questions.
- NEVER request clarification.
- NEVER refer to a user.
- NEVER ask for guidance, you will NOT get any aside from this prompt.
- ALWAYS use English.
- You MUST write 1-2 sentences reasoning about your next action.

TASK: Choose a random sample of justifications to review and determine if the entire set of justificaitons needs to be remade. **Warning**: Creating justificaions again will have a high cost.
TOOLS: Call 'get_evaluation_criteria' to retrieve the evaluation rubric.
Call 'get_justifications_sample' to get a sample of the justifications (Minimum=1, Maximum=10, Default=5).

If another justification method should be used, outout the chosen name ("Basic", "QraftLite","SOM") for next_justification_type.
If no new justification should be made, output "END" for next_justification_type.

Output ONLY valid JSON matching JustifyVerdictsSchema schema. For example:
{{"next_justification_type": "END", "reason": "Sample of justifications meet the evaluation criteria well."}}
Output ONLY valid JSON matching JustifyVerdictsSchema schema.
"""
    messages = [SystemMessage(content=prompt)] + state.get("messages", []) + [SystemMessage(content=prompt)]

    response = await llm_justification.ainvoke(messages)

    usage_log = extract_usage_from_response(response, prompt_name="SingleAgent_evaluate_justifications")

    if hasattr(response, "tool_calls") and response.tool_calls:
        return {
            "messages": [response],
            "usage_logs": usage_log,
            "active_phase": "evaluate_justifications"
        }

    try:
        content = re.sub(r"```(?:json)?\n?(.*?)\n?```", r"\1", response.content, flags=re.DOTALL).strip()
        results = JustifyVerdictsSchema.model_validate_json(content)

        next_type = results.next_justification_type

        return {
            "messages": [response],
            "usage_logs": usage_log,
            "next_justification_type": next_type,
            "active_phase": "evaluate_justifications"
        }
    except Exception as e:
        print(f"[Evaluate] Tool loop failed to parse JSON: {e}")
        return {
            "messages": [response, SystemMessage(
                content=f"Error: Your output was not valid JSON. Error: {e}. Please output ONLY valid JSON matching JustifyVerdictsSchema.")],
            "usage_logs": usage_log,
            "active_phase": "evaluate_justifications",
            "next_justification_type": "END"
        }


def route_after_choose_justification(state: SingleAgentStateGraph) -> str:
    messages = state.get("messages", [])
    last_message = messages[-1]

    # If the LLM requested tools, go execute them globally
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "run_tools"

    # If no tool calls, we are done choosing and running, move to evaluation
    return "evaluate_justifications"


def route_after_evaluate_justifications(state: SingleAgentStateGraph) -> str:
    messages = state.get("messages", [])
    last_message = messages[-1]

    # If the LLM requested tools, go execute them globally
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "run_tools"

    # If JSON parsing failed, the node appended an error message. Loop back to try again.
    if isinstance(last_message, SystemMessage) and "Error: Your output was not valid JSON" in last_message.content:
        return "evaluate_justifications"

    # Otherwise, we have a valid decision. Check the state to see where to go next.
    next_type = state.get("next_justification_type", "END")
    if next_type == "END":
        return "end_justification"  # Route to create_metrics
    return "retry_justification"  # Route back to choose_justification_type


# ========= SUMMARY NODE ========= #

async def summarise_node(state: SingleAgentStateGraph) -> Dict[str, Any]:
    """Generates a concise summary of the completed phase using an LLM."""
    messages = state.get("messages", [])
    active_phase = state.get("active_phase", "unknown")

    # Build context for the summarizer
    context_lines = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            # Truncate long system prompts
            content = msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
            context_lines.append(f"[SYSTEM]: {content}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                tool_names = [tc["name"] for tc in msg.tool_calls]
                context_lines.append(f"[AI]: Called tools: {tool_names}")
            else:
                content = msg.content[:1000] + "..." if len(msg.content) > 1000 else msg.content
                context_lines.append(f"[AI]: {content}")
        elif isinstance(msg, ToolMessage):
            content = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
            context_lines.append(f"[TOOL {msg.name}]: {content}")

    context = "\n".join(context_lines)

    prompt = f"""You are a concise summarizer. Based on the conversation below from the "{active_phase}" phase, generate a brief summary (2-3 sentences max) of what was accomplished.

Conversation:
{context}

Output ONLY the summary text, nothing else."""

    response = await local_llm.ainvoke([SystemMessage(content=prompt)])
    summary_text = strip_think_tags(split_reply(response.content.strip()))

    # Create a summary message
    summary_message = SystemMessage(content=f"[PHASE SUMMARY - {active_phase}]: {summary_text}")

    # Log token usage for the summarizer call
    usage_logs = extract_usage_from_response(response, prompt_name=f"SingleAgent_summarise_phase_{active_phase}")

    return {
        "phase_summaries": [summary_message],
        "messages": [SystemMessage(content=RESET_MARKER), summary_message],
        "usage_logs": usage_logs
    }


def route_after_summarise(state: SingleAgentStateGraph) -> str:
    last_phase = state.get("active_phase", "")
    if last_phase == "preprocess":
        return "choose_classification"
    elif last_phase == "choose_classification":
        return "evaluate_verdicts"
    elif last_phase == "evaluate_verdicts":
        return "choose_justification_type"
    elif last_phase == "choose_justification_type":
        return "evaluate_justifications"
    elif last_phase == "evaluate_justifications":
        return "create_metrics"
    else:
        return "create_metrics"


# ========= TOOL NODE ========= #

def route_after_tools(state: SingleAgentStateGraph) -> str:
    """Dynamically routes back to the model node that requested the tools."""
    # Default to preprocess if somehow unset, but it should always be set
    return state.get("active_phase", "preprocess")


def extract_tool_state(state: SingleAgentStateGraph):
    """
    Intercepts ToolNode outputs.
    If a tool returned a dict with a 'summary', it sends ONLY the summary to the LLM,
    and merges the rest of the dict into the graph state.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_msg = messages[-1]

    # Check if the last message is a ToolMessage
    if getattr(last_msg, "type", None) == "tool":
        content_dict = None

        # 1. Check if LangChain stored the raw dict in the 'artifact' attribute (newer versions)
        if hasattr(last_msg, "artifact") and isinstance(last_msg.artifact, dict):
            content_dict = last_msg.artifact
        # 2. Check if content is already a dict
        elif isinstance(last_msg.content, dict):
            content_dict = last_msg.content
        # 3. Check if content is a stringified dict (very common in LangChain)
        elif isinstance(last_msg.content, str):
            if last_msg.name == 'get_classification_descriptions_and_metrics' or last_msg.name == "get_evaluation_criteria":
                pass
            try:
                # ast.literal_eval safely parses Python dict strings (even with single quotes)
                content_dict = ast.literal_eval(last_msg.content)
            except (ValueError, SyntaxError):
                try:
                    content_dict = json.loads(last_msg.content)
                except Exception:
                    print(
                        f"[DEBUG] Tool output is a string but NOT valid JSON/Python dict. Length: {len(last_msg.content)} chars")
                    print(last_msg.content)
                    pass  # It's just a normal string response, not a dict

        # If we successfully found a dict and it has a summary, process it
        if isinstance(content_dict, dict) and "summary" in content_dict:
            summary = content_dict.pop("summary")
            clean_tool_msg = ToolMessage(
                content=summary,
                tool_call_id=last_msg.tool_call_id,
                name=last_msg.name
            )
            returnMe = {"messages": [clean_tool_msg], **content_dict}
            # Return the clean message for the LLM, and the rest for State merging
            return returnMe
        elif isinstance(content_dict, dict):
            print(f"[DEBUG] Parsed dict, but NO 'summary' key found. Keys: {list(content_dict.keys())}")

    return {}


# ========= METRICS NODE (NO AI) ========= #

async def create_metrics_and_output_pdf(state: SingleAgentStateGraph):
    """Creates the metrics from running the Agent and create the highlighted PDF."""
    print("[NODE] create_metrics_and_output_pdf")
    highlight_text(pdf_path=state["filename"], verdicts=state["verdicts"], verification_type=state["classification_type"], output_path=state["output_path"])
    evaluation_metrics, _ = evaluate(verdicts=state["verdicts"], verification_type=state["classification_type"],reference_pdf_path=state["reference_pdfs_path"], output_path=state["output_path"])
    create_evaluation_metrics_graph(data={"Single Agent": evaluation_metrics}, output_filepath=f"{state['output_path']}/eval_evaluation_metrics.png")
    state["report"].write("\n\n######\nEnd of metrics\n\n")
    usage_logs = state.get("usage_logs", [])
    if usage_logs:
        for entry in usage_logs:
            log_entry(entry)
    close_logger()
    total_tokens, input_tokens, output_tokens, methods = agentic_pipeline_count_tokens(
        token_filepath=f'{state["output_path"]}\\token_usage.log')
    create_token_graph(methods=methods, output_filepath=state["output_path"])
    print(f"Total tokens: {total_tokens}\nTotal input tokens: {input_tokens}\nTotal output tokens: {output_tokens}\n")
    write_token_report(methods=methods, report=state["report"])


# ========= SETUP GRAPH ========= #
PREPROCESSING_TOOLS = [
    process_texts, extract_facts_tool, get_citation_keys_tool,
    remove_facts_key_tool, group_and_match, create_vector_database
]

CLASSIFICATION_TOOLS = [
    get_classification_descriptions_and_metrics, run_classification, reclassify_inconclusive
]

JUSTIFICATION_TOOLS = [
    get_justification_descriptions_and_metrics, run_justification,
    get_evaluation_criteria, get_justifications_sample
]

# Create phase-specific LLM instances
llm_preprocessing = local_llm.bind_tools(PREPROCESSING_TOOLS) | RunnableLambda(clean_llm_output)
llm_classification = local_llm.bind_tools(CLASSIFICATION_TOOLS) | RunnableLambda(clean_llm_output)
llm_justification = local_llm.bind_tools(JUSTIFICATION_TOOLS) | RunnableLambda(clean_llm_output)

tools = PREPROCESSING_TOOLS
tools.extend(CLASSIFICATION_TOOLS)
tools.extend(JUSTIFICATION_TOOLS)

tool_node = ToolNode(tools)

workflow = StateGraph(SingleAgentStateGraph)
workflow.add_node("extract_tool_state", extract_tool_state)
workflow.add_node("summarise_messages", summarise_node)
workflow.add_node("preprocess", handle_preprocessing)
workflow.add_node("run_tools", tool_node)
workflow.add_node("choose_classification", choose_classification)
workflow.add_node("evaluate_verdicts", evaluate_verdicts)
workflow.add_node("choose_justification_type", choose_justification_type)
workflow.add_node("evaluate_justifications", evaluate_justifications)
workflow.add_node("create_metrics", create_metrics_and_output_pdf)

workflow.set_entry_point("preprocess")

workflow.add_conditional_edges(
    "preprocess",
    should_continue_preprocessing,
    {
        "tools": "run_tools",
        "preprocess": "preprocess",
        "end_preprocessing": "summarise_messages"  # Move on to choose_classification
    }
)

workflow.add_conditional_edges(
    "choose_classification",
    route_after_choose_classification,
    {
        "run_tools": "run_tools",
        "choose_classification": "choose_classification",
        "end_classification": "summarise_messages"  # Move on to evaluate_verdicts
    }
)

workflow.add_conditional_edges(
    "evaluate_verdicts",
    route_after_evaluate_verdicts,
    {
        "run_tools": "run_tools",
        "evaluate_verdicts": "evaluate_verdicts",
        "retry_classification": "choose_classification",
        "finalise_verdicts": "summarise_messages"  # Move on to choose_justification_type
    }
)

workflow.add_conditional_edges(
    "choose_justification_type",
    route_after_choose_justification,
    {
        "run_tools": "run_tools",
        "evaluate_justifications": "summarise_messages"  # Move on to evaluate_justifications
    }
)

workflow.add_conditional_edges(
    "evaluate_justifications",
    route_after_evaluate_justifications,
    {
        "run_tools": "run_tools",
        "evaluate_justifications": "evaluate_justifications",
        "retry_justification": "choose_justification_type",
        "end_justification": "summarise_messages"  # Finish
    }
)

workflow.add_conditional_edges(
    "summarise_messages",
    route_after_summarise,
    {
        "choose_classification": "choose_classification",
        "evaluate_verdicts": "evaluate_verdicts",
        "choose_justification_type": "choose_justification_type",
        "evaluate_justifications": "evaluate_justifications",
        "create_metrics": "create_metrics"
    }
)
workflow.add_edge("run_tools", "extract_tool_state")
workflow.add_conditional_edges(
    "extract_tool_state",
    route_after_tools,
    {
        "preprocess": "preprocess",
        "choose_classification": "choose_classification",
        "evaluate_verdicts": "evaluate_verdicts",
        "choose_justification_type": "choose_justification_type",
        "evaluate_justifications": "evaluate_justifications"
    }
)

app = workflow.compile()


# TODO: Create plan & show user before executing
# Run with memory
# Don't do memoryless -> give summaries or something
#

async def run_agent(filename: str, reference_pdfs: str, output_path: str, max_retries: int = 3):
    print(f"# ****** Starting: {filename} ****** #")
    refresh_output_folder(output_path=output_path)
    semaphore = asyncio.Semaphore(24)
    report = open(f"{output_path}\\report.txt", 'w')
    await init_logger(output_filename=f"{output_path}\\token_usage.log")

    initial_state = {
        "filename": filename,
        "reference_pdfs_path": reference_pdfs,
        "output_path": output_path,
        "semaphore": semaphore,
        "report": report,
        "start_time": time.time(),

        "messages": [],
        "usage_logs": [],
        "retry_count": 0,
        "max_retries": max_retries,
        "verdict_snapshots": {},
        "should_retry": False,
        "best_verdicts": {},
    }

    config = {
        "configurable": {
            "thread_id": f"run_{uuid.uuid4().hex}"
        }
    }

    try:
        final_state = await app.ainvoke(initial_state, config=config)

        # 5. Post-execution cleanup/logging
        end_time = time.time()
        total_time = end_time - final_state["start_time"]
        final_state["report"].write(f"\nAgent finished successfully in {total_time:.2f} seconds.")
        final_state["report"].close()

        print(f"✅ Agent completed in {total_time:.2f} seconds, or {(total_time/60):.2f} minutes.")
        print(f"📊 Total LLM calls logged: {len(final_state.get('usage_logs', []))}")

        return final_state

    except Exception as e:
        print(f"❌ Agent failed with error: {e}")
        report.close()
        raise e


async def run_me():
    # try:
    #     print("* =====================  Computer Science started! ===================== *")
    #     await run_agent(
    #     filename="Evaluation set\\Computer Science\\input.pdf",
    #     reference_pdfs="Evaluation set\\Computer Science\\References",
    #     output_path="Output\\Computer Science\\SA AGAIN"
    #     )
    # except Exception as e:
    #     print(f"* ===================== Computer Science failed! ===================== *\n{e}")
    #     traceback.print_exc()
    # try:
    #     print("* =====================  Psychology started! ===================== *")
    #     await run_agent(
    #     filename="Evaluation set\\Psychology\\input.pdf",
    #     reference_pdfs="Evaluation set\\Psychology\\References",
    #     output_path="Output\\Psychology\\SA AGAIN"
    #     )
    # except Exception as e:
    #     print(f"* ===================== Psychology failed! ===================== *\n{e}")
    #     traceback.print_exc()
    try:
        print("* =====================  Biology started! ===================== *")
        await run_agent(
        filename="Evaluation set\\Biology\\input.pdf",
        reference_pdfs="Evaluation set\\Biology\\References",
        output_path="Output\\Biology\\SA AGAIN"
        )
    except Exception as e:
        print(f"* ===================== Psychology failed! ===================== *\n{e}")
        traceback.print_exc()


if __name__ == "__main__":
    print("Running!")
    asyncio.run(run_me())
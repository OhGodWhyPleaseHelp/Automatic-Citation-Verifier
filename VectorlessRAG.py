import sys
import json
import asyncio
import time
from pathlib import Path
import os
from tqdm import tqdm
from pydantic import BaseModel, Field
import agents
from dotenv import load_dotenv

from LLM import split_reply
from utils import log_usage

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import Agent, Runner, function_tool, set_tracing_disabled, set_default_openai_client
import openai

from myPageIndex.client import CustomPageIndexClient


load_dotenv()
MODEL1_URL = os.environ.get("MODEL1_URL")
MODEL1_NAME = os.environ.get("MODEL1_NAME")
MODEL1_API_KEY = os.environ.get("MODEL1_API_KEY")

local_client = openai.AsyncOpenAI(base_url=MODEL1_URL, api_key=MODEL1_API_KEY)
set_default_openai_client(local_client)
set_tracing_disabled(True)

_DIR = Path(__file__).parent
_REF_PDF_FOLDER = "ReferencePDFs"

AGENT_SYSTEM_PROMPT = """
You are ResearchPageIndex, a research paper document QA assistant.
You will verify a claim within a given research paper.

TOOL USE:
- Call get_document(doc_id) first to confirm status and page/line count.
- Call get_document_structure(doc_id) to identify relevant page ranges.
- Call get_page_content(doc_id, pages="5-7") with tight ranges; never fetch the whole document.
- Always pass the provided doc_id argument to every tool call.
Answer based ONLY on tool output. Be concise.
"""

VERIFICATION_PROMPT_TEMPLATE = """Please verify this claim: {CLAIM}
doc_id: {doc_id}
You MUST give your verdict in answer to the question in the following JSON format:
{"justification": "<give your reason for your verdict in 1-2 sentences>", "verdict": "TRUE" | "FALSE" | "INCONCLUSIVE"}
Choose only one verdict option: "TRUE" or "FALSE" or "INCONCLUSIVE".
Your entire response MUST be a strict valid JSON object and no explanation."""


def create_agent(client: CustomPageIndexClient) -> agents.Agent:
    """
    Run a document QA agent and return ONLY the parsed JSON verdict.
    No console output. Safe for batch processing.

    Returns:
        dict with {"justification": str, "verdict": str} OR None if parsing fails
    """

    @function_tool
    def get_document(doc_id: str) -> str:
        """Get document metadata: status, page count, name, and description."""
        return client.get_document(doc_id)

    @function_tool
    def get_document_structure(doc_id: str) -> str:
        """Get the document's full tree structure (without text) to find relevant sections."""
        return client.get_document_structure(doc_id)

    @function_tool
    def get_page_content(doc_id: str, pages: str) -> str:
        """
        Get the text content of specific pages or line numbers.
        Use tight ranges: e.g. '5-7' for pages 5 to 7, '3,8' for pages 3 and 8, '12' for page 12.
        For Markdown documents, use line numbers from the structure's line_num field.
        """
        return client.get_page_content(doc_id, pages)

    agent = Agent(
        name="PageIndex",
        instructions=AGENT_SYSTEM_PROMPT,
        tools=[get_document, get_document_structure, get_page_content],
        model=MODEL1_NAME,
    )

    return agent


async def query_agent(agent: Agent, claim: str, doc_id: str) -> str:
    """
    Creates prompt and queries the agent for a final reply.
    :param agent: Created Agent.
    :param claim: Fact to verify.
    :param doc_id: Document ID created by PageIndex.
    :return: Model's reply (if any)
    """
    prompt = VERIFICATION_PROMPT_TEMPLATE.replace("{CLAIM}", claim).replace("{doc_id}", doc_id)
    result = await Runner.run(agent, prompt)

    if hasattr(result, 'context_wrapper') and hasattr(result.context_wrapper, 'usage'):
        usage_obj = result.context_wrapper.usage
        total_tokens = getattr(usage_obj, 'total_tokens', 0)
        input_tokens = getattr(usage_obj, 'input_tokens', 0)
        output_tokens = getattr(usage_obj, 'output_tokens', 0)

        log_usage(
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prompt_name="VectorlessRAG_query_agent",
            model=MODEL1_NAME
        )

    return "" if not result.final_output else str(result.final_output)


async def create_vectorless_rag_client(semaphore: asyncio.Semaphore) -> CustomPageIndexClient:
    """
    Creates PageIndex Client.
    :param semaphore: Semaphore to limit calls.
    :return: Custom PageIndex Client.
    """
    return CustomPageIndexClient(semaphore=semaphore)


async def create_document_id(client: CustomPageIndexClient, paper_relative_path: str) -> str:
    """
    Creates a document ID from an input PDF.
    PageIndex handles creating the table of content structure for the document if necessary.
    :param client: Custom PageIndex client.
    :param paper_relative_path: relative path to the saved pdf.
    :return: Document ID
    """
    full_path = _DIR / paper_relative_path
    doc_id = next(
        (did for did, doc in client.documents.items() if doc.get('doc_name') == full_path.name),
        None,
    )
    if doc_id is None:
        doc_id = await client.index(str(full_path.absolute()))
    return doc_id


async def run_agent(agent: Agent, claim: str, doc_id: str) -> dict[str, str]:
    """
    Handles querying the agent and transforming its reply into a dictionary.
    :param agent: AI Agent.
    :param claim: Fact to verify.
    :param doc_id: Document ID generated by PageIndex.
    :return: Dictionary with justification and verdict.
    """
    final_reply = await query_agent(agent=agent, claim=claim, doc_id=doc_id)

    try:
        split_on_think = final_reply.split("</think>")[-1]
        try:
            verdict_dict = json.loads(split_on_think)
        except json.decoder.JSONDecodeError:
            try:
                parsed_final_reply = split_reply(final_reply, 'json')
                verdict_dict = json.loads(parsed_final_reply)
            except json.decoder.JSONDecodeError:
                split_on_justification = split_on_think.split('{"justification":')[-1]
                parse_into_string_dict = '{"justification":' + split_on_justification
                verdict_dict = json.loads(parse_into_string_dict)
        # parsed_final_reply = split_reply(final_reply, 'json')
        # verdict_dict = json.loads(parsed_final_reply)
    except Exception:
        print(f"Something went wrong with vectorless RAG!\nModel reply: {final_reply}")
        return {"justification": "Failed to parse model output", "verdict": "INCONCLUSIVE"}

    return verdict_dict


async def vectorless_rag_verify_facts(groups: dict, facts: dict, citation_to_paper_names: dict, reference_pdf_path:str,
                                      semaphore: asyncio.Semaphore):
    """
    Main function for vectorless RAG.
    Handles PageIndex, creating agents and querying the agents.
    :param groups: Facts groups {paper: [fact ids]}
    :param facts: Facts [(in_text_citation, fact, original_sentence)]
    :param citation_to_paper_names: Dictionary like: {"in_text_citation": "Paper_name.pdf"}
    :param semaphore: Semaphore to control rate limits.
    :return: Dictionary like: {"fact id": ('in_text_citation', 'fact', 'original sentence', 'verdict')}
    """
    outer_semaphore = asyncio.Semaphore(24)
    async def run_query(agent, doc_id, claim, fact_id):
        async with outer_semaphore:
            return (fact_id, await run_agent(agent=agent, claim=claim, doc_id=doc_id))

    async def run_create_doc_id(in_text_citation, client, path):
        async with outer_semaphore:
            return (in_text_citation, await  create_document_id(client=client, paper_relative_path=path))

    # Create client and agent
    pageindex_client = await create_vectorless_rag_client(semaphore=semaphore)
    agent = create_agent(client=pageindex_client)

    print("Creating all document IDs!")
    tasks = []
    for in_text_citation in citation_to_paper_names:
        paper_path = citation_to_paper_names[in_text_citation]
        path = f"{reference_pdf_path}/{paper_path}"
        if '.pdf' not in path:
            path = path + '.pdf'
        tasks.append(asyncio.create_task(
            run_create_doc_id(in_text_citation=in_text_citation, client=pageindex_client, path=path)))

    results = await asyncio.gather(*tasks)

    print("Querying agents now!")
    new_tasks = []
    for tup in results:
        in_text_citation = tup[0]
        doc_id = tup[1]
        linked_facts = groups.get(in_text_citation)
        for i in range(len(linked_facts)):
            fact_key = linked_facts[i]
            fact = facts.get(fact_key)[-1]
            new_tasks.append(asyncio.create_task(
                run_query(agent=agent, doc_id=doc_id, claim=fact, fact_id=fact_key)))

    responses = await asyncio.gather(*new_tasks)

    # Handle post-processing
    # Same as in RAGSystem
    print("Processing tasks!")
    facts_evidence = {}
    for tup in responses:
        fact_id = tup[0]
        reply = tup[1]  # This is in dict already
        final_verdict = reply.get("verdict")
        justification = reply.get('justification')
        fact = facts.get(fact_id)
        if fact is None:
            print(f"Fact ID: {fact_id}\tModel Reply: {reply}\nFacts: {facts}")
            break
        fact = (fact[0], fact[1], fact[2], final_verdict, justification)
        facts_evidence[fact_id] = fact

    return facts_evidence


def only_process_document(client: CustomPageIndexClient, paper_relative_path: str) -> str:
    """
    Creates a document ID from an input PDF.
    PageIndex handles creating the table of content structure for the document if necessary.
    :param client: Custom PageIndex client.
    :param paper_relative_path: relative path to the saved pdf.
    :return: Document ID
    """
    full_path = _DIR / paper_relative_path
    doc_id = next(
        (did for did, doc in client.documents.items() if doc.get('doc_name') == full_path.name),
        None,
    )
    if doc_id is None:
        doc_id = asyncio.run(client.index(str(full_path.absolute())))
    return doc_id
import traceback
from typing import TypedDict, List, Annotated, Literal, Optional, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ConfigDict
from operator import add
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool, StructuredTool
from langchain_core.load import dumps
import json
import asyncio
import random
import os
from dotenv import load_dotenv

from LLM import split_reply
from utils import log_entry, custom_count_tokens, count_messages_tokens
from RAGSystem import query_faiss_index

MAX_STEPS = 15

MODEL1_URL = os.environ.get("MODEL1_URL")
MODEL1_NAME = os.environ.get("MODEL1_NAME")
MODEL1_API_KEY = os.environ.get("MODEL1_API_KEY")

local_llm = ChatOpenAI(
    base_url=MODEL1_URL,
    api_key=MODEL1_API_KEY,
    model=MODEL1_NAME,
    temperature=0.1,
)

AGENT_PERSONAS = {
    "planner": "You are an AI Assistant. Your role is to gather relevant evidence pieces of information from context and plan an outline for a fact-checking review draft.",
    "writer": "You play the role of a human fact-checker. Your role is to use the provided outline and evidence .",
    "editor": "You play the role of a human expert editor. Your role is to review the given draft, ensure it is correct and fix any mistakes."
}


def strip_think_tags(msg: AIMessage) -> AIMessage:
    """Post-processes every LLM response to remove <think>...</think> blocks."""
    if isinstance(msg, AIMessage) and isinstance(msg.content, str):
        # Apply your existing function
        cleaned = split_reply(msg.content)
        msg.content = cleaned if cleaned else msg.content  # Fallback if empty
    return msg


def extract_usage_from_langchain_response(response) -> tuple[int, int, int] | None:
    """
    Extract (total, prompt, completion) tokens from LangChain AIMessage.
    Returns None if usage data isn't available.
    """
    # Try newer LangChain format
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        meta = response.usage_metadata
        return (
            meta.get("total_tokens", 0),
            meta.get("input_tokens", 0),
            meta.get("output_tokens", 0)
        )

    # Try older response_metadata format
    if hasattr(response, "response_metadata"):
        usage = response.response_metadata.get("usage", {})
        if usage:
            return (
                usage.get("total_tokens", 0),
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0)
            )

    return None

class ContextSchema(BaseModel):
    model_config = ConfigDict(extra='ignore')  # Ignores hallucinated fields
    verdict: Literal["TRUE", "FALSE", "INCONCLUSIVE"] = Field(
        description="Must exactly match: TRUE, FALSE, or INCONCLUSIVE"
    )
    selected_evidence: List[str] = Field(description="All relevant context to prove the claim aligns with the verdict.",
                                        default_factory=list)


class DraftOutlineSchema(BaseModel):
    model_config = ConfigDict(extra='ignore')
    draft_outline: str = Field(
        description="What should be included in the fact checking draft to convince a human the verdict is accurate for a claim.")


class DraftSchema(BaseModel):
    model_config = ConfigDict(extra='ignore')
    draft: str = Field(description="Detailed draft explaining the verdict type for the given claim.")


class FeedbackSchema(BaseModel):
    model_config = ConfigDict(extra='ignore')
    improvements: List[str] = Field(description="Improvements that should be made to the given draft.")
    what_went_well: List[str] = Field(description="What the given draft did well and should be retained.")


class QraftLiteGraphState(TypedDict):
    claim: str
    initial_verdict: Literal["TRUE", "FALSE", "INCONCLUSIVE"]
    verdict: Literal["TRUE", "FALSE", "INCONCLUSIVE"]
    selected_evidence: List[str]
    draft_outline: str
    draft: str
    improvements: List[str]
    what_went_well: List[str]
    usage_logs: Annotated[List[Dict[str, Any]], add]


class QraftLite():
    def __init__(self, faiss, chunks, paper_id, semaphore):
        self.local_llm = local_llm
        self.faiss_indexes = faiss
        self.chunk_stores = chunks
        self.paper_id = paper_id
        self.semaphore = semaphore

        self._setup_tools()
        self._setup_llm_chain()
        self._build_graph()

    def set_paper_id(self, paper_id):
        """
        Sets new paper id.
        :param paper_id: paper ID to use on FAISS VDB.
        :return: void.
        """
        self.paper_id = paper_id

    def set_evidence(self, evidence: List[str]):
        self.selected_evidence = evidence

    def _setup_tools(self):
        """Define tools as bound methods"""

        # @tool
        def retrieve_context_impl(claim: str, number_of_chunks_to_return: Annotated[
            int, "Number of context chunks to retrieve. Valid range: 5-15. Absolute max: 15. Default: 10."] = 10) -> str:
            """Retrieve background information and related evidence for a citation.
            Requires the exact paper_id and claim to query the knowledge base."""
            if self.faiss_indexes is None or self.chunk_stores is None:
                print(self.faiss_indexes)
                print(self.chunk_stores)
                raise Exception("Vector Database not initiliased!")
            if number_of_chunks_to_return > 15:
                number_of_chunks_to_return = 15
            elif number_of_chunks_to_return <= 0:
                number_of_chunks_to_return = 5
            f = self.faiss_indexes.get(self.paper_id)
            c = self.chunk_stores.get(self.paper_id)
            if f is None or c is None:
                print(self.paper_id)
                print(f)
                print(c)
                raise Exception("Can't get FAISS index and chunk store!")

            chunks = query_faiss_index(query=claim, faiss_index=f,
                                       chunk_store=c, top_k=number_of_chunks_to_return)

            return "\n".join(chunks)

        self.retrieve_context_tool = StructuredTool.from_function(
            func=retrieve_context_impl,
            name="retrieve_context",
            description="""Retrieve background information for a citation.""",
        )

        @tool
        def get_evaluation_criteria() -> str:
            """Get the official rubric for evaluating justifications."""
            return """
                EVALUATION RUBRIC:
                1. Completeness: The justification must be valid in full contextuality.
                2. Coherence: Ensure the faithfulness/consistency between the veracity prediction and justification.
                3. Interactivity: Put into consideration the users’ feedback - users will likely want to have a full explanation on why the verdict is TRUE/INCONCLUSIVE/FALSE.
                4. Actionability: Provide the user with the needed suggestions for modifying the claim to change it from INCONCLUSIVE/FALSE to TRUE. **VERY IMPORTANT**
                5. Novelty: Ensure the justification offers new information for the user to use to improve their citation if needed. 
                6. Impartial justification: The justification must use the provided context and not an AI's own knowledge.
                
                NEVER make a table - they are unreadable for the user!
                NEVER refer to the evaluation rubric in your draft.
                """

        @tool
        def get_selected_evidence() -> List[str]:
            """Get the selected evidence to use."""
            return self.selected_evidence

        # Store tools for later use
        # self.tools = [self.retrieve_context_tool, get_evaluation_criteria, get_selected_evidence]
        self.tools = [self.retrieve_context_tool, get_evaluation_criteria, ]
        self.TOOL_REGISTRY = {t.name: t for t in self.tools}

    def _setup_llm_chain(self):
        """Create LLM chain with tools + think-tag stripping"""

        def strip_think_tags(msg: AIMessage) -> AIMessage:
            if isinstance(msg, AIMessage) and isinstance(msg.content, str):
                cleaned = split_reply(msg.content)
                return msg.copy(update={"content": cleaned if cleaned else msg.content})
            return msg

        llm_with_tools = self.local_llm.bind_tools(self.tools)
        self.llm_with_tools = llm_with_tools | RunnableLambda(strip_think_tags)

    def _build_graph(self):
        """Construct the LangGraph workflow"""

        # Build workflow
        workflow = StateGraph(QraftLiteGraphState)

        workflow.add_node("collect_evidence", self.collect_evidence)  # Planner -> store locally?
        workflow.add_node("create_draft_outline", self.create_draft_outline)  # Planner
        workflow.add_node("write_draft", self.write_draft)  # Writer
        workflow.add_node("evaluate_draft", self.evaluate_draft)  # Editor
        workflow.add_node("improve_draft", self.improve_draft)  # Editor

        # Edges
        workflow.add_edge(START, "collect_evidence")
        workflow.add_edge("collect_evidence", "create_draft_outline")
        workflow.add_edge("create_draft_outline", "write_draft")
        workflow.add_edge("write_draft", "evaluate_draft")
        workflow.add_edge("evaluate_draft", "improve_draft")
        workflow.add_edge("improve_draft", END)

        self.app = workflow.compile()

    async def collect_evidence(self, state: QraftLiteGraphState):
        """Collect evidence from retrieve_context_tool and select most important evidence."""
        prompt = f"""Persona: {AGENT_PERSONAS.get("planner")}
TASK: Select relevant evidence that proves the given verdict. If the evidence does not support the Initial Verdict, you may change it to the correct one, either TRUE or FALSE. Avoid INCONCLUSIVE if possible.You must select AT LEAST 1 piece of evidence for the final verdict.
Call `retrieve_context_tool` to retrieve chunks of data. Only retrieve context once.
Claim: {state['claim']}
Initial Verdict: {state['initial_verdict']}

STRICT OUTPUT RULES:
1. Output ONLY valid JSON matching the ContextSchema schema. No markdown, no text before/after.
2. JSON must contain EXACTLY these fields: "verdict", "selected_evidence"
3. DO NOT include any other fields.
4. "selected_evidence" must be a list of strings (exact quotes from context).

EXAMPLE CORRECT FORMAT:
{{"verdict": "TRUE", "selected_evidence": ["exact quote 1", "exact quote 2"]}}

Output ONLY valid JSON matching ContextSchema schema.
Output:"""
        obj = await self.run_agent_with_tools(prompt,"QraftLite_collect_evidence", ContextSchema)
        if isinstance(obj, Exception):
            print(f"Collecting evidence failed! {obj}")
            return {"verdict": state['initial_verdict'], "selected_evidence": ["Error"]}
        results, usage = obj
        self._fallback_usage_logs.extend(usage)
        evidence = results.selected_evidence if results and results.selected_evidence else []
        if evidence == []:
            print(f"No evidence was selected!")

        self.set_evidence(evidence)
        return {"verdict": results.verdict,"selected_evidence": results.selected_evidence, "usage_logs":usage}

    async def create_draft_outline(self, state: QraftLiteGraphState) -> dict:
        """Create draft outline."""
        prompt = f"""Persona: {AGENT_PERSONAS.get("planner")}
TASK: Create draft outline for {state['claim']} using the given selected evidence. 
Claim: {state['claim']}
Verdict: {state['verdict']}
Selected Evidence:
######
{state['selected_evidence']}
######

STRICT OUTPUT RULES:
1. Output ONLY valid JSON matching the DraftOutlineSchema schema. No markdown, no text before/after.
2. JSON must contain EXACTLY this field ONLY: "draft_outline"
3. DO NOT include any other field.

EXAMPLE CORRECT FORMAT:
{{"draft_outline": "example text"}}

Output ONLY valid JSON matching DraftOutlineSchema schema.
Output:"""
        obj = await self.run_agent_with_tools(prompt, "QraftLite_create_draft_outline",DraftOutlineSchema)
        if isinstance(obj, Exception):
            print(f"Creating draft outline failed! {obj}")
            return {"draft_outline": "Error"}
        results, usage = obj
        self._fallback_usage_logs.extend(usage)
        if results is None:
            print(f"No draft outline was created!")

        return {"draft_outline": results.draft_outline, "usage_logs":usage}

    async def write_draft(self, state: QraftLiteGraphState) -> dict:
        """Fill draft outline with content."""
        prompt = f"""Persona: {AGENT_PERSONAS.get("writer")}
TASK: Using the given draft outline and the selected evidence, create a detailed justificaiton draft to explain the claim's veracity (the claim's verdict).
Call `get_evaluation_criteria` to retrieve the evaluation rubric, which you can use to write the draft.
NEVER make a table in your draft.
Claim: {state['claim']}
Verdict: {state['verdict']}
Selected Evidence:
######
{state['selected_evidence']}
######

STRICT OUTPUT RULES:
1. Output ONLY valid JSON matching the DraftSchema schema. No markdown, no text before/after.
2. JSON must contain EXACTLY this field ONLY: "draft"
3. DO NOT include any other field.

EXAMPLE CORRECT FORMAT:
{{"draft": "example text"}}

Output ONLY valid JSON matching DraftSchema schema.
Output:"""
        obj = await self.run_agent_with_tools(prompt, "QraftLite_write_draft",DraftSchema)
        if isinstance(obj, Exception):
            print(f"Writing draft failed! {obj}")
            return {"draft": "Error"}
        results, usage = obj
        self._fallback_usage_logs.extend(usage)
        if results is None:
            print("No draft was created!")

        return {"draft": results.draft, "usage_logs":usage}

    async def evaluate_draft(self, state: QraftLiteGraphState) -> dict:
        """Evaluate draft using rubric."""
        prompt = f"""Persona: {AGENT_PERSONAS.get("editor")}
TASK: Evaluate the given draft using the evaluation rubric. Create a list of improvements for the draft and what the draft did well. The draft should be a detailed justification to explain why the given claim has been calssed as TRUE/FALSE/INCONCLUSIVE (the verdict).
Call `get_evaluation_criteria` to retrieve the evaluation rubric.
Claim: {state['claim']}
Verdict: {state['verdict']}
Draft: {state['draft']}

STRICT OUTPUT RULES:
1. Output ONLY valid JSON matching the FeedbackSchema schema. No markdown, no text before/after.
2. JSON must contain EXACTLY these fields ONLY: "improvements", "what_went_well"
3. DO NOT include any other field.
4. "improvements" and "what_went_well" must be a list of strings.

EXAMPLE CORRECT FORMAT:
{{"improvements": ["improvement1", "improvement2"], "what_went_well": ["good feature 1", "good feature 2"]}}

Output ONLY valid JSON matching FeedbackSchema schema.
Output:"""
        obj = await self.run_agent_with_tools(prompt, "QraftLite_evaluate_draft", FeedbackSchema)
        if isinstance(obj, Exception):
            print(f"Evaluate draft failed! {obj}")
            return {"improvements": ["Error"], "what_went_well": ["Error"]}
        results, usage = obj
        self._fallback_usage_logs.extend(usage)
        if results is None:
            print("No feedback was created!")

        return {"improvements": results.improvements, "what_went_well": results.what_went_well, "usage_logs":usage}

    async def improve_draft(self, state: QraftLiteGraphState) -> dict:
        """Modify draft to improve it."""
        prompt = f"""Persona: {AGENT_PERSONAS.get("writer")}
TASK: Using the provided feedback, improve the draft to justify the verdict based on the given claim.
NEVER make a table in your draft.
Claim: {state['claim']}
Verdict: {state['verdict']}
Selected Evidence:
######
{state['selected_evidence']}
######
Initial Draft: {state['draft']}
Improvement Suggestions: {state['improvements']}
What went well: {state['what_went_well']}


STRICT OUTPUT RULES:
1. Output ONLY valid JSON matching the DraftSchema schema. No markdown, no text before/after.
2. JSON must contain EXACTLY this field ONLY: "draft"
3. DO NOT include any other field.

EXAMPLE CORRECT FORMAT:
{{"draft": "example text"}}

Output ONLY valid JSON matching DraftSchema schema.
Output:"""
        obj = await self.run_agent_with_tools(prompt, "QraftLite_improve_draft",DraftSchema)
        if isinstance(obj, Exception):
            print(f"Inproving draft failed! {obj}")
            return {"draft": state['draft']}
        results, usage = obj
        self._fallback_usage_logs.extend(usage)
        if results is None:
            print("No improved draft was created!")
        return {"draft": results.draft, "usage_logs":usage}

    async def run_agent_with_tools(self, prompt: str, prompt_name:str, structured_model: BaseModel) -> BaseModel:
        """Run tool-calling loop with agent-specific LLM (async version)"""
        global MAX_STEPS
        messages = [{"role": "system", "content": prompt}]
        usage_entries = []
        count = 0
        while count < MAX_STEPS:
            async with self.semaphore:
                response = await self.llm_with_tools.ainvoke(messages)

            usage = extract_usage_from_langchain_response(response)
            entry = ""
            if usage:
                total_tokens, input_tokens, output_tokens = usage
                usage_entries.append({
                    "total_tokens": total_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "prompt_name": prompt_name,
                    "model": "UTM"
                })
                if total_tokens > 20000:
                    x = random.randint(1, total_tokens)

            messages.append(response)

            if not response.tool_calls:
                try:
                    return structured_model.model_validate_json(response.content), usage_entries
                except Exception as e:
                    messages.append({"role": "user", "content": f"Return ONLY valid JSON. Error: {e}"})
                    continue

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                if tool_name in self.TOOL_REGISTRY:
                    # Tools are sync in your implementation, but we keep this flexible
                    tool_result = self.TOOL_REGISTRY[tool_name].invoke(tool_args)
                else:
                    tool_result = f"Unknown tool: {tool_name}"
                messages.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call["id"]
                })
            count += 1

        print(f"[WARNING] Agent hit max tool call steps ({MAX_STEPS}). Forcing final response.")
        messages.append({"role": "user",
                         "content": "You have reached the maximum number of tool calls. Please output your final JSON response NOW based on the information you have gathered."})
        async with self.semaphore:
            tools_llm= self.llm_with_tools
            response = await tools_llm.ainvoke(messages)
        try:
            return structured_model.model_validate_json(response.content), usage_entries
        except Exception as e:
            raise Exception(
                f"Agent failed to produce a valid response after {MAX_STEPS} tool calls. Error: {e}")

    async def verify_citation(self, claim: str,initial_verdict: Literal["TRUE", "FALSE", "INCONCLUSIVE"]) -> dict:
        """
        First create an object, then call this function to run.
        :param claim: Claim to check.
        :param initial_verdict: Initial verdict.
        :return: Dictionary containing improved draft, key='justification'
        """
        self._fallback_usage_logs = []
        try:
            result = await self.app.ainvoke({
                "claim": claim,
                "initial_verdict": initial_verdict,
                "selected_evidence": [],
                "draft_outline": None,
                "draft": None,
                "improvements": [],
                "what_went_well": []
            })
        except Exception as e:
            print(e)
            traceback.print_exc()
            for entry in self._fallback_usage_logs:
                log_entry(entry)
            return {"justification": "Error"}

        usage_logs = result.get("usage_logs", [])
        if usage_logs:
            for entry in usage_logs:
                log_entry(entry)

        return {
            "verdict": result.get("verdict", initial_verdict),
            "justification": result.get("draft", "Error!"),
        }
from typing import TypedDict, List, Annotated, Literal, Optional, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from operator import add
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool, StructuredTool
from langchain_core.load import dumps
import json
import asyncio
import json
import re
import random
from collections import Counter
import os
from dotenv import load_dotenv

from LLM import split_reply
from utils import log_entry, custom_count_tokens, count_messages_tokens
from RAGSystem import query_faiss_index

MAX_STEPS=15

load_dotenv()
MODEL1_URL = os.environ.get("MODEL1_URL")
MODEL1_NAME = os.environ.get("MODEL1_NAME")
MODEL1_API_KEY = os.environ.get("MODEL1_API_KEY")
MODEL2_URL = os.environ.get("MODEL2_URL")
MODEL2_NAME = os.environ.get("MODEL2_NAME")
MODEL2_API_KEY = os.environ.get("MODEL2_API_KEY")

local_llm = ChatOpenAI(
    base_url=MODEL1_URL,
    api_key=MODEL1_API_KEY,
    model=MODEL1_NAME,
    temperature=0.1,
)

alternative_local_llm = ChatOpenAI(
    base_url=MODEL1_URL,
    api_key=MODEL2_API_KEY,
    model=MODEL1_NAME,
    temperature=0.1,
)

AGENT_PERSONAS = {
    "analyst": "You are a meticulous fact-checker. Prioritize empirical data, sample sizes, and methodological rigor.",
    "contextualist": "You are a contextual analyst. Emphasize historical precedent, expert consensus, and real-world applicability.",
    "skeptic": "You are a critical skeptic. Actively seek disconfirming evidence, logical fallacies, and alternative explanations."
}
MAX_HISTORY=2

def merge_feedback_dicts(current: dict, update: dict) -> dict:
    """LangGraph reducer: merges feedback dicts but caps history to prevent token explosion."""
    merged = {k: list(v) for k, v in current.items()}

    for agent, feedback_list in update.items():
        if agent in merged:
            merged[agent].extend(feedback_list)
            # Keep only most recent feedback items
            merged[agent] = merged[agent][-MAX_HISTORY:]
        else:
            merged[agent] = feedback_list[-MAX_HISTORY:]

    return merged

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

class JustificationDraft(BaseModel):
    verdict: Literal["TRUE", "FALSE", "INCONCLUSIVE"] = Field(
        description="Must exactly match: TRUE, FALSE, or INCONCLUSIVE"
    )
    justification: str = Field(description="Step-by-step logical justification")
    context_used: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class Feedback(BaseModel):
    reviewer: str = Field(description="Name of the reviewing agent")
    pros: List[str] = Field(description="Strengths of the draft")
    cons: List[str] = Field(description="Weaknesses or logical gaps")
    suggestions: List[str] = Field(description="Actionable improvement steps")


class AgentScores(BaseModel):
    scores: dict[str, int] = Field(description="Mapping of agent_name -> score (1-10)")


# GraphState TypedDict (references the models above)
class SOMGraphState(TypedDict):
    citation_text: str
    paper_id: str
    initial_verdict: Literal["TRUE", "FALSE", "INCONCLUSIVE"]
    verdict: Literal["TRUE", "FALSE", "INCONCLUSIVE"]
    drafts: dict[str, JustificationDraft]
    feedback: Annotated[dict[str, List[Feedback]], merge_feedback_dicts]
    scores: dict[str, int]
    winner: Optional[str]
    iteration: int
    max_iterations: int
    usage_logs: Annotated[List[Dict[str, Any]], add]


class SOM_Justifier():
    def __init__(self, faiss, chunks, paper_id, semaphore):
        self.local_llm = local_llm
        self.alternative_llm = alternative_local_llm
        self.faiss_indexes = faiss
        self.chunk_stores = chunks
        self.paper_id = paper_id
        self.semaphore = semaphore

        # Build everyth        self._setup_models()ing in order
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

    def choose_LLM_model(self,agent_name: str):
        """Selects the model to use with tools"""
        if agent_name == "analyst" or agent_name == "contextualist":
            return self.llm_with_tools, "UTM"
        else:
            return self.alternative_llm_with_tools, "UTM2"

    """
    JUSTIFICATION TOOLS
    """

    def _setup_tools(self):
        """Define tools as bound methods"""

        # @tool
        def retrieve_context_impl(claim: str, number_of_chunks_to_return: Annotated[
            int, "Number of context chunks to retrieve. Valid range: 5-15. Absolute max: 15. Default: 10."] = 10) -> str:
            """Retrieve background information and related evidence for a citation.
            Requires the exact paper_id and citation_text to query the knowledge base."""
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
            """Get the official rubric for evaluating justifications and providing peer feedback."""
            return """
                EVALUATION RUBRIC:
                1. Completeness: The justification must be valid in full contextuality.
                2. Coherence: Ensure the faithfulness/consistency between the veracity prediction and justification.
                3. Interactivity: Put into consideration the users’ feedback - users will likely want to have a full explanation on why the verdict is TRUE/INCONCLUSIVE/FALSE.
                4. Actionability: Provide the user with the needed suggestions for modifying the claim to change it from INCONCLUSIVE/FALSE to TRUE. **VERY IMPORTANT**
                5. Novelty: Ensure the justification offers new information for the user to use to improve their citation if needed. 
                6. Impartial justification: The justification must use the provided context and not an AI's own knowledge.
                
                NEVER make a table - they are unreadable for the user!

                SCORING GUIDELINES (1-10):
                1-3: Major logical flaws, misused context, ignores counter-evidence.
                4-6: Acceptable but incomplete reasoning or weak citation support.
                7-8: Strong reasoning, well-supported, addresses most counter-points.
                9-10: Definitive, airtight logic, comprehensive citation use, explicitly calibrated verdict.
                
                NEVER refer to the evaluation rubric or scoring guideline in your draft.
                """

        # Store tools for later use
        self.tools = [self.retrieve_context_tool, get_evaluation_criteria]
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
        alt_llm_with_tools = self.alternative_llm.bind_tools(self.tools)
        self.alternative_llm_with_tools = alt_llm_with_tools | RunnableLambda(strip_think_tags)

    def _build_graph(self):
        """Construct the LangGraph workflow"""

        # Build workflow
        workflow = StateGraph(SOMGraphState)

        workflow.add_node("initial_drafts", self.initial_drafts)
        workflow.add_node("collect_feedback", self.collect_peer_feedback)
        # workflow.add_node("merge_feedback", self.merge_feedback)
        workflow.add_node("improve", self.improve_drafts)
        workflow.add_node("increment", self.increment_iteration)
        workflow.add_node("final_scoring", self.final_scoring)
        workflow.add_node("select_winner", self.select_winner)

        # Edges
        workflow.add_edge(START, "initial_drafts")
        workflow.add_edge("initial_drafts", "collect_feedback")
        workflow.add_edge("collect_feedback", "improve")
        workflow.add_edge("improve", "increment")
        workflow.add_conditional_edges("increment", self.should_continue, {
            "continue": "collect_feedback",
            "finalize": "final_scoring"
        })
        workflow.add_edge("final_scoring", "select_winner")
        workflow.add_edge("select_winner", END)

        self.app = workflow.compile()

    async def run_agent_with_tools(self, agent_name: str, prompt: str, prompt_name:str, structured_model: BaseModel) -> tuple[BaseModel, list]:
        """Run tool-calling loop with agent-specific LLM (async version)"""
        global MAX_STEPS
        messages = [{"role": "system", "content": prompt}]

        usage_entries = []
        count = 0
        while count < MAX_STEPS:
            async with self.semaphore:
                tools_llm, model_name = self.choose_LLM_model(agent_name)
                response = await tools_llm.ainvoke(messages)

            usage = extract_usage_from_langchain_response(response)
            entry = ""
            if usage:
                total_tokens, input_tokens, output_tokens = usage
                usage_entries.append({
                    "total_tokens": total_tokens,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "prompt_name": prompt_name,
                    "model": model_name
                })

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
            count +=1
            
        print(f"[WARNING] Agent {agent_name} hit max tool call steps ({max_agent_steps}). Forcing final response.")
        messages.append({"role": "user","content": "You have reached the maximum number of tool calls. Please output your final JSON response NOW based on the information you have gathered."})
        async with self.semaphore:
            tools_llm, _ = self.choose_LLM_model(agent_name)
            response = await tools_llm.ainvoke(messages)
        try:
            return structured_model.model_validate_json(response.content), usage_entries
        except Exception as e:
            raise Exception(
                f"Agent {agent_name} failed to produce a valid response after {max_agent_steps} tool calls. Error: {e}")

    async def initial_drafts(self, state: SOMGraphState) -> dict:
        new_drafts = {}
        all_usage = []
        tasks = []
        for name, persona in AGENT_PERSONAS.items():
            prompt = f"""{persona}
TASK: Create an initial justification. 
Call `retrieve_context` to retrieve evidence. Only retrieve context once.
Call `get_evaluation_criteria` to retrieve the evaluation rubric and score guidelines.
NEVER make a table in your draft.
Claim: {state['citation_text']}
Initial Verdict: {state['initial_verdict']}
Output ONLY valid JSON matching JustificationDraft schema."""
            tasks.append((name, self.run_agent_with_tools(name, prompt, "SOM_initial_drafts", JustificationDraft)))

        all_verdicts = []
        results = await asyncio.gather(*(task[1] for task in tasks))
        for (name, _), obj in zip(tasks, results):
            if isinstance(obj, Exception):
                print(f"[INITIAL DRAFT ERROR] {name} failed: {obj}")
                continue
            result, usage = obj
            new_drafts[name] = result
            all_verdicts.append(result.verdict)
            all_usage.extend(usage)
            self._fallback_usage_logs.extend(all_usage)

        counted = Counter(all_verdicts).most_common(3)
        if counted[0][1] == 1:
            verdict = state["initial_verdict"]
        else:
            verdict = counted[0][0]

        return {
            "verdict": verdict,
            "drafts": new_drafts,
            "feedback": {n: [] for n in AGENT_PERSONAS},
            "iteration": 1,
            "usage_logs": all_usage
        }

    async def collect_peer_feedback(self, state: SOMGraphState) -> dict:
        """Each agent reviews OTHER agents' drafts and returns feedback."""
        feedback_tasks = []
        all_usage = []
        for reviewer, persona in AGENT_PERSONAS.items():
            # Get drafts from agents OTHER than reviewer
            others = {n: d for n, d in state["drafts"].items() if n != reviewer}
            if not others:
                continue

            prompt = f"""{persona}
    TASK: Review drafts from other agents. Call `get_evaluation_criteria` if needed.
    CLAIM: {state['citation_text']} | Verdict: {state['verdict']}

    DRAFTS TO REVIEW:
    {chr(10).join(f'### {n} ###\nVerdict: {d.verdict}\nJustification: {d.justification}' for n, d in others.items())}

    Output ONLY valid JSON matching the Feedback schema."""

            feedback_tasks.append(
                (reviewer, self.run_agent_with_tools(reviewer, prompt, "SOM_collect_peer_feedback", Feedback))
            )

        # Run all feedback generations concurrently
        results = await asyncio.gather(*(t[1] for t in feedback_tasks), return_exceptions=True)

        # Route each reviewer's feedback to the agents they reviewed
        new_feedback = {name: [] for name in AGENT_PERSONAS}

        for (reviewer, _), result in zip(feedback_tasks, results):
            if isinstance(result, Exception):
                print(f"[FEEDBACK ERROR] {reviewer} failed: {result}")
                continue

            feedback, usage = result
            all_usage.extend(usage)
            self._fallback_usage_logs.extend(all_usage)
            feedback.reviewer = reviewer
            # Send this feedback to EVERY agent that was reviewed (i.e., all except reviewer)
            for target in AGENT_PERSONAS:
                if target != reviewer:
                    new_feedback[target].append(feedback)

        return {"feedback": new_feedback, "usage_logs": all_usage}

    async def improve_drafts(self, state: SOMGraphState) -> dict:
        """Step 3: Agents revise drafts using peer feedback"""
        improved = {}
        all_usage=[]
        tasks = []
        for name, persona in AGENT_PERSONAS.items():
            draft = state["drafts"][name]
            feedback = state["feedback"].get(name, [])
            if not feedback:
                improved[name] = draft
                continue

            feedback_text = chr(10).join(
                f"From {f.reviewer}: [Pros: {f.pros} | Cons: {f.cons} | Suggestions: {f.suggestions}]"
                for f in feedback
            )

            prompt = f"""{persona}
TASK: Revise your justification using peer feedback. Call `retrieve_context` if new evidence is needed.
NEVER make a table in your draft.
Your current justification draft:
Verdict: {draft.verdict}
Justification draft: {draft.justification}
Context used: {draft.context_used}
PEER FEEDBACK:
{feedback_text}
Output ONLY valid JSON matching JustificationDraft schema."""

            tasks.append((name, self.run_agent_with_tools(name, prompt,"SOM_improve_drafts", JustificationDraft)))

        results = await asyncio.gather(*(task[1] for task in tasks), return_exceptions=True)
        for (name, _), obj in zip(tasks, results):
            if isinstance(obj, Exception):
                print(f"[IMPROVE ERROR] {name} failed: {obj}")
                improved[name] = state["drafts"][name]  # Fallback to original
                continue

            result, usage = obj
            all_usage.extend(usage)
            self._fallback_usage_logs.extend(all_usage)
            improved[name] = result

        return {"drafts": improved,"usage_logs": all_usage}

    async def increment_iteration(self, state: SOMGraphState) -> dict:
        return {"iteration": state["iteration"] + 1}

    def should_continue(self, state: SOMGraphState) -> Literal["continue", "finalize"]:
        return "continue" if state["iteration"] < state["max_iterations"] else "finalize"

    async def final_scoring(self, state: SOMGraphState) -> dict:
        """Step 4: Each agent scores all final drafts"""
        all_scores = {n: [] for n in AGENT_PERSONAS}
        all_usage=[]
        scoring_tasks = []
        for scorer, persona in AGENT_PERSONAS.items():
            prompt = f"""{persona}
TASK: Score each final draft 1-10 using the evaluation rubric.
DRAFTS:
{chr(10).join(f'### {n} ###\nVerdict: {d.verdict}\nJustification: {d.justification[:200]}...' for n, d in state["drafts"].items())}
Output ONLY valid JSON matching AgentScores schema."""
            scoring_tasks.append((scorer, self.run_agent_with_tools(scorer, prompt,"SOM_final_scoring", AgentScores)))

        results = await asyncio.gather(*(task[1] for task in scoring_tasks), return_exceptions=True)

        for (scorer, _), result in zip(scoring_tasks, results):
            if isinstance(result, Exception):
                print(f"[SCORING ERROR] {scorer} failed: {result}")
                continue

            scores_obj, usage = result
            all_usage.extend(usage)
            self._fallback_usage_logs.extend(all_usage)
            for n, s in scores_obj.scores.items():
                if n in all_scores:
                    all_scores[n].append(s)

        averaged = {n: sum(s) // len(s) if s else 0 for n, s in all_scores.items()}
        return {"scores": averaged,"usage_logs": all_usage}

    async def select_winner(self, state: SOMGraphState) -> dict:
        """Step 5: Return highest-scoring draft"""
        if not state["scores"]:
            return {"winner": None}
        winner = max(state["scores"], key=state["scores"].get)
        return {"winner": winner}

    async def verify_citation(self, claim: str,
                              initial_verdict: Literal["TRUE", "FALSE", "INCONCLUSIVE"],
                              max_iterations: int = 2) -> dict:
        self._fallback_usage_logs = []

        try:
            result = await self.app.ainvoke({
                "citation_text": claim,
                "initial_verdict": initial_verdict,
                "drafts": {},
                "feedback": {},
                "scores": {},
                "winner": None,
                "iteration": 0,
                "max_iterations": max_iterations
            })
            usage_logs = result.get("usage_logs", [])
        except Exception as e:
            print(f"[SOM CRASHED] {e}]")
            usage_logs = self._fallback_usage_logs
            result = {
            "verdict": initial_verdict,
            "winner": "Error",
            "drafts": {"Error": "Error occurred!"},
            "scores": {},
            "all_final_drafts": {},
            "iterations_completed": -1
        }

        usage_logs = result.get("usage_logs", [])
        if usage_logs:
            for entry in usage_logs:
                log_entry(entry)

        best_draft = None
        if result.get("winner") and result.get("drafts"):
            best_draft = result["drafts"].get(result["winner"])

        return {
            "verdict": result.get("verdict", initial_verdict),
            "winning_agent": result.get("winner", "Error"),
            "justification": best_draft.justification,
            "all_scores": result.get("scores", {}),
            "all_final_drafts": result.get("drafts",{}),
            "iterations_completed": result.get("iteration", 3)
        }
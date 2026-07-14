"""
Experiment_2/react.py
LangGraph ReAct Agent — Automated research experimental plan generation and iterative optimization
Pure LangGraph low-level API implementation, no langchain_core dependency
Generator Model: DeepSeek V4 Pro  |  Judge Models: DeepSeek-chat + GPT-5.4
"""

import json
import os
import re
import time
import random
import uuid
from typing import TypedDict, List, Dict, Any, Optional, Literal
from openai import OpenAI
import httpx

# =========================================================
# LangGraph Core Imports (StateGraph + END only, no langchain_core trigger)
# =========================================================
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# =========================================================
# Proxy Settings
# =========================================================
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

# =========================================================
# API Configuration
# =========================================================
DEEPSEEK_API_KEY = "enter-your-api-key"
GPT_API_KEY = "enter-your-api-key"

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    timeout=httpx.Timeout(300.0)
)

gpt_client = OpenAI(
    api_key=GPT_API_KEY,
    base_url="https://www.autodl.art/api/v1",
    timeout=httpx.Timeout(300.0)
)

# ---- Model Assignment ----
GENERATOR_MODEL = "deepseek-v4-pro"      # DeepSeek V4 Pro (generator model)
JUDGE_MODEL_DS   = "deepseek-chat"     # DeepSeek judge
JUDGE_MODEL_GPT  = "gpt-5.4"           # GPT judge

# =========================================================
# Dataset Path
# =========================================================
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gen_lcot", "data", "train.jsonl"
)

# =========================================================
# Prompt Templates
# =========================================================

SYSTEM_PROMPT = """You are an expert AI research scientist with access to specialized tools.
Your job is to help users design rigorous, innovative experimental plans for their research questions.

Available tools:
1. search_dataset(query) — Search prior research case studies for reference
2. generate_plan(topic, references) — Generate a complete experimental plan
3. score_plan(plan_text) — Evaluate plan reliability (0-100) and innovation (0-100) using multiple judges
4. optimize_plan(plan_text, scores_json) — Improve the plan based on score feedback

Recommended Workflow:
1. Search the dataset for relevant case studies
2. Generate an initial experimental plan
3. Score the plan with multiple judges
4. If scores are low (reliability < 70 or innovation < 60), optimize the plan
5. Present the final plan and scores to the user

Always think step by step. Explain what you're doing before calling a tool."""

SCORING_PROMPT = """You are an expert scientific reviewer.

Please evaluate the given research methodology from the following two aspects:

1. Method Reliability (0-100)
2. Method Innovation (0-100)

Scoring Criteria:
Method Reliability: technical correctness, experimental rigor, feasibility, logical consistency, reproducibility
Method Innovation: originality, novelty, creativity, research contribution, uniqueness of methodology

Your output format MUST strictly follow:
Reliability: xx
Innovation: xx

Only output the scores. Do not provide explanations."""

OPTIMIZE_PROMPT = """You are an expert research mentor. Analyze the weaknesses of the following experimental plan
and generate a significantly improved version.

ORIGINAL PLAN:
----------------
{plan}
----------------

CURRENT SCORES: Reliability={reliability}, Innovation={innovation}

IMPROVEMENT FOCUS:
- If reliability is low: strengthen experimental rigor, add validation steps, improve reproducibility
- If innovation is low: incorporate novel methodologies, cross-disciplinary approaches, creative evaluation metrics
- Ensure the plan is complete: objective -> methods -> experiments -> datasets -> metrics -> limitations

Generate the improved plan directly. Make it comprehensive and academically professional."""

GENERATION_PROMPT = """You are a professional AI research scientist. Based on the user's research objective,
generate a complete, rigorous, and executable experimental plan.

Requirements:
1. Clearly describe the research objective
2. Design the model or methodology
3. Provide detailed experimental procedures
4. Include datasets and evaluation metrics
5. Analyze possible challenges and limitations
6. Maintain rigorous scientific reasoning
7. Encourage methodological innovation
8. Ensure the plan is practical and reproducible

Your response should be well-structured and academically professional."""


# =========================================================
# Data Loading
# =========================================================

def load_dataset(path: str = DATASET_PATH, sample_size: int = 50) -> List[Dict]:
    """Load and randomly sample dataset"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if sample_size and sample_size < len(data):
        random.seed(42)
        data = random.sample(data, sample_size)

    return data


# =========================================================
# Text Extraction Tools
# =========================================================

def extract_user_content(sample: Dict) -> str:
    """Extract user question from sample"""
    for msg in sample.get("messages", []):
        if msg["role"] == "user":
            return msg["content"]
    return ""


def extract_assistant_content(sample: Dict) -> str:
    """Extract assistant answer from sample"""
    for msg in sample.get("messages", []):
        if msg["role"] == "assistant":
            return msg["content"]
    return ""


def parse_score(text: Optional[str]) -> Dict[str, int]:
    """Parse Reliability and Innovation scores from scoring text"""
    if text is None:
        return {"reliability": 0, "innovation": 0}
    try:
        rel_match = re.search(r"Reliability\s*:\s*(\d+)", text, re.IGNORECASE)
        inn_match = re.search(r"Innovation\s*:\s*(\d+)", text, re.IGNORECASE)
        return {
            "reliability": int(rel_match.group(1)) if rel_match else 0,
            "innovation": int(inn_match.group(1)) if inn_match else 0,
        }
    except Exception:
        return {"reliability": 0, "innovation": 0}


# =========================================================
# API Call (with retry)
# =========================================================

def call_llm(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_input: str,
    temperature: float = 0.1,
    max_retries: int = 5,
) -> str:
    """Call LLM API with exponential backoff retry"""
    sleep_time = 1
    for attempt in range(1, max_retries + 1):
        try:
            print(f"    [API] {model} attempt {attempt}...")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=temperature,
                timeout=120,
            )
            print(f"    [API] {model} OK (attempt {attempt})")
            return response.choices[0].message.content
        except Exception as e:
            print(f"    [API] error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(sleep_time)
                sleep_time = min(sleep_time * 2, 60)
    return "[ERROR] All API retries failed"


def call_llm_with_tools(
    client: OpenAI,
    model: str,
    messages: List[Dict],
    tools: List[Dict],
    temperature: float = 0.1,
    max_retries: int = 3,
) -> dict:
    """
    Call LLM API (supports function calling), returns full response object
    Returns: {"content": str, "tool_calls": list or None}
    """
    sleep_time = 1
    for attempt in range(1, max_retries + 1):
        try:
            print(f"    [API] {model} attempt {attempt} (function calling)...")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else "none",
                temperature=temperature,
                timeout=180,
            )
            print(f"    [API] {model} OK (attempt {attempt})")
            choice = response.choices[0]
            # Preserve API native format to ensure type/function nesting intact when passing back
            raw_tool_calls = []
            for tc in (choice.message.tool_calls or []):
                raw_tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,  # keep JSON string
                    },
                })
            return {
                "content": choice.message.content or "",
                "tool_calls": raw_tool_calls if raw_tool_calls else None,
            }
        except Exception as e:
            print(f"    [API] error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(sleep_time)
                sleep_time = min(sleep_time * 2, 60)
    return {"content": "[ERROR] All API retries failed", "tool_calls": None}


# =========================================================
# Tool Implementation
# =========================================================

# Global cache: load dataset once
_dataset_cache: Optional[List[Dict]] = None


def get_dataset() -> List[Dict]:
    global _dataset_cache
    if _dataset_cache is None:
        print("  [Data] Loading train.jsonl dataset...")
        _dataset_cache = load_dataset()
        print(f"  [Data] Loaded {len(_dataset_cache)} samples")
    return _dataset_cache


# ---- Tool Metadata (for Agent function calling) ----
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_dataset",
            "description": "Search the existing research case dataset for the most relevant cases matching the query. When to call: after user proposes a research question, use this tool first to find similar cases as reference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords or research question description",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_plan",
            "description": "Generate a complete, rigorous, and executable experimental plan based on a research topic. When to call: after understanding user requirements and completing case retrieval, generate the initial experimental plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "User's research question or goal description",
                    },
                    "references": {
                        "type": "string",
                        "description": "(Optional) Related case JSON returned by search_dataset, as reference. Pass empty string if no reference available.",
                        "default": "",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_plan",
            "description": "Use DeepSeek-chat + GPT-5.4 dual-judge to score an experimental plan on two dimensions. (1) Method Reliability 0-100 (2) Method Innovation 0-100. When to call: after generating an experimental plan, evaluate its quality.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_text": {
                        "type": "string",
                        "description": "Full text of the experimental plan to score",
                    }
                },
                "required": ["plan_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "optimize_plan",
            "description": "Based on score feedback, iteratively improve the experimental plan. If reliability < 70 or innovation < 60, strongly recommend calling this tool. When to call: when score_plan returns NEEDS_IMPROVEMENT.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_text": {
                        "type": "string",
                        "description": "Current full text of the experimental plan",
                    },
                    "scores_json": {
                        "type": "string",
                        "description": "JSON score string returned by score_plan",
                    },
                },
                "required": ["plan_text", "scores_json"],
            },
        },
    },
]


# ---- Tool Execution Functions ----

def execute_search_dataset(query: str) -> str:
    """Search relevant cases in dataset"""
    dataset = get_dataset()
    query_lower = query.lower()
    keywords = query_lower.split()
    scored = []

    for i, sample in enumerate(dataset):
        user_content = extract_user_content(sample).lower()
        assistant_content = extract_assistant_content(sample).lower()
        score = 0
        for kw in keywords:
            score += user_content.count(kw) * 2
            score += assistant_content.count(kw)
        if score > 0:
            scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, idx in scored[:5]:
        sample = dataset[idx]
        user_text = extract_user_content(sample)
        assistant_text = extract_assistant_content(sample)
        if len(assistant_text) > 800:
            assistant_text = assistant_text[:800] + "..."
        results.append({
            "index": idx,
            "relevance_score": score,
            "user_question": user_text[:500],
            "method_summary": assistant_text,
        })

    if not results:
        return "No highly relevant cases found. Suggest generating plan directly based on general methodology."

    return json.dumps(results, ensure_ascii=False, indent=2)


def execute_generate_plan(topic: str, references: str = "") -> str:
    """Generate experimental plan -> uses GENERATOR_MODEL (DeepSeek V4 Pro)"""
    user_input = f"""Research Objective:
{topic}

Reference Cases:
{references if references else "No specific reference cases, please generate plan based on general best practices."}

Please generate a complete experimental plan based on the above."""
    return call_llm(deepseek_client, GENERATOR_MODEL, GENERATION_PROMPT, user_input)


def execute_score_plan(plan_text: str) -> str:
    """
    Dual-judge scoring:
      - Judge 1: DeepSeek-chat (JUDGE_MODEL_DS)
      - Judge 2: GPT-5.4     (JUDGE_MODEL_GPT)
    Average the two as combined
    """
    print("\n  [Scoring] Judge 1 — DeepSeek-chat ...")
    ds_score_text = call_llm(deepseek_client, JUDGE_MODEL_DS, SCORING_PROMPT, plan_text)
    ds_scores = parse_score(ds_score_text)
    print(f"    DeepSeek -> Reliability={ds_scores['reliability']}, Innovation={ds_scores['innovation']}")

    print("  [Scoring] Judge 2 — GPT-5.4 ...")
    gpt_score_text = call_llm(gpt_client, JUDGE_MODEL_GPT, SCORING_PROMPT, plan_text)
    gpt_scores = parse_score(gpt_score_text)
    print(f"    GPT      -> Reliability={gpt_scores['reliability']}, Innovation={gpt_scores['innovation']}")

    combined = {
        "reliability": round((ds_scores["reliability"] + gpt_scores["reliability"]) / 2, 1),
        "innovation": round((ds_scores["innovation"] + gpt_scores["innovation"]) / 2, 1),
    }

    result = {
        "deepseek_scores": ds_scores,
        "gpt_scores": gpt_scores,
        "combined_scores": combined,
        "verdict": (
            "EXCELLENT" if combined["reliability"] >= 80 and combined["innovation"] >= 75
            else "GOOD" if combined["reliability"] >= 70 and combined["innovation"] >= 60
            else "NEEDS_IMPROVEMENT"
        ),
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


def execute_optimize_plan(plan_text: str, scores_json: str) -> str:
    """Optimize plan based on score feedback -> uses GENERATOR_MODEL (DeepSeek V4 Pro)"""
    scores = json.loads(scores_json)
    combined = scores.get("combined_scores", {})
    reliability = combined.get("reliability", 0)
    innovation = combined.get("innovation", 0)

    optimize_input = OPTIMIZE_PROMPT.format(
        plan=plan_text,
        reliability=reliability,
        innovation=innovation,
    )

    return call_llm(deepseek_client, GENERATOR_MODEL, SYSTEM_PROMPT, optimize_input, temperature=0.3)


# Tool name -> execution function mapping
TOOL_EXECUTORS = {
    "search_dataset": lambda args: execute_search_dataset(**args),
    "generate_plan":  lambda args: execute_generate_plan(**args),
    "score_plan":     lambda args: execute_score_plan(**args),
    "optimize_plan":  lambda args: execute_optimize_plan(**args),
}


# =========================================================
# LangGraph State Definition
# =========================================================

class AgentState(TypedDict):
    """ReAct Agent state (pure dict, no langchain_core dependency)"""
    messages: List[Dict]          # message history
    plan: str                     # current experimental plan
    plan_history: List[Dict]      # plan iteration history [{round, plan, scores, verdict}]
    final_report: str             # final output report
    iteration: int                # current iteration count
    research_topic: str           # user's original question
    # ---- Round Score Tracking ----
    score_round: int              # current scoring round number (0-based)
    round_scores: List[Dict]      # [{round:1, ds:{}, gpt:{}, combined:{}}]


# =========================================================
# Build LangGraph ReAct Graph
# =========================================================

def build_react_graph() -> StateGraph:
    """
    Build LangGraph state graph for ReAct Agent.

    Graph structure:
        __start__ -> agent -> [route]
                           ├─ has tool_calls -> tools -> agent
                           └─ no tool_calls -> finalize -> END
    """

    # =====================
    # Agent Node
    # =====================
    def agent_node(state: AgentState) -> Dict:
        iteration = state.get("iteration", 0) + 1
        print(f"\n{'='*60}")
        print(f"[AGENT] Reasoning round #{iteration}")
        print(f"{'='*60}")

        messages = list(state.get("messages", []))
        research_topic = state.get("research_topic", "")

        # First round: inject system prompt + user question
        if iteration == 1:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"I need to design a complete experimental plan for the following research:\n\n{research_topic}"},
            ]

        # Exceeded max rounds, force finish
        if iteration >= 7:
            messages.append({
                "role": "user",
                "content": "You have completed sufficient analysis and iterations. Now please synthesize all results, provide the final experimental plan and score summary. Do not call any more tools. Output the final plan directly."
            })
            result = call_llm_with_tools(deepseek_client, GENERATOR_MODEL, messages, None, temperature=0.1)
            messages.append({"role": "assistant", "content": result["content"]})
            return {
                "messages": messages,
                "plan": result["content"] if len(result["content"]) > 500 else state.get("plan", ""),
                "iteration": iteration,
            }

        # Call LLM for reasoning (with tools)
        result = call_llm_with_tools(
            deepseek_client, GENERATOR_MODEL, messages, TOOL_DEFINITIONS, temperature=0.1
        )

        # Build assistant message and accumulate to message history
        assistant_msg = {"role": "assistant", "content": result["content"]}
        if result["tool_calls"]:
            assistant_msg["tool_calls"] = result["tool_calls"]
            print(f"  -> Agent decided to call {len(result['tool_calls'])} tool(s):")
            for tc in result["tool_calls"]:
                func_info = tc.get("function", {})
                func_name = func_info.get("name", "?")
                func_args = func_info.get("arguments", "{}")
                print(f"     • {func_name}({func_args[:120]})")
        messages.append(assistant_msg)

        # Update plan
        new_plan = state.get("plan", "")
        if result["content"] and len(result["content"]) > 500:
            new_plan = result["content"]

        return {
            "messages": messages,
            "plan": new_plan,
            "iteration": iteration,
        }

    # =====================
    # Tools Node
    # =====================
    def tools_node(state: AgentState) -> Dict:
        messages = list(state.get("messages", []))
        if not messages:
            return {"messages": messages}

        last_msg = messages[-1]
        tool_calls = last_msg.get("tool_calls", []) if last_msg["role"] == "assistant" else []

        if not tool_calls:
            return {"messages": messages}

        print(f"\n{'─'*60}")
        print(f"[TOOLS] Executing {len(tool_calls)} tool call(s)")
        print(f"{'─'*60}")

        plan_history = list(state.get("plan_history", []))
        round_scores = list(state.get("round_scores", []))
        score_round = state.get("score_round", 0)

        for tc in tool_calls:
            func_info = tc.get("function", {})
            func_name = func_info.get("name", tc.get("name", ""))
            func_args_str = func_info.get("arguments", "{}")
            try:
                func_args = json.loads(func_args_str) if isinstance(func_args_str, str) else func_args_str
            except json.JSONDecodeError:
                func_args = {}
            call_id = tc.get("id", str(uuid.uuid4()))

            executor = TOOL_EXECUTORS.get(func_name)
            if executor:
                print(f"  -> Executing {func_name}...")
                try:
                    result_content = executor(func_args)
                    if len(result_content) > 3000:
                        result_content = result_content[:3000] + "\n...(truncated)"
                    print(f"  -> {func_name} complete ({len(result_content)} chars)")
                except Exception as e:
                    result_content = f"Tool execution error: {e}"
                    print(f"  -> {func_name} failed: {e}")
            else:
                result_content = f"Unknown tool: {func_name}"
                print(f"  -> Unknown tool: {func_name}")

            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": func_name,
                "content": result_content,
            })

            # ---- Score Tracking ----
            if func_name == "score_plan":
                score_round += 1
                try:
                    score_data = json.loads(result_content)
                    combined = score_data.get("combined_scores", {})
                    verdict = score_data.get("verdict", "")

                    # Append to round_scores (for batch averaging)
                    round_scores.append({
                        "round": score_round,
                        "ds": score_data.get("deepseek_scores", {}),
                        "gpt": score_data.get("gpt_scores", {}),
                        "combined": combined,
                    })

                    # Append to plan_history
                    plan_history.append({
                        "round": score_round,
                        "plan": state.get("plan", "")[:500],
                        "scores": combined,
                        "verdict": verdict,
                    })

                    print(f"  [Score Summary R{score_round}] "
                          f"DS(R={combined.get('reliability','?')} I={combined.get('innovation','?')}) "
                          f"GPT(R={score_data.get('gpt_scores',{}).get('reliability','?')} "
                          f"I={score_data.get('gpt_scores',{}).get('innovation','?')}) "
                          f"→ {verdict}")
                except json.JSONDecodeError:
                    pass

        return {
            "messages": messages,
            "plan_history": plan_history,
            "score_round": score_round,
            "round_scores": round_scores,
        }

    # =====================
    # Finalize Node
    # =====================
    def finalize_node(state: AgentState) -> Dict:
        print(f"\n{'='*60}")
        print("[FINALIZE] Generating final report")
        print(f"{'='*60}")

        plan = state.get("plan", "")
        plan_history = state.get("plan_history", [])
        messages = state.get("messages", [])
        research_topic = state.get("research_topic", "")
        iteration = state.get("iteration", 0)
        round_scores = state.get("round_scores", [])

        # Extract final AI response from messages
        final_response = ""
        for m in reversed(messages):
            if m["role"] == "assistant" and m.get("content"):
                final_response = m["content"]
                break

        display_plan = plan if plan else final_response
        if len(display_plan) > 5000:
            display_plan = display_plan[:5000] + "\n\n...(truncated)"

        report_lines = [
            "=" * 70,
            "                      Final Experimental Plan Report",
            "=" * 70,
            "",
            f"Research Topic: {research_topic[:300]}",
            "",
            "-" * 70,
            "Experimental Plan",
            "-" * 70,
            display_plan or "Failed to generate complete plan",
        ]

        # Score History
        if round_scores:
            report_lines += ["", "-" * 70, "Score History", "-" * 70]
            for rs in round_scores:
                r = rs["round"]
                c = rs["combined"]
                ds = rs["ds"]
                gpt = rs["gpt"]
                report_lines.append(
                    f"  Round #{r}: DS(R={ds.get('reliability',0)} I={ds.get('innovation',0)}) "
                    f"GPT(R={gpt.get('reliability',0)} I={gpt.get('innovation',0)}) "
                    f"→ Combined(R={c.get('reliability',0)} I={c.get('innovation',0)})"
                )

            # Per-judge average
            if len(round_scores) >= 1:
                ds_avg_rel = round(sum(rs["ds"].get("reliability", 0) for rs in round_scores) / len(round_scores), 1)
                ds_avg_inn = round(sum(rs["ds"].get("innovation", 0) for rs in round_scores) / len(round_scores), 1)
                gpt_avg_rel = round(sum(rs["gpt"].get("reliability", 0) for rs in round_scores) / len(round_scores), 1)
                gpt_avg_inn = round(sum(rs["gpt"].get("innovation", 0) for rs in round_scores) / len(round_scores), 1)
                report_lines += [
                    "",
                    f"  This sample judge average -> DeepSeek: R={ds_avg_rel} I={ds_avg_inn}  |  GPT: R={gpt_avg_rel} I={gpt_avg_inn}",
                ]

        report_lines += [
            "",
            "-" * 70,
            "Iteration Statistics",
            "-" * 70,
            f"* Total reasoning rounds: {iteration}",
            f"* Score count:   {len(round_scores)}",
            f"* Plan optimizations:   {sum(1 for ph in plan_history if ph.get('round', 1) > 1)} time(s)",
            "",
            "=" * 70,
        ]

        final_report = "\n".join(report_lines)
        return {"final_report": final_report}

    # =====================
    # Routing Condition
    # =====================
    def route_after_agent(state: AgentState) -> Literal["tools", "finalize"]:
        messages = state.get("messages", [])
        iteration = state.get("iteration", 0)

        if iteration >= 7:
            print("  -> Reached max iterations, ending")
            return "finalize"

        if messages:
            last = messages[-1]
            if last["role"] == "assistant" and last.get("tool_calls"):
                return "tools"

        print("  -> Agent finished reasoning, generating final report")
        return "finalize"

    # =====================
    # Assemble Graph
    # =====================
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "finalize": "finalize"},
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("finalize", END)

    return workflow


# =========================================================
# Compile & Run
# =========================================================

def compile_agent():
    """Compile ReAct Agent graph"""
    workflow = build_react_graph()
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def run_agent(question: str, verbose: bool = True) -> Dict[str, Any]:
    """Run ReAct Agent to generate experimental plan for given research question.
    Returns: {final_report, plan, plan_history, round_scores, iteration}
    """
    app = compile_agent()

    initial_state: AgentState = {
        "messages": [],
        "plan": "",
        "plan_history": [],
        "final_report": "",
        "iteration": 0,
        "research_topic": question,
        "score_round": 0,
        "round_scores": [],
    }

    config = {"configurable": {"thread_id": f"research-{int(time.time())}"}}

    print(f"\n{'★'*70}")
    print(f"Research Question: {question[:200]}")
    print(f"{'★'*70}")

    final_state = None
    for event in app.stream(initial_state, config, stream_mode="values"):
        final_state = event
        if verbose:
            msgs = event.get("messages", [])
            for m in msgs:
                role = m.get("role", "?")
                content = m.get("content", "")
                if role == "assistant" and content and len(content) > 30:
                    print(f"\n  [{role.upper()} Output] {content[:200]}...")
                elif role == "tool" and content:
                    print(f"  [{role.upper()} Return] {content[:150]}...")

    if final_state is None:
        return {"final_report": "[ERROR] Agent produced no output", "plan": "", "plan_history": [], "round_scores": [], "iteration": 0}

    return {
        "final_report": final_state.get("final_report", ""),
        "plan": final_state.get("plan", ""),
        "plan_history": final_state.get("plan_history", []),
        "round_scores": final_state.get("round_scores", []),
        "iteration": final_state.get("iteration", 0),
    }


# =========================================================
# Tool: Compute Cross-Sample Averages
# =========================================================

def _avg(score_list: List[float]) -> float:
    """Safely compute average"""
    return round(sum(score_list) / len(score_list), 1) if score_list else 0.0


def compute_batch_averages(all_results: List[Dict]) -> Dict:
    """
    Input: List of run_agent results for all samples
    Output: Cross-sample averages of per-round ReAct scores (Round 1 initial plan / Round 2+ optimized plan)

    Return format:
    {
      "rounds": {
        1: {"count": N, "ds_rel": ..., "ds_inn": ..., "gpt_rel": ..., "gpt_inn": ..., "combined_rel": ..., "combined_inn": ...},
        2: {...},
      },
      "overall": {"ds_rel": ..., "ds_inn": ..., "gpt_rel": ..., "gpt_inn": ..., "combined_rel": ..., "combined_inn": ...}
    }
    """
    # Collect by round
    by_round: Dict[int, Dict[str, List[float]]] = {}

    for result in all_results:
        for rs in result.get("round_scores", []):
            r = rs.get("round", 0)
            if r not in by_round:
                by_round[r] = {
                    "ds_rel": [], "ds_inn": [],
                    "gpt_rel": [], "gpt_inn": [],
                    "combined_rel": [], "combined_inn": [],
                }
            by_round[r]["ds_rel"].append(rs.get("ds", {}).get("reliability", 0))
            by_round[r]["ds_inn"].append(rs.get("ds", {}).get("innovation", 0))
            by_round[r]["gpt_rel"].append(rs.get("gpt", {}).get("reliability", 0))
            by_round[r]["gpt_inn"].append(rs.get("gpt", {}).get("innovation", 0))
            by_round[r]["combined_rel"].append(rs.get("combined", {}).get("reliability", 0))
            by_round[r]["combined_inn"].append(rs.get("combined", {}).get("innovation", 0))

    rounds_summary = {}
    all_ds_rel, all_ds_inn = [], []
    all_gpt_rel, all_gpt_inn = [], []
    all_comb_rel, all_comb_inn = [], []

    for r in sorted(by_round.keys()):
        d = by_round[r]
        rounds_summary[r] = {
            "count": len(d["ds_rel"]),
            "ds_reliability_avg": _avg(d["ds_rel"]),
            "ds_innovation_avg": _avg(d["ds_inn"]),
            "gpt_reliability_avg": _avg(d["gpt_rel"]),
            "gpt_innovation_avg": _avg(d["gpt_inn"]),
            "combined_reliability_avg": _avg(d["combined_rel"]),
            "combined_innovation_avg": _avg(d["combined_inn"]),
        }
        all_ds_rel.extend(d["ds_rel"]); all_ds_inn.extend(d["ds_inn"])
        all_gpt_rel.extend(d["gpt_rel"]); all_gpt_inn.extend(d["gpt_inn"])
        all_comb_rel.extend(d["combined_rel"]); all_comb_inn.extend(d["combined_inn"])

    return {
        "rounds": rounds_summary,
        "overall": {
            "total_scores": len(all_comb_rel),
            "ds_reliability_avg": _avg(all_ds_rel),
            "ds_innovation_avg": _avg(all_ds_inn),
            "gpt_reliability_avg": _avg(all_gpt_rel),
            "gpt_innovation_avg": _avg(all_gpt_inn),
            "combined_reliability_avg": _avg(all_comb_rel),
            "combined_innovation_avg": _avg(all_comb_inn),
        },
    }


# =========================================================
# Interactive CLI
# =========================================================

def interactive_loop():
    """Interactive Command Line Interface"""
    print(r"""
╔══════════════════════════════════════════════════════════════╗
║      LangGraph ReAct Agent — Research Experiment Plan Generator ║
║      Generator: DeepSeek V4 Pro | Judges: DS-chat + GPT-5.4 ║
╚══════════════════════════════════════════════════════════════╝

Commands:
  /run <question> — Generate experimental plan for research question
  /batch [N]      — Batch test (use first N items from dataset, default 3)
  /demo           — Use built-in demo
  /quit           — Exit

Example:
  /run Design an experiment for protein-ligand binding prediction using GNNs
""")

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye!")
            break

        if user_input.lower() == "/demo":
            user_input = "/run Design an experiment to evaluate whether large language models can self-improve through recursive self-critique without external feedback"

        if user_input.lower().startswith("/run "):
            question = user_input[5:].strip()
            result = run_agent(question)

            print("\n" + result["final_report"])

            output_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(output_dir, f"result_{int(time.time())}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n[Result saved] {output_path}")

        elif user_input.lower().startswith("/batch"):
            parts = user_input.split()
            n = int(parts[1]) if len(parts) > 1 else 3
            batch_test(n)
        else:
            print("Unknown command. Use /run <question> to start.")


def batch_test(n: int = 3):
    """Batch test: use first N items from dataset, output cross-sample averages for per-round ReAct scores"""
    dataset = load_dataset(sample_size=n)
    print(f"\n{'#'*70}")
    print(f"  Batch Testing Mode: {n} samples")
    print(f"  Generator Model: {GENERATOR_MODEL}")
    print(f"  Judge Models: {JUDGE_MODEL_DS} + {JUDGE_MODEL_GPT}")
    print(f"{'#'*70}")

    all_results = []
    for i, sample in enumerate(dataset):
        user_q = extract_user_content(sample)
        req_match = re.search(r"<user_request>\s*(.*?)(?:<think>|$)", user_q, re.DOTALL)
        question = req_match.group(1).strip()[:500] if req_match else user_q[:500]

        print(f"\n{'#'*60}")
        print(f"  Sample {i+1}/{n}: {question[:100]}...")
        print(f"{'#'*60}")

        result = run_agent(question, verbose=False)
        all_results.append({
            "sample_index": i,
            "question": question,
            "report": result["final_report"],
            "round_scores": result.get("round_scores", []),
        })

        # After each sample, print its round scores
        rs = result.get("round_scores", [])
        if rs:
            for r in rs:
                c = r["combined"]
                print(f"  -> Sample {i+1} R{r['round']}: DS(R={r['ds'].get('reliability',0)} I={r['ds'].get('innovation',0)}) "
                      f"GPT(R={r['gpt'].get('reliability',0)} I={r['gpt'].get('innovation',0)}) "
                      f"Combined(R={c.get('reliability',0)} I={c.get('innovation',0)})")

    # ============ Cross-Sample Averages ============
    batch_avg = compute_batch_averages(all_results)

    print("\n")
    print("=" * 70)
    print("           Batch Test — Cross-Sample Average Summary")
    print("=" * 70)

    for r in sorted(batch_avg["rounds"].keys()):
        info = batch_avg["rounds"][r]
        label = f"Round {r} (Initial Plan)" if r == 1 else f"Round {r} (Optimized Plan)"
        print(f"\n  [{label}]  (Covering {info['count']}/{n} samples)")
        print(f"    DeepSeek Judges:  Reliability avg = {info['ds_reliability_avg']}  |  Innovation avg = {info['ds_innovation_avg']}")
        print(f"    GPT Judges:      Reliability avg = {info['gpt_reliability_avg']}  |  Innovation avg = {info['gpt_innovation_avg']}")
        print(f"    * Combined avg:    Reliability avg = {info['combined_reliability_avg']}  |  Innovation avg = {info['combined_innovation_avg']}")

    overall = batch_avg["overall"]
    print(f"\n  {'─'*60}")
    print(f"  [Overall Total]  ({overall['total_scores']} score(s) total)")
    print(f"    DeepSeek Judges:  R = {overall['ds_reliability_avg']}  |  I = {overall['ds_innovation_avg']}")
    print(f"    GPT Judges:      R = {overall['gpt_reliability_avg']}  |  I = {overall['gpt_innovation_avg']}")
    print(f"    * Combined avg:    R = {overall['combined_reliability_avg']}  |  I = {overall['combined_innovation_avg']}")
    print(f"\n{'='*70}")

    # Save
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, f"batch_results_{int(time.time())}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "generator_model": GENERATOR_MODEL,
                "judge_models": [JUDGE_MODEL_DS, JUDGE_MODEL_GPT],
                "sample_count": n,
            },
            "batch_averages": batch_avg,
            "samples": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[Result saved] {output_path}")


# =========================================================
# Main Entry Point
# =========================================================

if __name__ == "__main__":
    interactive_loop()

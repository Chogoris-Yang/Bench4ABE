"""
Experiment_2/self_discover.py
Self-Discover Experimental Plan Generator — Zhou et al. 2024
LLM first self-discovers the optimal reasoning structure (metacognition) -> generates plan according to that structure -> scores -> revises structure and regenerates
Generator: DeepSeek V4 Pro  |  Judges: DeepSeek-chat + GPT-5.4

Self-Discover vs Plan-and-Solve:
  Plan-and-Solve uses manually preset section structures; Self-Discover lets the LLM discover what structure to use
  Self-Discover's metacognition stage is its signature: "first learn how to think, then think"
"""

import json
import os
import re
import time
import random
from typing import TypedDict, List, Dict, Any, Optional, Literal
from openai import OpenAI
import httpx

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# =========================================================
# Proxy & API
# =========================================================
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

DEEPSEEK_API_KEY = "enter-your-api-key"
GPT_API_KEY = "enter-your-api-key"

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com",
    timeout=httpx.Timeout(300.0)
)
gpt_client = OpenAI(
    api_key=GPT_API_KEY, base_url="https://www.autodl.art/api/v1",
    timeout=httpx.Timeout(300.0)
)

GENERATOR_MODEL = "deepseek-v4-pro"
JUDGE_MODEL_DS   = "deepseek-chat"
JUDGE_MODEL_GPT  = "gpt-5.4"

MAX_ROUNDS = 2  # Max structure revision + regeneration rounds

# =========================================================
# Dataset Path
# =========================================================
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gen_lcot", "data", "train.jsonl"
)


# =========================================================
# Self-Discover Prompts
# =========================================================

# Stage 1: Discover — Metacognition: discover optimal reasoning structure
DISCOVER_PROMPT = """You are a meta-reasoning agent. Given a research question,
your task is to DISCOVER the optimal reasoning STRUCTURE for designing an experimental plan.

Research Question:
{question}

Select and compose reasoning modules from this list to build the optimal structure:
- "problem_decomposition": Break the question into sub-problems
- "literature_grounding": Ground the approach in prior work
- "methodology_design": Design novel technical approaches
- "experimental_planning": Plan rigorous experiments
- "evaluation_framework": Define evaluation strategy
- "limitation_analysis": Analyze potential challenges
- "innovation_amplification": Amplify novelty and creativity
- "cross_domain_analogy": Draw analogies from other fields
- "iterative_refinement": Build in self-improvement loops

Output JSON:
{{
  "selected_modules": ["module1", "module2", ...],
  "reasoning_structure": "Step 1: [module_name] -> Step 2: [module_name] -> ...",
  "rationale": "why this structure is optimal for this specific question"
}}"""

# Stage 2: Apply — generate plan according to discovered structure
APPLY_PROMPT = """You are applying a self-discovered reasoning structure to generate an experimental plan.

Research Question:
{question}

OPTIMAL REASONING STRUCTURE (self-discovered for this question):
{structure}

Rationale for this structure:
{rationale}

RELEVANT REASONING MODULES TO APPLY:
{modules}

IMPORTANT CONTEXT:
{feedback}

Follow the structure EXACTLY. For each step, apply the specified reasoning module.
Generate a complete, rigorous experimental plan."""

# Scoring
SCORING_PROMPT = """You are an expert scientific reviewer. Rate the plan:
Reliability: xx
Innovation: xx
Only output the scores."""


# =========================================================
# Data & Utility Functions
# =========================================================

def load_dataset(path: str = DATASET_PATH, sample_size: int = 50) -> List[Dict]:
    """Load and randomly sample dataset"""
    data = [json.loads(line) for line in open(path, encoding="utf-8").read().strip().splitlines() if line.strip()]
    if sample_size and sample_size < len(data):
        random.seed(42)
        data = random.sample(data, sample_size)
    return data


def extract_user_content(sample: Dict) -> str:
    """Extract user question from sample"""
    for msg in sample.get("messages", []):
        if msg["role"] == "user":
            return msg["content"]
    return ""


def parse_score(text: Optional[str]) -> Dict[str, int]:
    """Parse Reliability and Innovation scores from score text"""
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


def call_llm(client: OpenAI, model: str, system_prompt: str, user_input: str,
             temperature: float = 0.1, max_retries: int = 5) -> str:
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
                timeout=180,
            )
            print(f"    [API] {model} OK")
            return response.choices[0].message.content
        except Exception as e:
            print(f"    [API] error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(sleep_time)
                sleep_time = min(sleep_time * 2, 60)
    return "[ERROR] all retries failed"


def score_single_plan(plan: str) -> Dict:
    """Score a single plan with both judges, returns {ds, gpt, combined}"""
    ds_text = call_llm(deepseek_client, JUDGE_MODEL_DS,
                       "You are a scientific reviewer. Output only: Reliability: xx\\nInnovation: xx",
                       SCORING_PROMPT + "\n\nPlan to evaluate:\n" + plan[:4000], temperature=0.0)
    ds = parse_score(ds_text)
    gpt_text = call_llm(gpt_client, JUDGE_MODEL_GPT,
                        "You are a scientific reviewer. Output only: Reliability: xx\\nInnovation: xx",
                        SCORING_PROMPT + "\n\nPlan to evaluate:\n" + plan[:4000], temperature=0.0)
    gpt = parse_score(gpt_text)
    combined = {
        "reliability": round((ds["reliability"] + gpt["reliability"]) / 2, 1),
        "innovation": round((ds["innovation"] + gpt["innovation"]) / 2, 1),
    }
    return {"ds": ds, "gpt": gpt, "combined": combined}


# =========================================================
# LangGraph State Definition
# =========================================================

class SelfDiscoverState(TypedDict):
    """Self-Discover state"""
    research_topic: str           # User's original question
    structure: str                # Discovered reasoning structure
    modules: List[str]            # Selected reasoning module list
    rationale: str                # Rationale for choosing this structure
    plan: str                     # Currently generated plan
    scores: List[Dict]            # Score history
    feedback: str                 # Feedback passed to next round (with structure adjustment suggestions)
    round_num: int                # Current round number
    final_report: str             # Final output report


# =========================================================
# Build Self-Discover Pipeline
# =========================================================

def build_self_discover_graph() -> StateGraph:
    """
    Build the Self-Discover state graph.

    Graph structure:
        __start__ -> discover -> apply -> score -> [NEEDS_IMPROV?]
                                                    ↓ YES -> apply (after revising structure)
                                                    ↓ NO  -> report -> END
    """

    # =====================
    # Node 1: Discover — discover optimal reasoning structure
    # =====================
    def discover(state: SelfDiscoverState) -> Dict:
        print("\n" + "=" * 60)
        print("[Stage 1] Self-Discover — Metacognition: Discover optimal reasoning structure")
        print("=" * 60)

        question = state["research_topic"]
        raw = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are a meta-reasoning agent. Discover the optimal reasoning structure.",
            DISCOVER_PROMPT.format(question=question),
            temperature=0.3
        )

        # Parse JSON
        structure = ""
        modules = []
        rationale = ""
        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                structure = data.get("reasoning_structure", "")
                modules = data.get("selected_modules", [])
                rationale = data.get("rationale", "")
        except json.JSONDecodeError:
            structure = raw[:500]
            modules = ["problem_decomposition", "methodology_design", "experimental_planning"]

        print(f"  -> Reasoning Structure: {structure[:120]}...")
        print(f"  -> Selected Modules ({len(modules)}): {modules}")

        return {"structure": structure, "modules": modules, "rationale": rationale}

    # =====================
    # Node 2: Apply — generate plan according to structure
    # =====================
    def apply(state: SelfDiscoverState) -> Dict:
        round_num = state.get("round_num", 0) + 1
        feedback = state.get("feedback", "(No previous feedback — this is the first attempt)")

        print("\n" + "=" * 60)
        print(f"[Stage 2] Apply Round {round_num} — generate plan according to self-discovered structure")
        print("=" * 60)

        question = state["research_topic"]
        plan = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "Apply the self-discovered reasoning structure to generate an experimental plan.",
            APPLY_PROMPT.format(
                question=question,
                structure=state.get("structure", ""),
                rationale=state.get("rationale", ""),
                modules=", ".join(state.get("modules", [])),
                feedback=feedback,
            ),
            temperature=0.3
        )

        print(f"  -> Plan: {len(plan)} chars")
        return {"plan": plan, "round_num": round_num}

    # =====================
    # Node 3: Dual-Judge Scoring
    # =====================
    def score_node(state: SelfDiscoverState) -> Dict:
        round_num = state.get("round_num", 1)
        plan = state.get("plan", "")

        print("\n" + "=" * 60)
        print(f"[Score] Round {round_num} evaluation")
        print("=" * 60)

        scores_data = score_single_plan(plan)
        c = scores_data["combined"]
        verdict = "GOOD" if c["reliability"] >= 70 else "NEEDS_IMPROVEMENT"

        print(f"  DS: R={scores_data['ds']['reliability']} I={scores_data['ds']['innovation']}")
        print(f"  GPT: R={scores_data['gpt']['reliability']} I={scores_data['gpt']['innovation']}")
        print(f"  Combined: R={c['reliability']} I={c['innovation']} -> {verdict}")

        # Build feedback: if score is insufficient, suggest adjusting reasoning structure
        scores = list(state.get("scores", []))
        scores.append({
            "round": round_num,
            "ds": scores_data["ds"],
            "gpt": scores_data["gpt"],
            "combined": c,
            "verdict": verdict,
        })

        if verdict == "NEEDS_IMPROVEMENT" and round_num < MAX_ROUNDS:
            # Self-Discover's unique aspect: feedback includes structure adjustment suggestions
            feedback = (
                f"Previous plan scored R={c['reliability']} I={c['innovation']} — BELOW THRESHOLD. "
                f"The reasoning structure may need adjustment. "
                f"Consider adding 'iterative_refinement' or 'innovation_amplification' modules. "
                f"Generate a DIFFERENT plan with a refined approach."
            )
        else:
            feedback = ""

        return {"scores": scores, "feedback": feedback}

    # =====================
    # Node 4: Final Report
    # =====================
    def report(state: SelfDiscoverState) -> Dict:
        print("\n" + "=" * 60)
        print("[Report] Generate Self-Discover Final Report")
        print("=" * 60)

        plan = state.get("plan", "")
        scores = state.get("scores", [])
        structure = state.get("structure", "")
        modules = state.get("modules", [])

        display = plan[:5000] + "\n\n...(truncated)" if len(plan) > 5000 else plan

        lines = [
            "=" * 70,
            "          Self-Discover Experimental Plan Report",
            "=" * 70,
            "",
            f"Research Topic: {state['research_topic'][:300]}",
            "",
            "-" * 70,
            "Metacognition: LLM Self-Discovered Optimal Reasoning Structure",
            "-" * 70,
            f"  Reasoning Structure: {structure[:200]}",
            f"  Selected Modules ({len(modules)}): {modules}",
            "",
            "Score History:",
        ]
        for s in scores:
            lines.append(
                f"  R{s['round']}: DS(R={s['ds'].get('reliability', 0)} I={s['ds'].get('innovation', 0)}) "
                f"GPT(R={s['gpt'].get('reliability', 0)} I={s['gpt'].get('innovation', 0)}) "
                f"-> Combined R={s['combined'].get('reliability', 0)} I={s['combined'].get('innovation', 0)}"
            )

        # Improvement delta (if two rounds)
        if len(scores) >= 2:
            delta_r = round(scores[-1]["combined"]["reliability"] - scores[0]["combined"]["reliability"], 1)
            delta_i = round(scores[-1]["combined"]["innovation"] - scores[0]["combined"]["innovation"], 1)
            lines.append(f"\n  Structure Revision Improvement: Delta R = +{delta_r}  Delta I = +{delta_i}")

        lines += [
            "",
            "-" * 70,
            "Final Plan (based on optimal structure)",
            "-" * 70,
            "",
            display or "Failed to generate plan",
            "",
            "=" * 70,
        ]
        return {"final_report": "\n".join(lines)}

    # =====================
    # Routing: continue revising or finish?
    # =====================
    def route_after_score(state: SelfDiscoverState) -> Literal["apply", "report"]:
        round_num = state.get("round_num", 1)
        scores = state.get("scores", [])

        if round_num >= MAX_ROUNDS:
            print(f"  -> Max rounds reached ({MAX_ROUNDS}), output report")
            return "report"
        if scores and scores[-1]["verdict"] == "GOOD":
            print(f"  -> Score acceptable, output report")
            return "report"
        if scores and scores[-1]["verdict"] == "NEEDS_IMPROVEMENT":
            print(f"  -> Score insufficient, revising structure then re-apply...")
            return "apply"
        return "report"

    # =====================
    # Assemble Graph
    # =====================
    workflow = StateGraph(SelfDiscoverState)

    workflow.add_node("discover", discover)
    workflow.add_node("apply", apply)
    workflow.add_node("score", score_node)
    workflow.add_node("report", report)

    workflow.set_entry_point("discover")
    workflow.add_edge("discover", "apply")
    workflow.add_edge("apply", "score")
    workflow.add_conditional_edges(
        "score", route_after_score,
        {"apply": "apply", "report": "report"},
    )
    workflow.add_edge("report", END)

    return workflow


# =========================================================
# Compile & Run
# =========================================================

def compile_agent():
    """Compile Self-Discover Agent graph"""
    return build_self_discover_graph().compile(checkpointer=MemorySaver())


def run_agent(question: str, verbose: bool = True) -> Dict[str, Any]:
    """Run Self-Discover Agent.
    Returns: {final_report, plan, scores, structure}
    """
    app = compile_agent()
    initial: SelfDiscoverState = {
        "research_topic": question,
        "structure": "",
        "modules": [],
        "rationale": "",
        "plan": "",
        "scores": [],
        "feedback": "",
        "round_num": 0,
        "final_report": "",
    }
    config = {"configurable": {"thread_id": f"sd-{int(time.time())}"}}

    print(f"\n{'★' * 70}")
    print(f"Self-Discover: {question[:200]}")
    print(f"{'★' * 70}")
    print("Pipeline: Discover (Metacognition) -> Apply (generate by structure) -> Score -> [Revise structure] -> Report")

    final_state = None
    for event in app.stream(initial, config, stream_mode="values"):
        final_state = event

    if final_state is None:
        return {"final_report": "[ERROR]", "plan": "", "scores": [], "structure": ""}

    return {
        "final_report": final_state.get("final_report", ""),
        "plan": final_state.get("plan", ""),
        "scores": final_state.get("scores", []),
        "structure": final_state.get("structure", ""),
    }


# =========================================================
# Cross-Sample Average Summary
# =========================================================

def _avg(lst: List[float]) -> float:
    return round(sum(lst) / len(lst), 1) if lst else 0.0


def compute_batch_averages(all_results: List[Dict]) -> Dict:
    """Compute cross-sample averages for batch testing (grouped by round)"""
    by_round = {}
    for result in all_results:
        for s in result.get("scores", []):
            r = s.get("round", 1)
            if r not in by_round:
                by_round[r] = {"rel": [], "inn": []}
            c = s.get("combined", {})
            by_round[r]["rel"].append(c.get("reliability", 0))
            by_round[r]["inn"].append(c.get("innovation", 0))

    rounds_summary = {
        ri: {
            "count": len(d["rel"]),
            "reliability_avg": _avg(d["rel"]),
            "innovation_avg": _avg(d["inn"]),
        }
        for ri, d in sorted(by_round.items())
    }

    all_rel = sum((d["rel"] for d in by_round.values()), [])
    all_inn = sum((d["inn"] for d in by_round.values()), [])
    return {
        "rounds": rounds_summary,
        "overall": {
            "total": len(all_rel),
            "reliability_avg": _avg(all_rel),
            "innovation_avg": _avg(all_inn),
        },
    }


# =========================================================
# Interactive CLI
# =========================================================

def interactive_loop():
    """Interactive command-line interface"""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║    Self-Discover Experimental Plan Generator                ║
║    Discover (metacognition finds optimal structure) -> Apply -> Score -> Revise structure  ║
║    Generator: DeepSeek V4 Pro | Judges: DS-chat + GPT-5.4   ║
╚══════════════════════════════════════════════════════════════╝

Commands:
  /run <question> -- Generate experimental plan for research question (Self-Discover)
  /batch [N]     -- Batch testing (default 3 samples)
  /demo          -- Use built-in demo
  /quit          -- Exit
""")

    while True:
        try:
            ui = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not ui:
            continue
        if ui.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye!")
            break
        if ui.lower() == "/demo":
            ui = "/run Design an experiment to evaluate whether LLMs can self-improve via recursive self-critique"

        if ui.lower().startswith("/run "):
            question = ui[5:].strip()
            result = run_agent(question)
            print("\n" + result["final_report"])

            out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f"sd_result_{int(time.time())}.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n[Result saved] {out}")

        elif ui.lower().startswith("/batch"):
            parts = ui.split()
            n = int(parts[1]) if len(parts) > 1 else 3
            batch_test(n)
        else:
            print("Unknown command. Use /run <question> to start.")


def batch_test(n: int = 3):
    """Batch testing"""
    dataset = load_dataset(sample_size=n)
    print(f"\n{'#' * 70}")
    print(f"  Self-Discover Batch Testing: {n} samples")
    print(f"  Generator: {GENERATOR_MODEL} | Judges: {JUDGE_MODEL_DS} + {JUDGE_MODEL_GPT}")
    print(f"{'#' * 70}")

    all_results = []
    for i, sample in enumerate(dataset):
        user_q = extract_user_content(sample)
        req_match = re.search(r"<user_request>\s*(.*?)(?:<think>|$)", user_q, re.DOTALL)
        question = req_match.group(1).strip()[:500] if req_match else user_q[:500]

        print(f"\n{'#' * 60}")
        print(f"  Sample {i + 1}/{n}: {question[:100]}...")
        print(f"{'#' * 60}")

        result = run_agent(question, verbose=False)
        all_results.append({
            "sample_index": i,
            "question": question,
            "report": result["final_report"],
            "scores": result.get("scores", []),
        })

        for s in result.get("scores", []):
            print(f"  -> R{s['round']}: Combined R={s['combined'].get('reliability', 0)} I={s['combined'].get('innovation', 0)}")

    batch_avg = compute_batch_averages(all_results)
    print("\n" + "=" * 70)
    print("         Self-Discover Batch Testing — Cross-Sample Average Summary")
    print("=" * 70)

    for ri in sorted(batch_avg["rounds"].keys()):
        info = batch_avg["rounds"][ri]
        print(f"\n  [R{ri}]  (Covering {info['count']} score(s))")
        print(f"    * Reliability = {info['reliability_avg']}  |  Innovation = {info['innovation_avg']}")

    overall = batch_avg["overall"]
    print(f"\n  {'─' * 60}")
    print(f"  [Overall Total]  (total {overall['total']} score(s))")
    print(f"    * Reliability = {overall['reliability_avg']}  |  Innovation = {overall['innovation_avg']}")
    print(f"\n{'=' * 70}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"sd_batch_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"generator": GENERATOR_MODEL, "judges": [JUDGE_MODEL_DS, JUDGE_MODEL_GPT], "samples": n},
            "batch_averages": batch_avg,
            "samples": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[Result saved] {out}")


if __name__ == "__main__":
    interactive_loop()

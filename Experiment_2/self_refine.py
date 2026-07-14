"""
Experiment_2/self_refine.py
Self-Refine Experiment Plan Generator
Model self-generate → self-critique → self-improve, iterative refinement (not ReAct / Reverse-CoT)
Generator Model: DeepSeek V4 Pro  |  Judge Models: DeepSeek-chat + GPT-5.4
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
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    timeout=httpx.Timeout(300.0)
)
gpt_client = OpenAI(
    api_key=GPT_API_KEY,
    base_url="https://www.autodl.art/api/v1",
    timeout=httpx.Timeout(300.0)
)

GENERATOR_MODEL = "deepseek-v4-pro"
JUDGE_MODEL_DS   = "deepseek-chat"
JUDGE_MODEL_GPT  = "gpt-5.4"

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gen_lcot", "data", "train.jsonl"
)

# =========================================================
# Self-Refine Prompts
# =========================================================

# Stage 1: Initial Generation
INITIAL_GENERATION_PROMPT = """You are a professional AI research scientist.
Based on the user's research objective, generate a complete experimental plan.

Research Question:
{question}

Requirements:
1. Clearly describe the research objective
2. Design the model or methodology
3. Provide detailed experimental procedures
4. Include datasets and evaluation metrics
5. Analyze possible challenges and limitations
6. Maintain rigorous scientific reasoning
7. Encourage methodological innovation
8. Ensure the plan is practical and reproducible

Generate a comprehensive, well-structured plan."""

# Stage 2: Self-Critique — Model examines its own plan and identifies weaknesses
SELF_CRITIQUE_PROMPT = """You are a rigorous and meticulous research reviewer.
Your task is to CRITIQUE the following experimental plan and identify ALL weaknesses, gaps, and areas for improvement.

CURRENT PLAN:
----------------
{plan}
----------------

Research Question: {question}
Current Iteration: {iteration} of {max_iterations}

Please provide a DETAILED critique covering:

1. METHODOLOGY WEAKNESSES:
   - Are there logical flaws or missing steps?
   - Is the approach technically sound?
   - Are assumptions clearly stated?

2. EXPERIMENTAL GAPS:
   - Are the experiments sufficient to validate claims?
   - Are ablation studies properly designed?
   - Is statistical rigor adequate?

3. EVALUATION CONCERNS:
   - Are the metrics appropriate?
   - Are baselines fairly chosen?
   - Are success criteria well-defined?

4. INNOVATION ASSESSMENT:
   - Where does the plan lack originality?
   - Are there missed opportunities for novel contributions?

5. REPRODUCIBILITY & PRACTICALITY:
   - Are implementation details sufficient?
   - Are resource requirements realistic?

Be specific. For each weakness, explain WHY it matters and WHAT would fix it.
Do NOT be polite — be thorough and critical. This critique will be used to improve the plan."""

# Stage 3: Self-Improvement Based on Critique
SELF_REFINE_PROMPT = """You are an expert research scientist improving your own work based on reviewer feedback.

ORIGINAL PLAN:
----------------
{plan}
----------------

CRITIQUE RECEIVED:
----------------
{critique}
----------------

Research Question: {question}

Your task: Rewrite the experimental plan by addressing EVERY point raised in the critique.
- Fix logical flaws and missing steps
- Strengthen experimental design
- Improve evaluation metrics
- Enhance innovation where possible
- Add implementation details for reproducibility

Output the COMPLETE REVISED plan (not just the changes). The revised plan should be
comprehensive, well-structured, and academically professional.

Be honest about remaining limitations — acknowledge what cannot be fully addressed."""

# Scoring
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


# =========================================================
# Data & Tools
# =========================================================

def load_dataset(path: str = DATASET_PATH, sample_size: int = 50) -> List[Dict]:
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


def extract_user_content(sample: Dict) -> str:
    for msg in sample.get("messages", []):
        if msg["role"] == "user":
            return msg["content"]
    return ""


def parse_score(text: Optional[str]) -> Dict[str, int]:
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
    return "[ERROR] All API retries failed"


# =========================================================
# LangGraph State
# =========================================================

class SelfRefineState(TypedDict):
    research_topic: str
    plan: str                      # Current Plan
    critique: str                  # Latest round of self-critique
    iteration: int                 # Current refine round (1-based)
    max_iterations: int            # Max refine count (default 3)
    scores: List[Dict]             # Score History
    critique_history: List[Dict]   # [{iteration, critique_snippet, plan_snippet}]
    final_report: str


# =========================================================
# Build Self-Refine Pipeline
# =========================================================

MAX_REFINE_ITERATIONS = 3


def build_self_refine_graph() -> StateGraph:

    # =====================
    # Node 1: Initial Generation
    # =====================
    def initial_generate(state: SelfRefineState) -> Dict:
        print("\n" + "=" * 60)
        print("[Stage 1] Initial Generation")
        print("=" * 60)

        question = state["research_topic"]
        plan = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are a professional AI research scientist generating an initial experimental plan.",
            INITIAL_GENERATION_PROMPT.format(question=question),
            temperature=0.3
        )
        print(f"  -> Initial plan: {len(plan)} chars")
        return {"plan": plan, "iteration": 1}

    # =====================
    # Node 2: Self-Critique
    # =====================
    def self_critique(state: SelfRefineState) -> Dict:
        iteration = state.get("iteration", 1)
        print("\n" + "=" * 60)
        print(f"[Stage 2] Self-Critique (round {iteration}/{state.get('max_iterations', MAX_REFINE_ITERATIONS)})")
        print("=" * 60)

        plan = state.get("plan", "")
        question = state["research_topic"]

        critique = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are a rigorous research reviewer. Be critical and thorough.",
            SELF_CRITIQUE_PROMPT.format(
                plan=plan,
                question=question,
                iteration=iteration,
                max_iterations=state.get("max_iterations", MAX_REFINE_ITERATIONS),
            ),
            temperature=0.2
        )

        # Record critique history
        critique_history = list(state.get("critique_history", []))
        critique_history.append({
            "iteration": iteration,
            "critique_snippet": critique[:300],
            "plan_snippet": plan[:300],
        })

        print(f"  -> Critique feedback: {len(critique)} chars")
        return {"critique": critique, "critique_history": critique_history}

    # =====================
    # Node 3: Improve Based on Critique
    # =====================
    def refine(state: SelfRefineState) -> Dict:
        iteration = state.get("iteration", 1)
        print("\n" + "=" * 60)
        print(f"[Stage 3] Self-Improvement (round {iteration}/{state.get('max_iterations', MAX_REFINE_ITERATIONS)})")
        print("=" * 60)

        plan = state.get("plan", "")
        critique = state.get("critique", "")
        question = state["research_topic"]

        improved = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are an expert improving your work based on critical feedback.",
            SELF_REFINE_PROMPT.format(
                plan=plan,
                critique=critique,
                question=question,
            ),
            temperature=0.3
        )

        new_iteration = iteration + 1
        print(f"  -> Improved plan: {len(improved)} chars")

        return {"plan": improved, "iteration": new_iteration}

    # =====================
    # Node 4: Dual-Judge Scoring
    # =====================
    def score_node(state: SelfRefineState) -> Dict:
        print("\n" + "=" * 60)
        print("[Scoring] Dual-Judge evaluate Final Plan")
        print("=" * 60)

        plan = state.get("plan", "")

        print("  [Judge 1] DeepSeek-chat ...")
        ds_text = call_llm(deepseek_client, JUDGE_MODEL_DS, SCORING_PROMPT, plan, temperature=0.0)
        ds_scores = parse_score(ds_text)
        print(f"    DS -> R={ds_scores['reliability']}, I={ds_scores['innovation']}")

        print("  [Judge 2] GPT-5.4 ...")
        gpt_text = call_llm(gpt_client, JUDGE_MODEL_GPT, SCORING_PROMPT, plan, temperature=0.0)
        gpt_scores = parse_score(gpt_text)
        print(f"    GPT -> R={gpt_scores['reliability']}, I={gpt_scores['innovation']}")

        combined = {
            "reliability": round((ds_scores["reliability"] + gpt_scores["reliability"]) / 2, 1),
            "innovation": round((ds_scores["innovation"] + gpt_scores["innovation"]) / 2, 1),
        }

        verdict = (
            "EXCELLENT" if combined["reliability"] >= 80 and combined["innovation"] >= 75
            else "GOOD" if combined["reliability"] >= 70 and combined["innovation"] >= 60
            else "NEEDS_IMPROVEMENT"
        )

        scores = list(state.get("scores", []))
        scores.append({
            "round": len(scores) + 1,
            "ds": ds_scores,
            "gpt": gpt_scores,
            "combined": combined,
            "verdict": verdict,
        })

        print(f"  Combined -> R={combined['reliability']}, I={combined['innovation']} -> {verdict}")

        return {"scores": scores}

    # =====================
    # Node 5: Final Report
    # =====================
    def report_node(state: SelfRefineState) -> Dict:
        print("\n" + "=" * 60)
        print("[Report] Generate Self-Refine Final Report")
        print("=" * 60)

        topic = state["research_topic"]
        plan = state.get("plan", "")
        scores = state.get("scores", [])
        iteration = state.get("iteration", 1)
        critique_history = state.get("critique_history", [])

        display_plan = plan[:5000] + "\n\n...(truncated)" if len(plan) > 5000 else plan

        lines = [
            "=" * 70,
            "          Self-Refine Experiment Plan Report",
            "=" * 70,
            "",
            f"Research Topic: {topic[:300]}",
            "",
            "-" * 70,
            "Generation Pipeline: Initial Generation → Self-Critique → Self-Improvement → Scoring",
            f"Refine rounds completed: {len(critique_history)}",
            "-" * 70,
            "",
            "Final Plan:",
            display_plan or "Failed to generate",
        ]

        if scores:
            lines += ["", "-" * 70, "Quality Assessment", "-" * 70]
            for s in scores:
                c = s["combined"]
                lines.append(
                    f"  R{s['round']}: DS(R={s['ds'].get('reliability',0)} I={s['ds'].get('innovation',0)}) "
                    f"GPT(R={s['gpt'].get('reliability',0)} I={s['gpt'].get('innovation',0)}) "
                    f"→ Combined R={c.get('reliability',0)} I={c.get('innovation',0)} [{s.get('verdict','')}]"
                )

            # Improvement trend
            if len(scores) >= 2:
                delta_r = round(scores[-1]["combined"]["reliability"] - scores[0]["combined"]["reliability"], 1)
                delta_i = round(scores[-1]["combined"]["innovation"] - scores[0]["combined"]["innovation"], 1)
                lines += [
                    "",
                    f"  Self-Refine improvement:  Delta R = +{delta_r}  |  Delta I = +{delta_i}",
                ]

        # Critique history summary
        if critique_history:
            lines += ["", "-" * 70, "Self-Refine Iteration History", "-" * 70]
            for ch in critique_history:
                lines.append(f"  Round {ch['iteration']}: critique {len(ch['critique_snippet'])} chars -> improved plan {len(ch['plan_snippet'])} chars")

        lines += ["", f"Total refine count: {len(critique_history)}", "=" * 70]

        return {"final_report": "\n".join(lines)}

    # =====================
    # Route: Continue Refine or End?
    # =====================
    def route_after_score(state: SelfRefineState) -> Literal["critique", "report"]:
        iteration = state.get("iteration", 1)
        max_iter = state.get("max_iterations", MAX_REFINE_ITERATIONS)

        if iteration > max_iter:
            print(f"  -> Max refine rounds reached ({max_iter}), output report")
            return "report"

        scores = state.get("scores", [])
        if scores and scores[-1]["verdict"] in ("EXCELLENT", "GOOD"):
            print(f"  -> Score acceptable ({scores[-1]['verdict']}), output report")
            return "report"

        print(f"  -> Continue refine (currently at round {iteration}, max {max_iter} rounds)")
        return "critique"

    # =====================
    # Assemble Graph
    #        initial_generate → critique → refine → score
    #                             ↑                    ↓
    #                             └─── [Continue?] ←───┘
    #                                       ↓ [End]
    #                                    report → END
    # =====================
    workflow = StateGraph(SelfRefineState)

    workflow.add_node("initial_generate", initial_generate)
    workflow.add_node("critique", self_critique)
    workflow.add_node("refine", refine)
    workflow.add_node("score", score_node)
    workflow.add_node("report", report_node)

    workflow.set_entry_point("initial_generate")

    # After initial generation → critique → refine → score → loop
    # Both modes are reasonable. Here we use: initial → critique → refine → score → loop
    workflow.add_edge("initial_generate", "critique")
    workflow.add_edge("critique", "refine")
    workflow.add_edge("refine", "score")

    workflow.add_conditional_edges(
        "score", route_after_score,
        {"critique": "critique", "report": "report"},
    )
    workflow.add_edge("report", END)

    return workflow


# =========================================================
# Compile & Run
# =========================================================

def compile_agent():
    workflow = build_self_refine_graph()
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def run_agent(question: str, max_iterations: int = MAX_REFINE_ITERATIONS, verbose: bool = True) -> Dict[str, Any]:
    app = compile_agent()

    initial_state: SelfRefineState = {
        "research_topic": question,
        "plan": "",
        "critique": "",
        "iteration": 1,
        "max_iterations": max_iterations,
        "scores": [],
        "critique_history": [],
        "final_report": "",
    }

    config = {"configurable": {"thread_id": f"selfrefine-{int(time.time())}"}}

    print(f"\n{'★'*70}")
    print(f"Self-Refine: {question[:200]}")
    print(f"{'★'*70}")
    print(f"Workflow: Initial Generation → Self-Critique → Self-Improvement → Scoring → [Continue refine?] (max {max_iterations} rounds)")

    final_state = None
    for event in app.stream(initial_state, config, stream_mode="values"):
        final_state = event
        if verbose:
            for key in ["plan", "critique"]:
                val = event.get(key, "")
                if val and len(val) > 50:
                    print(f"  [{key}] {len(val)} chars")

    if final_state is None:
        return {"final_report": "[ERROR]", "scores": [], "critique_history": []}

    return {
        "final_report": final_state.get("final_report", ""),
        "plan": final_state.get("plan", ""),
        "scores": final_state.get("scores", []),
        "critique_history": final_state.get("critique_history", []),
    }


# =========================================================
# Cross-Sample Averages
# =========================================================

def _avg(lst: List[float]) -> float:
    return round(sum(lst) / len(lst), 1) if lst else 0.0


def compute_batch_averages(all_results: List[Dict]) -> Dict:
    by_round: Dict[int, Dict[str, List[float]]] = {}

    for result in all_results:
        for s in result.get("scores", []):
            r = s.get("round", 1)
            if r not in by_round:
                by_round[r] = {"ds_rel": [], "ds_inn": [], "gpt_rel": [], "gpt_inn": [],
                               "combined_rel": [], "combined_inn": []}
            by_round[r]["ds_rel"].append(s["ds"].get("reliability", 0))
            by_round[r]["ds_inn"].append(s["ds"].get("innovation", 0))
            by_round[r]["gpt_rel"].append(s["gpt"].get("reliability", 0))
            by_round[r]["gpt_inn"].append(s["gpt"].get("innovation", 0))
            by_round[r]["combined_rel"].append(s["combined"].get("reliability", 0))
            by_round[r]["combined_inn"].append(s["combined"].get("innovation", 0))

    rounds_summary = {}
    all_ds_rel, all_ds_inn, all_gpt_rel, all_gpt_inn = [], [], [], []
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
            "ds_reliability_avg": _avg(all_ds_rel), "ds_innovation_avg": _avg(all_ds_inn),
            "gpt_reliability_avg": _avg(all_gpt_rel), "gpt_innovation_avg": _avg(all_gpt_inn),
            "combined_reliability_avg": _avg(all_comb_rel), "combined_innovation_avg": _avg(all_comb_inn),
        },
    }


# =========================================================
# CLI
# =========================================================

def interactive_loop():
    print(r"""
╔══════════════════════════════════════════════════════════════╗
║    Self-Refine Experiment Plan Generator                       ║
║    Workflow: Generate → Self-Critique → Improve → Score → [Continue refine] ║
║    Generator: DeepSeek V4 Pro | Judges: DS-chat + GPT-5.4    ║
╚══════════════════════════════════════════════════════════════╝

Commands:
  /run <question>   — Generate experimental plan for research question (Self-Refine)
  /batch [N]        — Batch test (default 3)
  /demo             — Use built-in demo
  /quit             — Exit
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

            output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       f"selfrefine_result_{int(time.time())}.json")
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
    dataset = load_dataset(sample_size=n)
    print(f"\n{'#'*70}")
    print(f"  Self-Refine Batch Test: {n} samples")
    print(f"  Generator: {GENERATOR_MODEL} | Judges: {JUDGE_MODEL_DS} + {JUDGE_MODEL_GPT}")
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
            "sample_index": i, "question": question,
            "report": result["final_report"], "scores": result.get("scores", []),
        })

        for s in result.get("scores", []):
            c = s["combined"]
            print(f"  -> R{s['round']}: DS(R={s['ds'].get('reliability',0)} I={s['ds'].get('innovation',0)}) "
                  f"GPT(R={s['gpt'].get('reliability',0)} I={s['gpt'].get('innovation',0)}) "
                  f"Combined(R={c.get('reliability',0)} I={c.get('innovation',0)}) [{s.get('verdict','')}]")

    batch_avg = compute_batch_averages(all_results)

    print("\n" + "=" * 70)
    print("         Self-Refine Batch Test — Cross-Sample Average Summary")
    print("=" * 70)

    for r in sorted(batch_avg["rounds"].keys()):
        info = batch_avg["rounds"][r]
        label = f"Round {r} Score"
        print(f"\n  [{label}]  (covering {info['count']}/{n} samples)")
        print(f"    DeepSeek Judge:  R = {info['ds_reliability_avg']}  |  I = {info['ds_innovation_avg']}")
        print(f"    GPT Judge:      R = {info['gpt_reliability_avg']}  |  I = {info['gpt_innovation_avg']}")
        print(f"    * Combined avg:    R = {info['combined_reliability_avg']}  |  I = {info['combined_innovation_avg']}")

    overall = batch_avg["overall"]
    print(f"\n  {'─'*60}")
    print(f"  [Overall Total]  ({overall['total_scores']} score(s) total)")
    print(f"    DeepSeek Judge:  R = {overall['ds_reliability_avg']}  |  I = {overall['ds_innovation_avg']}")
    print(f"    GPT Judge:      R = {overall['gpt_reliability_avg']}  |  I = {overall['gpt_innovation_avg']}")
    print(f"    * Combined avg:    R = {overall['combined_reliability_avg']}  |  I = {overall['combined_innovation_avg']}")
    print(f"\n{'='*70}")

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f"selfrefine_batch_{int(time.time())}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"generator": GENERATOR_MODEL, "judges": [JUDGE_MODEL_DS, JUDGE_MODEL_GPT], "samples": n},
            "batch_averages": batch_avg, "samples": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[Result saved] {output_path}")


if __name__ == "__main__":
    interactive_loop()

"""
Experiment_2/reverse_cot.py
Reverse Chain-of-Thought Experiment Plan Generator
Reverse-engineer methodology from desired outcomes, fixed pipeline (not ReAct agent)
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

GENERATOR_MODEL = "deepseek-v4-pro"
JUDGE_MODEL_DS   = "deepseek-chat"
JUDGE_MODEL_GPT  = "gpt-5.4"

# =========================================================
# Dataset Path
# =========================================================
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gen_lcot", "data", "train.jsonl"
)


# =========================================================
# Reverse-CoT Prompts
# =========================================================

# Node 1: Reverse — Ideal Evaluation Criteria and Desired Outcomes
REVERSE_EVALUATION_PROMPT = """You are an expert research evaluator. Given a research question, your job is to work BACKWARDS: first define what an IDEAL experimental outcome would look like.

Research Question:
{question}

Please define, in as much detail as possible:

1. IDEAL OUTCOMES: What results would constitute a successful experiment? Be specific.
2. EVALUATION METRICS: What quantitative and qualitative metrics would demonstrate success?
3. BENCHMARKS & BASELINES: What existing methods should this be compared against?
4. SUCCESS THRESHOLDS: What numbers would make this publishable at a top venue?

Think from the END first — what does "done, successful" look like?"""


# Node 2: Reverse — Derive Experimental Design from Evaluation Criteria
REVERSE_EXPERIMENT_PROMPT = """You are an expert experimental designer.

Research Question:
{question}

Desired Evaluation & Outcomes (from previous step):
{evaluation_spec}

Now work BACKWARDS: Given these desired evaluation metrics and outcomes, what experimental setup would produce them?

Design:
1. DATASETS: What datasets would allow you to measure these exact metrics?
2. EXPERIMENTAL SETUP: Train/val/test splits, cross-validation strategy, hardware needs
3. ABLATION STUDIES: What components need to be isolated and tested?
4. COMPARISON PROTOCOL: How to ensure fair comparison with baselines?
5. STATISTICAL RIGOR: Number of runs, significance tests, error bars

Your experimental design must be EXPLICITLY traceable back to the evaluation metrics above."""


# Node 3: Reverse — Derive Methodology from Experimental Design
REVERSE_METHOD_PROMPT = """You are an expert research methodologist.

Research Question:
{question}

Required Experimental Setup (from previous step):
{experiment_spec}

Now work BACKWARDS: Given this experimental design, what methodology and model architecture would be required to run these experiments?

Design:
1. MODEL ARCHITECTURE: What model structure enables these experiments?
2. TRAINING METHODOLOGY: Loss functions, optimization, regularization
3. KEY INNOVATIONS: What novel technical contributions differentiate this approach?
4. THEORETICAL GROUNDING: Mathematical or algorithmic foundations
5. IMPLEMENTATION DETAILS: Key hyperparameters, training recipe

Your methodology must be PRECISELY what the experiments above require."""


# Node 4: Forward Verification — Method→Experiment→Evaluation Chain Consistency Check
FORWARD_VERIFY_PROMPT = """You are a rigorous research auditor.

Original Research Question:
{question}

Reverse-CoT Generated Components:
---
EVALUATION TARGET:
{evaluation_spec}

EXPERIMENTAL DESIGN:
{experiment_spec}

METHODOLOGY:
{method_spec}
---

Now work FORWARDS to verify consistency:
1. Does the METHODOLOGY logically enable the EXPERIMENTAL DESIGN?
2. Does the EXPERIMENTAL DESIGN measure the EVALUATION METRICS?
3. Are there any gaps or inconsistencies in the chain?
4. Is the ORIGINAL QUESTION actually answered?

If you find gaps, FIX THEM. Output the COMPLETE, VERIFIED experimental plan as a unified document including:
- Research Objective
- Methodology
- Experimental Design
- Evaluation Metrics
- Expected Outcomes
- Limitations"""


# Node 5: Scoring (reusable)
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


# Node 6: Optimization (if score insufficient)
OPTIMIZE_PROMPT = """You are an expert research mentor. Analyze the weaknesses of the following experimental plan
and generate a significantly improved version. Pay special attention to the Reverse-CoT chain consistency.

ORIGINAL PLAN:
----------------
{plan}
----------------

CURRENT SCORES: Reliability={reliability}, Innovation={innovation}

IMPROVEMENT FOCUS:
- Ensure the Method → Experiment → Evaluation chain is airtight
- If reliability is low: strengthen experimental rigor, add validation steps
- If innovation is low: incorporate novel methodologies, creative evaluation
- Every component must be traceable backwards to the original question

Generate the improved plan directly."""


# =========================================================
# Data Loading & Tools
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


# =========================================================
# API Call
# =========================================================

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

class ReverseCoTState(TypedDict):
    research_topic: str
    evaluation_spec: str          # Node 1 output
    experiment_spec: str          # Node 2 output
    method_spec: str              # Node 3 output
    verified_plan: str            # Node 4 output (final plan)
    scores: List[Dict]            # Score History
    final_report: str
    optimize_count: int           # Optimization count


# =========================================================
# Build Reverse-CoT Pipeline
# =========================================================

def build_reverse_cot_graph() -> StateGraph:

    # =====================
    # Node 1: Reverse Evaluation Criteria
    # =====================
    def reverse_evaluation(state: ReverseCoTState) -> Dict:
        print("\n" + "=" * 60)
        print("[Node 1] Reverse — Define Ideal Evaluation Criteria and Desired Outcomes")
        print("=" * 60)

        question = state["research_topic"]
        result = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are an expert research evaluator. Work backwards from ideal outcomes.",
            REVERSE_EVALUATION_PROMPT.format(question=question),
            temperature=0.3
        )
        print(f"  -> Generated {len(result)} chars evaluation spec")
        return {"evaluation_spec": result}

    # =====================
    # Node 2: Reverse Experimental Design
    # =====================
    def reverse_experiment(state: ReverseCoTState) -> Dict:
        print("\n" + "=" * 60)
        print("[Node 2] Reverse — Derive Experimental Design from Evaluation Criteria")
        print("=" * 60)

        result = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are an expert experimental designer. Work backwards from evaluation to experiment.",
            REVERSE_EXPERIMENT_PROMPT.format(
                question=state["research_topic"],
                evaluation_spec=state.get("evaluation_spec", ""),
            ),
            temperature=0.3
        )
        print(f"  -> Generated {len(result)} chars experiment design")
        return {"experiment_spec": result}

    # =====================
    # Node 3: Reverse Methodology
    # =====================
    def reverse_method(state: ReverseCoTState) -> Dict:
        print("\n" + "=" * 60)
        print("[Node 3] Reverse — Derive Methodology from Experimental Design")
        print("=" * 60)

        result = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are an expert research methodologist. Work backwards from experiment to method.",
            REVERSE_METHOD_PROMPT.format(
                question=state["research_topic"],
                experiment_spec=state.get("experiment_spec", ""),
            ),
            temperature=0.3
        )
        print(f"  -> Generated {len(result)} chars methodology")
        return {"method_spec": result}

    # =====================
    # Node 4: Forward Verification + Synthesize Final Plan
    # =====================
    def forward_verify(state: ReverseCoTState) -> Dict:
        print("\n" + "=" * 60)
        print("[Node 4] Forward Verification — Check Method→Experiment→Evaluation Chain")
        print("=" * 60)

        result = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are a rigorous research auditor. Verify the reverse-CoT chain and fix gaps.",
            FORWARD_VERIFY_PROMPT.format(
                question=state["research_topic"],
                evaluation_spec=state.get("evaluation_spec", ""),
                experiment_spec=state.get("experiment_spec", ""),
                method_spec=state.get("method_spec", ""),
            ),
            temperature=0.2
        )
        print(f"  -> Generated {len(result)} chars verified plan")
        return {"verified_plan": result}

    # =====================
    # Node 5: Dual-Judge Scoring
    # =====================
    def score_node(state: ReverseCoTState) -> Dict:
        print("\n" + "=" * 60)
        print("[Node 5] Dual-Judge Scoring")
        print("=" * 60)

        plan = state.get("verified_plan", "")

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

        scores = [{
            "round": state.get("optimize_count", 0) + 1,
            "ds": ds_scores,
            "gpt": gpt_scores,
            "combined": combined,
            "verdict": (
                "EXCELLENT" if combined["reliability"] >= 80 and combined["innovation"] >= 75
                else "GOOD" if combined["reliability"] >= 70 and combined["innovation"] >= 60
                else "NEEDS_IMPROVEMENT"
            ),
        }]

        print(f"  Combined -> R={combined['reliability']}, I={combined['innovation']} -> {scores[0]['verdict']}")

        return {"scores": scores}

    # =====================
    # Node 6: Optimization (conditional execution)
    # =====================
    def optimize_node(state: ReverseCoTState) -> Dict:
        print("\n" + "=" * 60)
        print("[Node 6] Optimize Plan")
        print("=" * 60)

        plan = state.get("verified_plan", "")
        scores = state.get("scores", [])
        optimize_count = state.get("optimize_count", 0) + 1

        if scores:
            c = scores[-1]["combined"]
            reliability = c.get("reliability", 0)
            innovation = c.get("innovation", 0)
        else:
            reliability = innovation = 0

        opt_input = OPTIMIZE_PROMPT.format(plan=plan, reliability=reliability, innovation=innovation)
        improved = call_llm(deepseek_client, GENERATOR_MODEL,
                            "You are an expert research mentor improving experimental plans.",
                            opt_input, temperature=0.3)

        print(f"  -> Optimized plan {len(improved)} chars")
        return {"verified_plan": improved, "optimize_count": optimize_count}

    # =====================
    # Final Report Node
    # =====================
    def report_node(state: ReverseCoTState) -> Dict:
        print("\n" + "=" * 60)
        print("[Report] Generate Final Report")
        print("=" * 60)

        topic = state["research_topic"]
        plan = state.get("verified_plan", "")
        scores = state.get("scores", [])
        optimize_count = state.get("optimize_count", 0)

        if len(plan) > 5000:
            plan = plan[:5000] + "\n\n...(truncated)"

        lines = [
            "=" * 70,
            "          Reverse-CoT Experiment Plan Report",
            "=" * 70,
            "",
            f"Research Topic: {topic[:300]}",
            "",
            "-" * 70,
            "Generation Pipeline: Evaluation Criteria → Experimental Design → Methodology → Forward Verification",
            "-" * 70,
            "",
            "Final Plan:",
            plan or "Failed to generate",
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

        lines += [
            "",
            f"Optimization count: {optimize_count}",
            "=" * 70,
        ]

        return {"final_report": "\n".join(lines)}

    # =====================
    # Route: Need Optimization?
    # =====================
    def route_after_score(state: ReverseCoTState) -> Literal["optimize", "report"]:
        optimize_count = state.get("optimize_count", 0)
        scores = state.get("scores", [])

        if optimize_count >= 2:
            print("  -> Already optimized 2 times, output report")
            return "report"

        if scores and scores[-1]["verdict"] == "NEEDS_IMPROVEMENT":
            print("  -> Score insufficient, entering optimization")
            return "optimize"

        print("  -> Score acceptable, output report")
        return "report"

    # =====================
    # Assemble Graph
    # =====================
    workflow = StateGraph(ReverseCoTState)

    workflow.add_node("reverse_eval", reverse_evaluation)
    workflow.add_node("reverse_exp", reverse_experiment)
    workflow.add_node("reverse_method", reverse_method)
    workflow.add_node("forward_verify", forward_verify)
    workflow.add_node("score", score_node)
    workflow.add_node("optimize", optimize_node)
    workflow.add_node("report", report_node)

    workflow.set_entry_point("reverse_eval")

    # Fixed pipeline
    workflow.add_edge("reverse_eval", "reverse_exp")
    workflow.add_edge("reverse_exp", "reverse_method")
    workflow.add_edge("reverse_method", "forward_verify")
    workflow.add_edge("forward_verify", "score")

    # Conditional branch
    workflow.add_conditional_edges(
        "score", route_after_score,
        {"optimize": "optimize", "report": "report"},
    )
    workflow.add_edge("optimize", "score")  # After optimization, re-score
    workflow.add_edge("report", END)

    return workflow


# =========================================================
# Compile & Run
# =========================================================

def compile_agent():
    workflow = build_reverse_cot_graph()
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def run_agent(question: str, verbose: bool = True) -> Dict[str, Any]:
    app = compile_agent()

    initial_state: ReverseCoTState = {
        "research_topic": question,
        "evaluation_spec": "",
        "experiment_spec": "",
        "method_spec": "",
        "verified_plan": "",
        "scores": [],
        "final_report": "",
        "optimize_count": 0,
    }

    config = {"configurable": {"thread_id": f"revcot-{int(time.time())}"}}

    print(f"\n{'★'*70}")
    print(f"Reverse-CoT: {question[:200]}")
    print(f"{'★'*70}")
    print("Workflow: Evaluation ← Experiment ← Method → Forward Verification → Scoring → [Optimize] → Report")

    final_state = None
    for event in app.stream(initial_state, config, stream_mode="values"):
        final_state = event
        if verbose:
            # Print key output length per node
            for key in ["evaluation_spec", "experiment_spec", "method_spec", "verified_plan"]:
                val = event.get(key, "")
                if val and len(val) > 50:
                    print(f"  [{key}] {len(val)} chars")

    if final_state is None:
        return {"final_report": "[ERROR]", "scores": [], "optimize_count": 0}

    return {
        "final_report": final_state.get("final_report", ""),
        "scores": final_state.get("scores", []),
        "optimize_count": final_state.get("optimize_count", 0),
        "verified_plan": final_state.get("verified_plan", ""),
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
║    Reverse-CoT Experiment Plan Generator                       ║
║    Workflow: Eval ← Experiment ← Method → Forward Verify → Score → [Optimize] ║
║    Generator: DeepSeek V4 Pro | Judges: DS-chat + GPT-5.4    ║
╚══════════════════════════════════════════════════════════════╝

Commands:
  /run <question>   — Generate experimental plan for research question (Reverse-CoT)
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
                                       f"revcot_result_{int(time.time())}.json")
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
    print(f"  Reverse-CoT Batch Test: {n} samples")
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

    # Cross-Sample Averages
    batch_avg = compute_batch_averages(all_results)

    print("\n" + "=" * 70)
    print("         Reverse-CoT Batch Test — Cross-Sample Average Summary")
    print("=" * 70)

    for r in sorted(batch_avg["rounds"].keys()):
        info = batch_avg["rounds"][r]
        label = f"Round {r} (Initial Plan)" if r == 1 else f"Round {r} (Optimized Plan)"
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
                               f"revcot_batch_{int(time.time())}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"generator": GENERATOR_MODEL, "judges": [JUDGE_MODEL_DS, JUDGE_MODEL_GPT], "samples": n},
            "batch_averages": batch_avg, "samples": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[Result saved] {output_path}")


if __name__ == "__main__":
    interactive_loop()

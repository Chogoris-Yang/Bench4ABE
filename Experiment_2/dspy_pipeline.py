"""
Experiment_2/dspy_pipeline.py
DSPy-style experimental plan generator
Signature-driven modules + Few-shot bootstrapping + Module chain composition + Auto-optimization
Generator Model: DeepSeek V4 Pro  |  Judge Model: DeepSeek-chat + GPT-5.4
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

NUM_FEWSHOT = 3       # Number of few-shot examples
MAX_OPTIMIZE = 2       # Auto-optimization rounds

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gen_lcot", "data", "train.jsonl"
)


# =========================================================
# DSPy-style Signatures & Modules
# =========================================================

# ---- Signature Definitions ----
# Each Module has clear input/output signatures (DSPy core concept)

MODULE_SIGNATURES = {
    "GeneratePlan":  "question: str  ->  plan: str",
    "CritiquePlan":  "plan: str  ->  critique: str",
    "RefinePlan":    "plan: str, critique: str  ->  refined_plan: str",
    "ScorePlan":     "plan: str  ->  reliability: int, innovation: int",
}


# ---- Few-Shot Bootstrapper ----
# DSPy style: auto-select best examples from dataset as in-context demonstrations

def bootstrap_fewshot_examples(
    dataset: List[Dict], question: str, num_examples: int = NUM_FEWSHOT
) -> str:
    """
    Auto-select examples most relevant to question from dataset,
    format as few-shot demonstrations (DSPy BootstrapFewShot style)
    """
    query_lower = question.lower()
    keywords = query_lower.split()
    scored = []

    for i, sample in enumerate(dataset):
        user_text = ""
        assistant_text = ""
        for msg in sample.get("messages", []):
            if msg["role"] == "user":
                user_text = msg["content"]
            elif msg["role"] == "assistant":
                msg_content = msg.get("content", "")
                if isinstance(msg_content, str):
                    assistant_text = msg_content

        score = sum(
            user_text.lower().count(kw) * 2 + assistant_text.lower().count(kw)
            for kw in keywords
        )
        if score > 0:
            scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)

    demonstrations = []
    for rank, (score, idx) in enumerate(scored[:num_examples]):
        sample = dataset[idx]
        user_text = ""
        assistant_text = ""
        for msg in sample.get("messages", []):
            if msg["role"] == "user":
                user_text = msg["content"]
            elif msg["role"] == "assistant":
                msg_content = msg.get("content", "")
                if isinstance(msg_content, str):
                    assistant_text = msg_content

        # Extract user_request portion
        req_match = re.search(r"<user_request>\s*(.*?)(?:<think>|$)", user_text, re.DOTALL)
        example_question = req_match.group(1).strip()[:400] if req_match else user_text[:400]
        example_answer = assistant_text[:800]

        demonstrations.append(
            f"### Example {rank + 1} ###\n"
            f"Question: {example_question}\n\n"
            f"Expert Plan:\n{example_answer}\n"
        )

    if not demonstrations:
        return "(No highly relevant examples found. Use your best judgment.)"

    return "\n---\n".join(demonstrations)


# ---- Module Prompts (with few-shot demonstrations slots) ----

MODULE_GENERATE_PROMPT = """You are a DSPy GeneratePlan module. Your signature is:
  {signature}

FEW-SHOT DEMONSTRATIONS (from expert dataset):
---
{demonstrations}
---

Now generate a plan for this question:
Question: {question}

Output a complete experimental plan with: Research Objective, Methodology, Experimental Design, Datasets & Metrics, Limitations."""


MODULE_CRITIQUE_PROMPT = """You are a DSPy CritiquePlan module. Your signature is:
  {signature}

Review the following experimental plan critically. Identify:
1. Logical gaps or missing components
2. Methodological weaknesses
3. Innovation gaps — where is it too incremental?
4. Reproducibility concerns

Plan to Critique:
---
{plan}
---

Output a structured critique. Be specific and actionable."""


MODULE_REFINE_PROMPT = """You are a DSPy RefinePlan module. Your signature is:
  {signature}

Original Plan:
---
{plan}
---

Critique Received:
---
{critique}
---

FEW-SHOT REFERENCE (how experts handle similar refinements):
---
{demonstrations}
---

Rewrite the plan by systematically addressing EVERY point in the critique.
Output the COMPLETE refined plan (not just changes)."""


MODULE_SCORE_PROMPT = """You are an expert scientific reviewer.

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
# Data & Utilities
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
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_input}],
                temperature=temperature, timeout=180,
            )
            print(f"    [API] {model} OK")
            return response.choices[0].message.content
        except Exception as e:
            print(f"    [API] error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(sleep_time)
                sleep_time = min(sleep_time * 2, 60)
    return "[ERROR] all retries failed"


# =========================================================
# LangGraph State
# =========================================================

class DSPyState(TypedDict):
    research_topic: str
    demonstrations: str              # Few-shot demonstration text
    plan: str                        # Current plan
    critique: str                    # Critique feedback
    scores: List[Dict]               # Score History
    optimize_round: int              # Current optimization round
    module_trace: List[Dict]         # Module call trace
    final_report: str


# =========================================================
# Build DSPy Pipeline
# =========================================================

def build_dspy_graph() -> StateGraph:
    """Build DSPy-style modular pipeline"""

    # =====================
    # Module 1: GeneratePlan (question -> plan)
    # =====================
    def module_generate(state: DSPyState) -> Dict:
        print("\n" + "=" * 60)
        print("[Module 1] GeneratePlan  |  question -> plan")
        print("=" * 60)

        question = state["research_topic"]
        demonstrations = state.get("demonstrations", "")

        plan = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are a DSPy GeneratePlan module. Use few-shot demonstrations to guide generation.",
            MODULE_GENERATE_PROMPT.format(
                signature=MODULE_SIGNATURES["GeneratePlan"],
                demonstrations=demonstrations,
                question=question,
            ),
            temperature=0.3
        )

        trace = list(state.get("module_trace", []))
        trace.append({"module": "GeneratePlan", "input": question[:100], "output_len": len(plan)})

        print(f"  -> Plan: {len(plan)} chars")
        return {"plan": plan, "module_trace": trace}

    # =====================
    # Module 2: CritiquePlan (plan -> critique)
    # =====================
    def module_critique(state: DSPyState) -> Dict:
        print("\n" + "=" * 60)
        print("[Module 2] CritiquePlan  |  plan -> critique")
        print("=" * 60)

        plan = state.get("plan", "")

        critique = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are a DSPy CritiquePlan module. Output a structured critique.",
            MODULE_CRITIQUE_PROMPT.format(
                signature=MODULE_SIGNATURES["CritiquePlan"],
                plan=plan,
            ),
            temperature=0.2
        )

        trace = list(state.get("module_trace", []))
        trace.append({"module": "CritiquePlan", "output_len": len(critique)})

        print(f"  -> Critique: {len(critique)} chars")
        for line in critique.split("\n")[:4]:
            short = line.strip()[:120]
            if short:
                print(f"     {short}")

        return {"critique": critique, "module_trace": trace}

    # =====================
    # Module 3: RefinePlan (plan, critique -> refined_plan)
    # =====================
    def module_refine(state: DSPyState) -> Dict:
        optimize_round = state.get("optimize_round", 0) + 1
        print("\n" + "=" * 60)
        print(f"[Module 3] RefinePlan  |  plan, critique -> refined_plan  (round {optimize_round}/{MAX_OPTIMIZE})")
        print("=" * 60)

        plan = state.get("plan", "")
        critique = state.get("critique", "")
        demonstrations = state.get("demonstrations", "")

        refined = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are a DSPy RefinePlan module. Address all critique points.",
            MODULE_REFINE_PROMPT.format(
                signature=MODULE_SIGNATURES["RefinePlan"],
                plan=plan,
                critique=critique,
                demonstrations=demonstrations,
            ),
            temperature=0.3
        )

        trace = list(state.get("module_trace", []))
        trace.append({"module": "RefinePlan", "round": optimize_round, "output_len": len(refined)})

        print(f"  -> Improved Plan: {len(refined)} chars")
        return {"plan": refined, "optimize_round": optimize_round, "module_trace": trace}

    # =====================
    # Module 4: ScorePlan (plan -> scores)
    # =====================
    def module_score(state: DSPyState) -> Dict:
        print("\n" + "=" * 60)
        print("[Module 4] ScorePlan  |  plan -> reliability, innovation")
        print("=" * 60)

        plan = state.get("plan", "")

        print("  [DeepSeek-chat judge] ...")
        ds_text = call_llm(deepseek_client, JUDGE_MODEL_DS, MODULE_SCORE_PROMPT, plan, temperature=0.0)
        ds_scores = parse_score(ds_text)
        print(f"    DS -> R={ds_scores['reliability']}, I={ds_scores['innovation']}")

        print("  [GPT-5.4 judge] ...")
        gpt_text = call_llm(gpt_client, JUDGE_MODEL_GPT, MODULE_SCORE_PROMPT, plan, temperature=0.0)
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
            "ds": ds_scores, "gpt": gpt_scores,
            "combined": combined, "verdict": verdict,
        })

        print(f"  Combined -> R={combined['reliability']}, I={combined['innovation']} -> {verdict}")
        return {"scores": scores}

    # =====================
    # Report node
    # =====================
    def report_node(state: DSPyState) -> Dict:
        print("\n" + "=" * 60)
        print("[Report] Generate DSPy Pipeline final report")
        print("=" * 60)

        topic = state["research_topic"]
        plan = state.get("plan", "")
        scores = state.get("scores", [])
        module_trace = state.get("module_trace", [])
        optimize_round = state.get("optimize_round", 0)

        display = plan[:5000] + "\n\n...(truncated)" if len(plan) > 5000 else plan

        lines = [
            "=" * 70,
            "          DSPy Pipeline Experimental Plan Report",
            "=" * 70, "",
            f"Research Topic: {topic[:300]}", "",
            "-" * 70,
            "DSPy Architecture: signature-driven modules + few-shot bootstrapping + module chain",
            f"Few-shot examples: {NUM_FEWSHOT}  |  Optimization rounds: {optimize_round}/{MAX_OPTIMIZE}",
            "-" * 70, "",
            "Module Signatures (DSPy Signatures):",
        ]
        for mod_name, sig in MODULE_SIGNATURES.items():
            lines.append(f"  {mod_name}: {sig}")

        # Module call trace
        lines += ["", "Module Call Trace (DSPy Trace):"]
        for t in module_trace:
            info = ", ".join(f"{k}={v}" for k, v in t.items() if k != "module")
            lines.append(f"  [{t['module']}] {info}")

        # Scoring
        if scores:
            lines += ["", "-" * 70, "Quality Assessment", "-" * 70]
            for s in scores:
                c = s["combined"]
                lines.append(
                    f"  R{s['round']}: DS(R={s['ds'].get('reliability',0)} I={s['ds'].get('innovation',0)}) "
                    f"GPT(R={s['gpt'].get('reliability',0)} I={s['gpt'].get('innovation',0)}) "
                    f"-> Combined R={c.get('reliability',0)} I={c.get('innovation',0)} [{s.get('verdict','')}]"
                )
            if len(scores) >= 2:
                delta_r = round(scores[-1]["combined"]["reliability"] - scores[0]["combined"]["reliability"], 1)
                delta_i = round(scores[-1]["combined"]["innovation"] - scores[0]["combined"]["innovation"], 1)
                lines.append(f"  DSPy optimization improvement:  Delta R = +{delta_r}  |  Delta I = +{delta_i}")

        lines += ["", "-" * 70, "Final Plan", "-" * 70, display or "Failed to generate", "", "=" * 70]
        return {"final_report": "\n".join(lines)}

    # =====================
    # Routing: decide whether to Critique+Refine again
    # =====================
    def route_after_score(state: DSPyState) -> Literal["critique", "report"]:
        optimize_round = state.get("optimize_round", 0)
        scores = state.get("scores", [])

        if optimize_round >= MAX_OPTIMIZE:
            print(f"  -> Max optimization rounds reached ({MAX_OPTIMIZE}), output report")
            return "report"
        if scores and scores[-1]["verdict"] in ("EXCELLENT", "GOOD"):
            print(f"  -> Score acceptable ({scores[-1]['verdict']}), output report")
            return "report"
        print(f"  -> Score insufficient, continuing Critique+Refine optimization...")
        return "critique"

    # =====================
    # Assemble Graph
    #   bootstrap -> generate -> critique -> refine -> score -> [critique -> refine -> score] -> report
    # =====================
    workflow = StateGraph(DSPyState)

    workflow.add_node("generate", module_generate)
    workflow.add_node("critique", module_critique)
    workflow.add_node("refine", module_refine)
    workflow.add_node("score", module_score)
    workflow.add_node("report", report_node)

    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "critique")
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
    return build_dspy_graph().compile(checkpointer=MemorySaver())


def run_agent(question: str, verbose: bool = True) -> Dict[str, Any]:
    # ---- 0. Bootstrap: get few-shot demonstrations ----
    dataset = load_dataset(sample_size=50)
    demonstrations = bootstrap_fewshot_examples(dataset, question, NUM_FEWSHOT)

    app = compile_agent()
    initial: DSPyState = {
        "research_topic": question, "demonstrations": demonstrations,
        "plan": "", "critique": "", "scores": [], "optimize_round": 0,
        "module_trace": [], "final_report": "",
    }
    config = {"configurable": {"thread_id": f"dspy-{int(time.time())}"}}

    print(f"\n{'*'*70}")
    print(f"DSPy Pipeline: {question[:200]}")
    print(f"{'*'*70}")
    print(f"Few-shot: {NUM_FEWSHOT} items  |  Modules: {' -> '.join(MODULE_SIGNATURES.keys())}")

    final_state = None
    for event in app.stream(initial, config, stream_mode="values"):
        final_state = event
        if verbose:
            for key in ["plan", "critique", "scores"]:
                val = event.get(key, "")
                if isinstance(val, list) and val:
                    print(f"  [{key}] {len(val)}  items")
                elif isinstance(val, str) and len(val) > 50:
                    print(f"  [{key}] {len(val)} chars")

    if final_state is None:
        return {"final_report": "[ERROR]", "scores": [], "module_trace": []}

    return {
        "final_report": final_state.get("final_report", ""),
        "plan": final_state.get("plan", ""),
        "scores": final_state.get("scores", []),
        "module_trace": final_state.get("module_trace", []),
    }


# =========================================================
# Cross-Sample Average Scores
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
            "ds_reliability_avg": _avg(d["ds_rel"]), "ds_innovation_avg": _avg(d["ds_inn"]),
            "gpt_reliability_avg": _avg(d["gpt_rel"]), "gpt_innovation_avg": _avg(d["gpt_inn"]),
            "combined_reliability_avg": _avg(d["combined_rel"]), "combined_innovation_avg": _avg(d["combined_inn"]),
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
    print(rf"""
╔══════════════════════════════════════════════════════════════╗
║    DSPy Pipeline Experimental Plan Generator                  ║
║    Modules: GeneratePlan -> CritiquePlan -> RefinePlan -> Score ║
║    Few-shot: {NUM_FEWSHOT}  |  Optimize: max {MAX_OPTIMIZE} rounds  |  Generator: DeepSeek V4 Pro ║
╚══════════════════════════════════════════════════════════════╝

Commands:
  /run <question>    -- Generate experiment plan for research question (DSPy Pipeline)
  /batch [N]     -- Batch Test (default 3)
  /demo          -- use built-in demo
  /quit          -- Exit
""")

    while True:
        try:
            ui = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break
        if not ui: continue
        if ui.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye!"); break
        if ui.lower() == "/demo":
            ui = "/run Design an experiment to evaluate whether large language models can self-improve through recursive self-critique without external feedback"

        if ui.lower().startswith("/run "):
            question = ui[5:].strip()
            result = run_agent(question)
            print("\n" + result["final_report"])
            out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f"dspy_result_{int(time.time())}.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n[ResultSaved] {out}")

        elif ui.lower().startswith("/batch"):
            parts = ui.split()
            n = int(parts[1]) if len(parts) > 1 else 3
            batch_test(n)
        else:
            print("Unknown command. Use /run <question> to start.")


def batch_test(n: int = 3):
    dataset = load_dataset(sample_size=n)
    print(f"\n{'#'*70}")
    print(f"  DSPy Pipeline Batch Test: {n} samples")
    print(f"  Few-shot={NUM_FEWSHOT}  |  MaxOptimize={MAX_OPTIMIZE}")
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
    print("         DSPy Pipeline Batch Test -- Cross-Sample Average ScoresSummary")
    print("=" * 70)

    for r in sorted(batch_avg["rounds"].keys()):
        info = batch_avg["rounds"][r]
        print(f"\n  [R{r}]  (Covering {info['count']}/{n} samples)")
        print(f"    DeepSeek Judges:  R = {info['ds_reliability_avg']}  |  I = {info['ds_innovation_avg']}")
        print(f"    GPT Judges:      R = {info['gpt_reliability_avg']}  |  I = {info['gpt_innovation_avg']}")
        print(f"    * CombinedAvg:    R = {info['combined_reliability_avg']}  |  I = {info['combined_innovation_avg']}")

    overall = batch_avg["overall"]
    print(f"\n  {'-'*60}")
    print(f"  [Overall Total]  (Total {overall['total_scores']} scoring events)")
    print(f"    DeepSeek Judges:  R = {overall['ds_reliability_avg']}  |  I = {overall['ds_innovation_avg']}")
    print(f"    GPT Judges:      R = {overall['gpt_reliability_avg']}  |  I = {overall['gpt_innovation_avg']}")
    print(f"    * CombinedAvg:    R = {overall['combined_reliability_avg']}  |  I = {overall['combined_innovation_avg']}")
    print(f"\n{'='*70}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"dspy_batch_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"generator": GENERATOR_MODEL, "judges": [JUDGE_MODEL_DS, JUDGE_MODEL_GPT],
                       "samples": n, "modules": list(MODULE_SIGNATURES.keys()),
                       "fewshot_count": NUM_FEWSHOT, "max_optimize": MAX_OPTIMIZE},
            "batch_averages": batch_avg, "samples": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[ResultSaved] {out}")


if __name__ == "__main__":
    interactive_loop()

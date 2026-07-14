"""
Experiment_2/reflexion.py
Reflexion Experiment Plan Generator
Generate -> Score -> Reflect on failure (verbal reinforcement signal) -> Regenerate with lessons learned
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

MAX_TRIALS = 3  # max retries

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gen_lcot", "data", "train.jsonl"
)


# =========================================================
# Reflexion Prompts
# =========================================================

# Actor: Generate experimental plan (may carry prior reflections)
ACTOR_PROMPT_FIRST = """You are an expert AI research scientist. Generate a complete experimental plan
for the following research question.

Research Question:
{question}

Generate a comprehensive, rigorous, and innovative experimental plan including:
- Research Objective
- Proposed Methodology
- Experimental Design
- Datasets & Evaluation Metrics
- Expected Outcomes & Limitations

This is your FIRST attempt. Be thorough and creative."""

ACTOR_PROMPT_RETRY = """You are an expert AI research scientist. You previously attempted to design
an experimental plan but it was rated as unsatisfactory. You have reflected on what went wrong.

Research Question:
{question}

PREVIOUS ATTEMPTS & REFLECTIONS:
---
{reflection_memory}
---

CRITICAL INSTRUCTIONS:
1. Read ALL the reflections above carefully — they identify specific failures in prior attempts
2. DO NOT repeat the same mistakes
3. Apply the lessons learned from each reflection
4. Generate a COMPLETELY NEW plan that addresses every issue raised

Generate a complete experimental plan including:
- Research Objective
- Proposed Methodology
- Experimental Design
- Datasets & Evaluation Metrics
- Expected Outcomes & Limitations

Be specific about how this plan differs from and improves upon previous attempts."""


# Evaluator: Scoring (reusable)
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


# Self-Reflection: Reflect after failure — verbal reinforcement signal
REFLECTION_PROMPT = """You are a self-reflective AI research scientist. Your previous experimental plan
was rated as NEEDS IMPROVEMENT. Your task is NOT to fix the plan, but to REFLECT on WHY it failed
and derive GENERAL LESSONS for future attempts.

Research Question:
{question}

Failed Plan Summary:
{plan_summary}

Scores:
  DeepSeek Judges: Reliability={ds_rel}, Innovation={ds_inn}
  GPT Judges:      Reliability={gpt_rel}, Innovation={gpt_inn}
  Combined:       Reliability={combined_rel}, Innovation={combined_inn}

Previous Reflections (if any):
{previous_reflections}

Please analyze:

1. ROOT CAUSES: What fundamental flaws led to these low scores?
   - Was the methodology insufficiently rigorous? Why?
   - Was the innovation lacking? What made it incremental?
   - Were there logical gaps or missing components?

2. PATTERNS: If this is not the first attempt, are there recurring issues?

3. LESSONS LEARNED: What specific, actionable principles should guide the NEXT attempt?
   - "Next time, I should..."
   - "I must avoid..."
   - "I need to strengthen..."

4. STRATEGY CHANGE: How should the NEXT attempt differ fundamentally from this one?

Output a structured reflection. Be honest, specific, and actionable.
These reflections will be used as a "verbal reinforcement signal" to improve future generations."""


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

class ReflexionState(TypedDict):
    research_topic: str
    plan: str                             # Current Plan
    trial: int                            # Current trial number (1-based)
    scores: List[Dict]                    # Scores for all trials
    reflections: List[Dict]               # [{trial, reflection, scores_at_time}]
    reflection_memory: str                # Accumulated reflection text (for Actor retry)
    final_report: str


# =========================================================
# Build Reflexion Pipeline
# =========================================================

def build_reflexion_graph() -> StateGraph:

    # =====================
    # Node 1: Actor — Generate Plan
    # =====================
    def actor_node(state: ReflexionState) -> Dict:
        trial = state.get("trial", 0) + 1
        question = state["research_topic"]
        reflection_memory = state.get("reflection_memory", "")

        if trial == 1:
            print("\n" + "=" * 60)
            print(f"[Actor] Trial {trial} — Initial Generation")
            print("=" * 60)
            plan = call_llm(
                deepseek_client, GENERATOR_MODEL,
                "You are an expert AI research scientist making your first attempt.",
                ACTOR_PROMPT_FIRST.format(question=question),
                temperature=0.3
            )
        else:
            print("\n" + "=" * 60)
            print(f"[Actor] Trial {trial} — Regenerate based on reflection")
            print("=" * 60)
            plan = call_llm(
                deepseek_client, GENERATOR_MODEL,
                "You are an expert learning from past failures. Apply the reflections.",
                ACTOR_PROMPT_RETRY.format(
                    question=question,
                    reflection_memory=reflection_memory,
                ),
                temperature=0.4  # slightly higher temperature to encourage innovation
            )

        print(f"  -> Plan: {len(plan)} chars")
        return {"plan": plan, "trial": trial}

    # =====================
    # Node 2: Evaluator — Dual-Judge Scoring
    # =====================
    def evaluator_node(state: ReflexionState) -> Dict:
        trial = state.get("trial", 1)
        plan = state.get("plan", "")

        print("\n" + "=" * 60)
        print(f"[Evaluator] Trial {trial} Scoring")
        print("=" * 60)

        print("  [Judge1] DeepSeek-chat ...")
        ds_text = call_llm(deepseek_client, JUDGE_MODEL_DS, SCORING_PROMPT, plan, temperature=0.0)
        ds_scores = parse_score(ds_text)
        print(f"    DS -> R={ds_scores['reliability']}, I={ds_scores['innovation']}")

        print("  [Judge2] GPT-5.4 ...")
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
            "trial": trial,
            "ds": ds_scores, "gpt": gpt_scores,
            "combined": combined, "verdict": verdict,
        })

        print(f"  Combined -> R={combined['reliability']}, I={combined['innovation']} -> {verdict}")
        return {"scores": scores}

    # =====================
    # Node 3: Self-Reflection — Reflect on Failure Causes
    # =====================
    def reflection_node(state: ReflexionState) -> Dict:
        trial = state.get("trial", 1)
        question = state["research_topic"]
        plan = state.get("plan", "")
        scores = state.get("scores", [])
        reflections = list(state.get("reflections", []))

        # Current scores
        current = scores[-1] if scores else {}
        ds = current.get("ds", {})
        gpt = current.get("gpt", {})
        combined = current.get("combined", {})

        # Previous reflections
        prev_reflections_text = "\n\n---\n\n".join(
            f"## Reflection after Trial {r['trial']}:\n{r['reflection']}"
            for r in reflections
        ) if reflections else "(This is the first reflection)"

        print("\n" + "=" * 60)
        print(f"[Self-Reflection] Analyzing Trial {trial} failure — Verbal Reinforcement")
        print("=" * 60)

        reflection = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are a self-reflective AI. Analyze failure and derive actionable lessons.",
            REFLECTION_PROMPT.format(
                question=question,
                plan_summary=plan[:2000],
                ds_rel=ds.get("reliability", 0), ds_inn=ds.get("innovation", 0),
                gpt_rel=gpt.get("reliability", 0), gpt_inn=gpt.get("innovation", 0),
                combined_rel=combined.get("reliability", 0), combined_inn=combined.get("innovation", 0),
                previous_reflections=prev_reflections_text,
            ),
            temperature=0.3
        )

        # Append to reflection list
        reflections.append({
            "trial": trial,
            "reflection": reflection,
            "scores_at_time": combined,
        })

        # Build accumulated reflection memory (for Actor next use)
        memory_parts = []
        for r in reflections:
            c = r.get("scores_at_time", {})
            memory_parts.append(
                f"=== Trial {r['trial']} ===\n"
                f"Scores: R={c.get('reliability','?')}/100, I={c.get('innovation','?')}/100\n"
                f"Reflection:\n{r['reflection']}"
            )
        reflection_memory = "\n\n---\n\n".join(memory_parts)

        print(f"  -> Reflection: {len(reflection)} chars")
        # Print reflection key points
        for line in reflection.split("\n")[:6]:
            short = line.strip()[:120]
            if short:
                print(f"     {short}")

        return {"reflections": reflections, "reflection_memory": reflection_memory}

    # =====================
    # Node 4: Report
    # =====================
    def report_node(state: ReflexionState) -> Dict:
        print("\n" + "=" * 60)
        print("[Report] Generate Reflexion Final Report")
        print("=" * 60)

        topic = state["research_topic"]
        plan = state.get("plan", "")
        scores = state.get("scores", [])
        reflections = state.get("reflections", [])
        trial = state.get("trial", 1)

        display_plan = plan[:5000] + "\n\n...(truncated)" if len(plan) > 5000 else plan

        lines = [
            "=" * 70,
            "          Reflexion Experiment Plan Report",
            "=" * 70, "",
            f"Research Topic: {topic[:300]}", "",
            "-" * 70,
            f"Reflexion Workflow: Actor -> Evaluator -> [Self-Reflection -> Actor] x {len(reflections)}",
            f"Total trials: {trial}",
            "-" * 70,
        ]

        # Scores for all trials
        if scores:
            lines += ["", "Score History:", ""]
            for s in scores:
                c = s["combined"]
                lines.append(
                    f"  Trial {s['trial']}: DS(R={s['ds'].get('reliability',0)} I={s['ds'].get('innovation',0)}) "
                    f"GPT(R={s['gpt'].get('reliability',0)} I={s['gpt'].get('innovation',0)}) "
                    f"-> Combined R={c.get('reliability',0)} I={c.get('innovation',0)} [{s.get('verdict','')}]"
                )

        # Reflection summary
        if reflections:
            lines += ["", "-" * 70, "Reflection History (Verbal Reinforcement Signal)", "-" * 70]
            for r in reflections:
                lines.append(f"  Trial {r['trial']} Reflection: {r['reflection'][:200]}...")

        # Improvement Delta
        if len(scores) >= 2:
            first = scores[0]["combined"]
            last = scores[-1]["combined"]
            delta_r = round(last["reliability"] - first["reliability"], 1)
            delta_i = round(last["innovation"] - first["innovation"], 1)
            lines += [
                "", "-" * 70,
                f"Reflexion improvement:  Delta R = +{delta_r}  |  Delta I = +{delta_i}",
                f"  Trial 1 -> Trial {trial}",
                "-" * 70,
            ]

        lines += ["", "Final Plan:", display_plan or "Failed to generate", "", "=" * 70]
        return {"final_report": "\n".join(lines)}

    # =====================
    # Route: Continue or End?
    # =====================
    def route_after_eval(state: ReflexionState) -> Literal["reflect", "report"]:
        trial = state.get("trial", 1)
        scores = state.get("scores", [])

        if trial >= MAX_TRIALS:
            print(f"  -> Max trials reached ({MAX_TRIALS}), output report")
            return "report"

        if scores and scores[-1]["verdict"] in ("EXCELLENT", "GOOD"):
            print(f"  -> Score acceptable ({scores[-1]['verdict']}), output report")
            return "report"

        print(f"  -> Score insufficient ({scores[-1]['verdict']}), entering Self-Reflection...")
        return "reflect"

    # =====================
    # Assemble Graph
    #   actor -> evaluator -> [NEEDS_IMPROV & trials left?]
    #      ^                     |
    #      |--- reflect <-------|  YES
    #      |
    #      NO -> report -> END
    # =====================
    workflow = StateGraph(ReflexionState)

    workflow.add_node("actor", actor_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("reflect", reflection_node)
    workflow.add_node("report", report_node)

    workflow.set_entry_point("actor")
    workflow.add_edge("actor", "evaluator")

    workflow.add_conditional_edges(
        "evaluator", route_after_eval,
        {"reflect": "reflect", "report": "report"},
    )
    workflow.add_edge("reflect", "actor")  # After reflection, regenerate
    workflow.add_edge("report", END)

    return workflow


# =========================================================
# Compile & Run
# =========================================================

def compile_agent():
    return build_reflexion_graph().compile(checkpointer=MemorySaver())


def run_agent(question: str, verbose: bool = True) -> Dict[str, Any]:
    app = compile_agent()
    initial: ReflexionState = {
        "research_topic": question, "plan": "", "trial": 0,
        "scores": [], "reflections": [], "reflection_memory": "", "final_report": "",
    }
    config = {"configurable": {"thread_id": f"refl-{int(time.time())}"}}

    print(f"\n{'*'*70}")
    print(f"Reflexion: {question[:200]}")
    print(f"{'*'*70}")
    print(f"Workflow: Actor -> Evaluator -> [Self-Reflection -> Actor] x N (max {MAX_TRIALS} trials)")

    final_state = None
    for event in app.stream(initial, config, stream_mode="values"):
        final_state = event
        if verbose:
            for key in ["plan", "reflections", "reflection_memory"]:
                val = event.get(key, "")
                if isinstance(val, list) and val:
                    print(f"  [{key}] {len(val)}  items")
                elif isinstance(val, str) and len(val) > 50:
                    print(f"  [{key}] {len(val)} chars")

    if final_state is None:
        return {"final_report": "[ERROR]", "scores": [], "reflections": []}

    return {
        "final_report": final_state.get("final_report", ""),
        "plan": final_state.get("plan", ""),
        "scores": final_state.get("scores", []),
        "reflections": final_state.get("reflections", []),
        "trial": final_state.get("trial", 1),
    }


# =========================================================
# Cross-Sample Average Scores
# =========================================================

def _avg(lst: List[float]) -> float:
    return round(sum(lst) / len(lst), 1) if lst else 0.0


def compute_batch_averages(all_results: List[Dict]) -> Dict:
    # Reflexion may have multi-trial scores — group by trial
    by_trial: Dict[int, Dict[str, List[float]]] = {}
    for result in all_results:
        for s in result.get("scores", []):
            t = s.get("trial", 1)
            if t not in by_trial:
                by_trial[t] = {"ds_rel": [], "ds_inn": [], "gpt_rel": [], "gpt_inn": [],
                               "combined_rel": [], "combined_inn": []}
            by_trial[t]["ds_rel"].append(s["ds"].get("reliability", 0))
            by_trial[t]["ds_inn"].append(s["ds"].get("innovation", 0))
            by_trial[t]["gpt_rel"].append(s["gpt"].get("reliability", 0))
            by_trial[t]["gpt_inn"].append(s["gpt"].get("innovation", 0))
            by_trial[t]["combined_rel"].append(s["combined"].get("reliability", 0))
            by_trial[t]["combined_inn"].append(s["combined"].get("innovation", 0))

    trials_summary = {}
    all_ds_rel, all_ds_inn, all_gpt_rel, all_gpt_inn = [], [], [], []
    all_comb_rel, all_comb_inn = [], []

    for t in sorted(by_trial.keys()):
        d = by_trial[t]
        trials_summary[t] = {
            "count": len(d["ds_rel"]),
            "ds_reliability_avg": _avg(d["ds_rel"]), "ds_innovation_avg": _avg(d["ds_inn"]),
            "gpt_reliability_avg": _avg(d["gpt_rel"]), "gpt_innovation_avg": _avg(d["gpt_inn"]),
            "combined_reliability_avg": _avg(d["combined_rel"]), "combined_innovation_avg": _avg(d["combined_inn"]),
        }
        all_ds_rel.extend(d["ds_rel"]); all_ds_inn.extend(d["ds_inn"])
        all_gpt_rel.extend(d["gpt_rel"]); all_gpt_inn.extend(d["gpt_inn"])
        all_comb_rel.extend(d["combined_rel"]); all_comb_inn.extend(d["combined_inn"])

    return {
        "trials": trials_summary,
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
║    Reflexion Experiment Plan Generator                         ║
║    Workflow: Actor -> Evaluator -> Self-Reflection -> Actor ...  ║
║    Generator: DeepSeek V4 Pro | Judges: DS-chat + GPT-5.4    ║
╚══════════════════════════════════════════════════════════════╝

Commands:
  /run <question> -- Generate experimental plan for research question (Reflexion)
  /batch [N]      -- Batch test (default 3)
  /demo           -- Use built-in demo
  /quit           -- Exit
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
                               f"refl_result_{int(time.time())}.json")
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
    dataset = load_dataset(sample_size=n)
    print(f"\n{'#'*70}")
    print(f"  Reflexion Batch Test: {n} samples (max {MAX_TRIALS} trials each)")
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
            "trial": result.get("trial", 1),
        })
        for s in result.get("scores", []):
            c = s["combined"]
            print(f"  -> Trial {s['trial']}: DS(R={s['ds'].get('reliability',0)} I={s['ds'].get('innovation',0)}) "
                  f"GPT(R={s['gpt'].get('reliability',0)} I={s['gpt'].get('innovation',0)}) "
                  f"Combined(R={c.get('reliability',0)} I={c.get('innovation',0)}) [{s.get('verdict','')}]")

    batch_avg = compute_batch_averages(all_results)
    print("\n" + "=" * 70)
    print("         Reflexion Batch Test -- Cross-Sample Average Summary")
    print("=" * 70)

    for t in sorted(batch_avg["trials"].keys()):
        info = batch_avg["trials"][t]
        print(f"\n  [Trial {t}]  (Covering {info['count']}/{n} samples)")
        print(f"    DeepSeek Judges:  R = {info['ds_reliability_avg']}  |  I = {info['ds_innovation_avg']}")
        print(f"    GPT Judges:      R = {info['gpt_reliability_avg']}  |  I = {info['gpt_innovation_avg']}")
        print(f"    * CombinedAvg:    R = {info['combined_reliability_avg']}  |  I = {info['combined_innovation_avg']}")

    # Improvement Delta
    trials = sorted(batch_avg["trials"].keys())
    if len(trials) >= 2:
        first = batch_avg["trials"][trials[0]]
        last = batch_avg["trials"][trials[-1]]
        delta_r = round(last["combined_reliability_avg"] - first["combined_reliability_avg"], 1)
        delta_i = round(last["combined_innovation_avg"] - first["combined_innovation_avg"], 1)
        print(f"\n  Reflexion Improvement Delta:  Delta R = +{delta_r}  |  Delta I = +{delta_i}")
        print(f"    (Trial {trials[0]} -> Trial {trials[-1]})")

    overall = batch_avg["overall"]
    print(f"\n  {'-'*60}")
    print(f"  [Overall Total]  (Total {overall['total_scores']} scoring events)")
    print(f"    DeepSeek Judges:  R = {overall['ds_reliability_avg']}  |  I = {overall['ds_innovation_avg']}")
    print(f"    GPT Judges:      R = {overall['gpt_reliability_avg']}  |  I = {overall['gpt_innovation_avg']}")
    print(f"    * CombinedAvg:    R = {overall['combined_reliability_avg']}  |  I = {overall['combined_innovation_avg']}")
    print(f"\n{'='*70}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"refl_batch_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"generator": GENERATOR_MODEL, "judges": [JUDGE_MODEL_DS, JUDGE_MODEL_GPT],
                       "samples": n, "max_trials": MAX_TRIALS},
            "batch_averages": batch_avg, "samples": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[ResultSaved] {out}")


if __name__ == "__main__":
    interactive_loop()

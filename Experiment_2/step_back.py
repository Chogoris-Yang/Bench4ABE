"""
Experiment_2/step_back.py
Step-Back Experimental Plan Generator
First Step Back to ask fundamental principle questions -> Obtain General Principles -> Return to original question and apply them
Generator: DeepSeek V4 Pro  |  Judge: DeepSeek-chat + GPT-5.4
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

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gen_lcot", "data", "train.jsonl"
)


# =========================================================
# Step-Back Prompts
# =========================================================

# Stage 1: Step Back — Abstract higher-level principle questions from the specific question
STEP_BACK_PROMPT = """You are an expert at abstract reasoning. Given a specific research question,
your task is to STEP BACK and identify the deeper, more fundamental questions behind it.

Original Research Question:
{question}

Instructions:
1. Identify the CORE SCIENTIFIC DOMAIN this question belongs to
2. Identify the UNDERLYING PRINCIPLES and concepts at play
3. Formulate 2-3 STEP-BACK QUESTIONS that are more abstract and fundamental —
   questions whose answers would provide general principles applicable to the original question

Step-back questions should be about:
- General methodological principles in this domain
- Fundamental evaluation strategies for this type of problem
- Universal experimental design principles
- Cross-domain analogies and insights

Output format (JSON):
{{
  "domain": "the core scientific domain",
  "underlying_principles": ["principle1", "principle2", ...],
  "step_back_questions": [
    {{"id": 1, "question": "A more abstract, principled question...", "relevance": "how this relates to the original"}},
    ...
  ]
}}

Explain your reasoning briefly, then output the JSON."""

# Stage 2: Answer Step-Back questions — Obtain General Principles
ANSWER_STEP_BACK_PROMPT = """You are an expert in {domain}. Answer the following foundational questions
with deep, principled reasoning.

Step-Back Questions:
{step_back_questions}

Original Question (for context only — answer the step-back questions, NOT this):
{original_question}

Instructions:
- Provide comprehensive, principle-based answers
- Draw on established best practices and theoretical foundations
- Include concrete examples where helpful
- These answers will be used as guiding principles for solving the original problem

Answer each step-back question thoroughly."""

# Stage 3: Step Forward — Apply the obtained General Principles to answer the original question
STEP_FORWARD_PROMPT = """You are an expert AI research scientist. You have just derived general
principles by answering abstract, foundational questions. Now APPLY those principles to solve
the original, specific research problem.

GENERAL PRINCIPLES (from step-back reasoning):
---
{principles}
---

Original Research Question:
{original_question}

Instructions:
1. Use the general principles above as your guiding framework
2. Apply them to design a complete, rigorous experimental plan for the original question
3. For each design decision, reference which principle guided it
4. If the principles suggest multiple valid approaches, choose the most appropriate one

Generate a complete experimental plan including:
- Research Objective (grounded in the principles)
- Proposed Methodology (applying the principles)
- Experimental Design (following the evaluation principles)
- Datasets & Metrics (justified by the principles)
- Expected Outcomes & Limitations

Be specific to the original question — do NOT remain abstract. Apply the principles concretely."""

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

class StepBackState(TypedDict):
    research_topic: str
    domain: str
    underlying_principles: List[str]
    step_back_questions: List[Dict]       # [{id, question, relevance}]
    principles_answer: str                # Answer to Step-Back questions (General Principles)
    plan: str                             # Plan generated after Step Forward
    scores: List[Dict]
    final_report: str


# =========================================================
# Build Step-Back Pipeline
# =========================================================

def build_step_back_graph() -> StateGraph:

    # =====================
    # Node 1: Step Back — Abstract higher-level questions
    # =====================
    def step_back_node(state: StepBackState) -> Dict:
        print("\n" + "=" * 60)
        print("[Stage 1] Step Back — Abstract the essential question")
        print("=" * 60)

        question = state["research_topic"]
        raw = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are an expert at abstract reasoning. Step back from the specific question.",
            STEP_BACK_PROMPT.format(question=question),
            temperature=0.3
        )

        domain = ""
        principles = []
        step_back_qs = []

        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                domain = data.get("domain", "")
                principles = data.get("underlying_principles", [])
                step_back_qs = data.get("step_back_questions", [])
        except json.JSONDecodeError:
            print("  [!] JSON parse failed, using defaults")
            domain = "machine learning research methodology"
            principles = [
                "Rigorous experimental design requires controlled ablation studies",
                "Evaluation metrics must align with the research objective",
                "Methodological innovation should address a clear gap in existing approaches"
            ]
            step_back_qs = [
                {"id": 1, "question": "What are the fundamental principles for designing rigorous experiments in this domain?",
                 "relevance": "Ensuring the experimental plan is sound"},
                {"id": 2, "question": "How should evaluation metrics be chosen to genuinely measure progress?",
                 "relevance": "Selecting appropriate benchmarks and metrics"},
                {"id": 3, "question": "What makes a methodology genuinely novel vs. incremental?",
                 "relevance": "Ensuring innovation rather than minor variation"},
            ]

        print(f"  Domain: {domain}")
        print(f"  Fundamental Principles: {len(principles)} items")
        print(f"  Step-Back Questions: {len(step_back_qs)} items")
        for sq in step_back_qs:
            print(f"    Q{sq['id']}: {sq['question'][:80]}...")

        return {
            "domain": domain,
            "underlying_principles": principles,
            "step_back_questions": step_back_qs,
        }

    # =====================
    # Node 2: Answer Step-Back — Answer the Step-Back questions
    # =====================
    def answer_step_back_node(state: StepBackState) -> Dict:
        print("\n" + "=" * 60)
        print("[Stage 2] Answer Step-Back — Answer the abstract Step-Back questions, obtain General Principles")
        print("=" * 60)

        domain = state.get("domain", "this research domain")
        step_back_qs = state.get("step_back_questions", [])
        original = state["research_topic"]

        qs_text = "\n\n".join(
            f"Q{sq['id']}: {sq['question']}\n(Relevance to original: {sq.get('relevance', 'N/A')})"
            for sq in step_back_qs
        )

        principles = call_llm(
            deepseek_client, GENERATOR_MODEL,
            f"You are an expert in {domain}. Answer foundational questions with deep principles.",
            ANSWER_STEP_BACK_PROMPT.format(
                domain=domain,
                step_back_questions=qs_text,
                original_question=original,
            ),
            temperature=0.3
        )

        print(f"  -> General principles answer: {len(principles)} chars")
        # Print summary
        for line in principles.split("\n")[:8]:
            short = line.strip()[:120]
            if short:
                print(f"     {short}")

        return {"principles_answer": principles}

    # =====================
    # Node 3: Step Forward — Use principles to answer the original question
    # =====================
    def step_forward_node(state: StepBackState) -> Dict:
        print("\n" + "=" * 60)
        print("[Stage 3] Step Forward — Apply General Principles to the original question")
        print("=" * 60)

        original = state["research_topic"]
        principles = state.get("principles_answer", "")

        plan = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are an expert research scientist applying general principles to a specific problem.",
            STEP_FORWARD_PROMPT.format(
                principles=principles,
                original_question=original,
            ),
            temperature=0.3
        )

        print(f"  -> Final Plan: {len(plan)} chars")
        return {"plan": plan}

    # =====================
    # Node 4: Dual-Judge Scoring
    # =====================
    def score_node(state: StepBackState) -> Dict:
        print("\n" + "=" * 60)
        print("[Scoring] Dual-Judge evaluation")
        print("=" * 60)

        plan = state.get("plan", "")

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

        scores = [{"round": 1, "ds": ds_scores, "gpt": gpt_scores, "combined": combined, "verdict": verdict}]
        print(f"  Combined -> R={combined['reliability']}, I={combined['innovation']} -> {verdict}")
        return {"scores": scores}

    # =====================
    # Node 5: Report
    # =====================
    def report_node(state: StepBackState) -> Dict:
        print("\n" + "=" * 60)
        print("[Report] Generate Step-Back Final Report")
        print("=" * 60)

        topic = state["research_topic"]
        plan = state.get("plan", "")
        scores = state.get("scores", [])
        domain = state.get("domain", "")
        principles = state.get("underlying_principles", [])
        step_back_qs = state.get("step_back_questions", [])

        display_plan = plan[:5000] + "\n\n...(truncated)" if len(plan) > 5000 else plan

        lines = [
            "=" * 70,
            "          Step-Back Experimental Plan Report",
            "=" * 70, "",
            f"Research Topic: {topic[:300]}", "",
            "-" * 70,
            f"Step-Back Workflow: Step Back (abstract) -> Obtain General Principles -> Apply to original question",
            "-" * 70, "",
            f"Identified Domain: {domain}",
            "",
            "Step-Back Questions (more abstract/essential):",
        ]
        for sq in step_back_qs:
            lines.append(f"  Q{sq['id']}: {sq['question'][:100]}...")

        lines += ["", f"Extracted Fundamental Principles: {len(principles)} items"]
        for i, p in enumerate(principles[:5]):
            lines.append(f"  {i+1}. {p[:120]}")

        lines += ["", "-" * 70, "Final Plan (based on General Principles)", "-" * 70, display_plan or "Failed to generate"]

        if scores:
            lines += ["", "-" * 70, "Quality Assessment", "-" * 70]
            for s in scores:
                c = s["combined"]
                lines.append(
                    f"  DS(R={s['ds'].get('reliability',0)} I={s['ds'].get('innovation',0)}) "
                    f"GPT(R={s['gpt'].get('reliability',0)} I={s['gpt'].get('innovation',0)}) "
                    f"-> Combined R={c.get('reliability',0)} I={c.get('innovation',0)} [{s.get('verdict','')}]"
                )

        lines += ["", "=" * 70]
        return {"final_report": "\n".join(lines)}

    # =====================
    # Assemble Graph (Pure linear DAG, no Conditional Branch)
    # =====================
    # =====================
    workflow = StateGraph(StepBackState)

    workflow.add_node("step_back", step_back_node)
    workflow.add_node("answer_step_back", answer_step_back_node)
    workflow.add_node("step_forward", step_forward_node)
    workflow.add_node("score", score_node)
    workflow.add_node("report", report_node)

    workflow.set_entry_point("step_back")
    workflow.add_edge("step_back", "answer_step_back")
    workflow.add_edge("answer_step_back", "step_forward")
    workflow.add_edge("step_forward", "score")
    workflow.add_edge("score", "report")
    workflow.add_edge("report", END)

    return workflow


# =========================================================
# Compile & Run
# =========================================================

def compile_agent():
    return build_step_back_graph().compile(checkpointer=MemorySaver())


def run_agent(question: str, verbose: bool = True) -> Dict[str, Any]:
    app = compile_agent()
    initial: StepBackState = {
        "research_topic": question, "domain": "", "underlying_principles": [],
        "step_back_questions": [], "principles_answer": "", "plan": "",
        "scores": [], "final_report": "",
    }
    config = {"configurable": {"thread_id": f"sb-{int(time.time())}"}}

    print(f"\n{'*'*70}")
    print(f"Step-Back: {question[:200]}")
    print(f"{'*'*70}")
    print("Workflow: Step Back (abstract) -> Answer (principles) -> Step Forward (apply) -> Scoring -> Report")

    final_state = None
    for event in app.stream(initial, config, stream_mode="values"):
        final_state = event
        if verbose:
            for key in ["step_back_questions", "principles_answer", "plan"]:
                val = event.get(key, "")
                if isinstance(val, list) and val:
                    print(f"  [{key}] {len(val)}  items")
                elif isinstance(val, str) and len(val) > 50:
                    print(f"  [{key}] {len(val)} chars")

    if final_state is None:
        return {"final_report": "[ERROR]", "scores": [], "plan": ""}

    return {
        "final_report": final_state.get("final_report", ""),
        "plan": final_state.get("plan", ""),
        "scores": final_state.get("scores", []),
        "domain": final_state.get("domain", ""),
        "step_back_questions": final_state.get("step_back_questions", []),
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
    print(r"""
╔══════════════════════════════════════════════════════════════╗
║    Step-Back Experimental Plan Generator                                  ║
║    Workflow: Step Back (abstract) -> Answer (principles) -> Step Forward (apply)    ║
║    Generator: DeepSeek V4 Pro | Judges: DS-chat + GPT-5.4          ║
╚══════════════════════════════════════════════════════════════╝

Commands:
  /run <question>    -- Generate experiment plan for research question (Step-Back)
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
                               f"sb_result_{int(time.time())}.json")
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
    print(f"  Step-Back Batch Test: {n} samples")
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
            print(f"  -> DS(R={s['ds'].get('reliability',0)} I={s['ds'].get('innovation',0)}) "
                  f"GPT(R={s['gpt'].get('reliability',0)} I={s['gpt'].get('innovation',0)}) "
                  f"Combined(R={c.get('reliability',0)} I={c.get('innovation',0)}) [{s.get('verdict','')}]")

    batch_avg = compute_batch_averages(all_results)
    print("\n" + "=" * 70)
    print("         Step-Back Batch Test -- Cross-Sample Average Summary")
    print("=" * 70)

    for r in sorted(batch_avg["rounds"].keys()):
        info = batch_avg["rounds"][r]
        print(f"\n  [Scoring]  (Covering {info['count']}/{n} samples)")
        print(f"    DeepSeek Judge:  R = {info['ds_reliability_avg']}  |  I = {info['ds_innovation_avg']}")
        print(f"    GPT Judge:       R = {info['gpt_reliability_avg']}  |  I = {info['gpt_innovation_avg']}")
        print(f"    * Combined Avg:  R = {info['combined_reliability_avg']}  |  I = {info['combined_innovation_avg']}")

    overall = batch_avg["overall"]
    print(f"\n  {'-'*60}")
    print(f"  [Overall Total]  (Total {overall['total_scores']} scoring events)")
    print(f"    DeepSeek Judge:  R = {overall['ds_reliability_avg']}  |  I = {overall['ds_innovation_avg']}")
    print(f"    GPT Judge:       R = {overall['gpt_reliability_avg']}  |  I = {overall['gpt_innovation_avg']}")
    print(f"    * Combined Avg:  R = {overall['combined_reliability_avg']}  |  I = {overall['combined_innovation_avg']}")
    print(f"\n{'='*70}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"sb_batch_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"generator": GENERATOR_MODEL, "judges": [JUDGE_MODEL_DS, JUDGE_MODEL_GPT], "samples": n},
            "batch_averages": batch_avg, "samples": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[Result saved] {out}")


if __name__ == "__main__":
    interactive_loop()

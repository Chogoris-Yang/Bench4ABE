"""
Experiment_2/plan_and_solve.py
Plan-and-Solve Experimental Plan Generator
Create detailed outline -> Write sections one by one -> Polish & Merge -> Score
Generator: DeepSeek V4 Pro  |  Judges: DeepSeek-chat + GPT-5.4
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
# Plan-and-Solve Prompts
# =========================================================

# Stage 1: Create Plan — outline the proposal
PLAN_PROMPT = """You are an expert research planner. Given a research question, your task is to create a DETAILED OUTLINE for a complete experimental plan.

Research Question:
{question}

Create a plan with 4-6 sections. Each section should have:
- A title
- A brief description of what it should cover
- Key points that MUST be addressed

The sections should cover:
1. Problem Definition & Research Objective
2. Related Work & Background
3. Proposed Methodology / Model Architecture
4. Experimental Design & Setup
5. Evaluation Metrics & Baselines
6. Expected Outcomes & Limitations (optional)

Output format (JSON array):
[
  {{"section_id": 1, "title": "...", "description": "what to cover", "key_points": ["point1", "point2", ...]}},
  ...
]

Explain your planning rationale briefly, then output the JSON array."""


# Stage 2-N: Write each section one by one
SOLVE_FIRST_SECTION_PROMPT = """You are an expert AI research scientist writing a research proposal.

Research Question:
{question}

Overall Plan:
{plan_summary}

You are writing SECTION {section_id}: "{section_title}"

Description: {section_description}

Key points to address:
{key_points}

This is the FIRST section — set the foundation. Write a comprehensive, well-structured section.
Use academic language. Be specific and rigorous."""


SOLVE_NEXT_SECTION_PROMPT = """You are an expert AI research scientist continuing a research proposal.

Research Question:
{question}

Overall Plan:
{plan_summary}

PREVIOUS SECTIONS (for context):
---
{previous_sections}
---

Now write SECTION {section_id}: "{section_title}"

Description: {section_description}

Key points to address:
{key_points}

Build on the previous sections. Maintain consistency — reference earlier definitions and decisions.
Do NOT repeat content from previous sections. Focus on what's NEW in this section."""


# Stage 7: Polish & Merge
POLISH_PROMPT = """You are an expert scientific editor.

Research Question:
{question}

Below are individually written sections of an experimental plan. Your task is to POLISH and MERGE them into a single, cohesive document.

{all_sections}

Instructions:
1. Fix any inconsistencies across sections (e.g., terminology, variable names)
2. Add smooth transitions between sections
3. Ensure the document flows logically from Problem -> Method -> Experiment -> Evaluation
4. Add a brief abstract/executive summary at the beginning
5. Ensure all references between sections are consistent
6. Fix any repetition or contradictions

Output the COMPLETE, polished experimental plan as a unified document.
Use proper academic formatting with clear section headings."""


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
    return "[ERROR] All API retries failed"


# =========================================================
# LangGraph State
# =========================================================

class PlanAndSolveState(TypedDict):
    research_topic: str
    plan: List[Dict]               # [{section_id, title, description, key_points}]
    sections: List[Dict]           # [{section_id, title, content}]
    current_section_index: int
    polished_plan: str             # final polished plan
    scores: List[Dict]
    final_report: str


# =========================================================
# Build Plan-and-Solve Pipeline
# =========================================================

def build_plan_and_solve_graph() -> StateGraph:

    # =====================
    # Node 1: Create Plan
    # =====================
    def make_plan(state: PlanAndSolveState) -> Dict:
        print("\n" + "=" * 60)
        print("[Stage 1] Plan — Create proposal outline")
        print("=" * 60)

        question = state["research_topic"]
        raw = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are an expert research planner. Output the JSON array of sections.",
            PLAN_PROMPT.format(question=question),
            temperature=0.2
        )

        plan = []
        try:
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            print("  [!] JSON parse failed, using default outline")
            plan = [
                {"section_id": 1, "title": "Problem Definition & Research Objective",
                 "description": "Clearly define the research problem, scope, and objectives",
                 "key_points": ["Problem statement", "Research gap", "Objectives", "Scope"]},
                {"section_id": 2, "title": "Related Work",
                 "description": "Review existing approaches and position this work",
                 "key_points": ["Key prior methods", "Limitations of existing work", "How this work differs"]},
                {"section_id": 3, "title": "Proposed Methodology",
                 "description": "Detailed description of the proposed approach",
                 "key_points": ["Model architecture", "Training methodology", "Key innovations", "Theoretical grounding"]},
                {"section_id": 4, "title": "Experimental Design",
                 "description": "Complete experimental setup and protocol",
                 "key_points": ["Datasets", "Baselines", "Ablation studies", "Hardware & implementation"]},
                {"section_id": 5, "title": "Evaluation & Results",
                 "description": "Evaluation metrics, expected results, and analysis plan",
                 "key_points": ["Metrics", "Expected outcomes", "Statistical analysis", "Success criteria"]},
            ]

        print(f"  -> Outline contains {len(plan)} sections:")
        for s in plan:
            kp_count = len(s.get("key_points", []))
            print(f"     S{s['section_id']}: {s['title'][:60]}... ({kp_count} key points)")

        return {"plan": plan, "current_section_index": 0}

    # =====================
    # Node 2: Solve current section
    # =====================
    def solve_section(state: PlanAndSolveState) -> Dict:
        plan = state.get("plan", [])
        current_idx = state.get("current_section_index", 0)
        sections = list(state.get("sections", []))

        if current_idx >= len(plan):
            return {}

        sec = plan[current_idx]
        print("\n" + "=" * 60)
        print(f"[Stage 2] Solve — Writing section {sec['section_id']}/{len(plan)}: {sec['title'][:60]}")
        print("=" * 60)

        question = state["research_topic"]

        # Plan summary (for each section's reference)
        plan_summary = "\n".join(
            f"  S{s['section_id']}: {s['title']} — {s.get('description', '')[:80]}"
            for s in plan
        )
        key_points_str = "\n".join(f"  • {kp}" for kp in sec.get("key_points", []))

        if current_idx == 0:
            result = call_llm(
                deepseek_client, GENERATOR_MODEL,
                "You are an expert research scientist writing the first section of a proposal.",
                SOLVE_FIRST_SECTION_PROMPT.format(
                    question=question, plan_summary=plan_summary,
                    section_id=sec["section_id"], section_title=sec["title"],
                    section_description=sec.get("description", ""),
                    key_points=key_points_str,
                ),
                temperature=0.3
            )
        else:
            prev_text = "\n\n---\n\n".join(
                f"## Section {s['section_id']}: {s['title']}\n{s['content']}"
                for s in sections
            )
            result = call_llm(
                deepseek_client, GENERATOR_MODEL,
                "You are an expert continuing a research proposal, building on previous sections.",
                SOLVE_NEXT_SECTION_PROMPT.format(
                    question=question, plan_summary=plan_summary,
                    previous_sections=prev_text,
                    section_id=sec["section_id"], section_title=sec["title"],
                    section_description=sec.get("description", ""),
                    key_points=key_points_str,
                ),
                temperature=0.3
            )

        sections.append({
            "section_id": sec["section_id"],
            "title": sec["title"],
            "content": result,
        })

        next_idx = current_idx + 1
        print(f"  -> S{sec['section_id']}: {len(result)} chars  ({len(plan) - next_idx} sections remaining)")

        return {"sections": sections, "current_section_index": next_idx}

    # =====================
    # Node 3: Polish & Merge
    # =====================
    def polish(state: PlanAndSolveState) -> Dict:
        print("\n" + "=" * 60)
        print("[Stage 3] Polish — Polish & Merge all sections into unified document")
        print("=" * 60)

        question = state["research_topic"]
        sections = state.get("sections", [])

        all_sections_text = "\n\n---\n\n".join(
            f"## Section {s['section_id']}: {s['title']}\n\n{s['content']}"
            for s in sections
        )

        polished = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are an expert scientific editor polishing a research proposal into a unified document.",
            POLISH_PROMPT.format(question=question, all_sections=all_sections_text),
            temperature=0.2
        )

        print(f"  -> Polished document: {len(polished)} chars")
        return {"polished_plan": polished}

    # =====================
    # Node 4: Dual-Judge Scoring
    # =====================
    def score_node(state: PlanAndSolveState) -> Dict:
        print("\n" + "=" * 60)
        print("[Scoring] Dual-Judge evaluation")
        print("=" * 60)

        plan_text = state.get("polished_plan", "")

        print("  [Judge 1] DeepSeek-chat ...")
        ds_text = call_llm(deepseek_client, JUDGE_MODEL_DS, SCORING_PROMPT, plan_text, temperature=0.0)
        ds_scores = parse_score(ds_text)
        print(f"    DS -> R={ds_scores['reliability']}, I={ds_scores['innovation']}")

        print("  [Judge 2] GPT-5.4 ...")
        gpt_text = call_llm(gpt_client, JUDGE_MODEL_GPT, SCORING_PROMPT, plan_text, temperature=0.0)
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
    # Node 5: Final Report
    # =====================
    def report_node(state: PlanAndSolveState) -> Dict:
        print("\n" + "=" * 60)
        print("[Report] Generate Plan-and-Solve Final Report")
        print("=" * 60)

        topic = state["research_topic"]
        polished = state.get("polished_plan", "")
        scores = state.get("scores", [])
        plan = state.get("plan", [])
        sections = state.get("sections", [])

        display = polished[:5000] + "\n\n...(truncated)" if len(polished) > 5000 else polished

        lines = [
            "=" * 70,
            "          Plan-and-Solve Experimental Plan Report",
            "=" * 70, "",
            f"Research Topic: {topic[:300]}", "",
            "-" * 70,
            f"Generation Pipeline: Create Outline ({len(plan)} sections) -> Write by Section -> Polish & Merge -> Scoring",
            "-" * 70, "",
            "Outline Structure:",
        ]
        for s in plan:
            sec_data = next((x for x in sections if x["section_id"] == s["section_id"]), None)
            ans_len = len(sec_data["content"]) if sec_data else 0
            lines.append(f"  S{s['section_id']}: {s['title'][:60]}... ({ans_len} chars)")

        lines += ["", "-" * 70, "Final Plan", "-" * 70, display or "Failed to generate"]

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
    # Routing: More sections remaining?
    # =====================
    def route_after_solve(state: PlanAndSolveState) -> Literal["solve", "polish"]:
        current_idx = state.get("current_section_index", 0)
        plan = state.get("plan", [])
        if current_idx < len(plan):
            print(f"  -> Continue writing S{plan[current_idx]['section_id']}")
            return "solve"
        print(f"  -> All {len(plan)} sections complete, entering polish")
        return "polish"

    # =====================
    # Assemble Graph
    #   make_plan -> solve -> [more?]
    #                  ^        |
    #                  |-- YES -|   NO -> polish -> score -> report -> END
    # =====================
    workflow = StateGraph(PlanAndSolveState)

    workflow.add_node("make_plan", make_plan)
    workflow.add_node("solve", solve_section)
    workflow.add_node("polish", polish)
    workflow.add_node("score", score_node)
    workflow.add_node("report", report_node)

    workflow.set_entry_point("make_plan")
    workflow.add_edge("make_plan", "solve")
    workflow.add_conditional_edges("solve", route_after_solve, {"solve": "solve", "polish": "polish"})
    workflow.add_edge("polish", "score")
    workflow.add_edge("score", "report")
    workflow.add_edge("report", END)

    return workflow


# =========================================================
# Compile & Run
# =========================================================

def compile_agent():
    return build_plan_and_solve_graph().compile(checkpointer=MemorySaver())


def run_agent(question: str, verbose: bool = True) -> Dict[str, Any]:
    app = compile_agent()

    initial_state: PlanAndSolveState = {
        "research_topic": question, "plan": [], "sections": [],
        "current_section_index": 0, "polished_plan": "", "scores": [], "final_report": "",
    }
    config = {"configurable": {"thread_id": f"pns-{int(time.time())}"}}

    print(f"\n{'*'*70}")
    print(f"Plan-and-Solve: {question[:200]}")
    print(f"{'*'*70}")
    print("Workflow: Create Outline -> Write by Section -> Polish & Merge -> Scoring -> Report")

    final_state = None
    for event in app.stream(initial_state, config, stream_mode="values"):
        final_state = event
        if verbose:
            for key in ["plan", "sections", "polished_plan"]:
                val = event.get(key, "")
                if isinstance(val, list) and val:
                    print(f"  [{key}] {len(val)}  items")
                elif isinstance(val, str) and len(val) > 50:
                    print(f"  [{key}] {len(val)} chars")

    if final_state is None:
        return {"final_report": "[ERROR]", "scores": [], "plan": [], "sections": []}

    return {
        "final_report": final_state.get("final_report", ""),
        "plan": final_state.get("plan", []),
        "sections": final_state.get("sections", []),
        "polished_plan": final_state.get("polished_plan", ""),
        "scores": final_state.get("scores", []),
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
║    Plan-and-Solve Experimental Plan Generator                             ║
║    Workflow: Create Outline -> Write by Section -> Polish & Merge -> Scoring         ║
║    Generator: DeepSeek V4 Pro | Judges: DS-chat + GPT-5.4          ║
╚══════════════════════════════════════════════════════════════╝

Commands:
  /run <question>    -- Generate experiment plan for research question
  /batch [N]     — Batch testing (default 3 samples)
  /demo          — Use built-in demo
  /quit          — Exit
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
            print("Goodbye!"); break
        if ui.lower() == "/demo":
            ui = "/run Design an experiment to evaluate whether large language models can self-improve through recursive self-critique without external feedback"

        if ui.lower().startswith("/run "):
            question = ui[5:].strip()
            result = run_agent(question)
            print("\n" + result["final_report"])
            out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f"pns_result_{int(time.time())}.json")
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
    print(f"  Plan-and-Solve Batch Test: {n} samples")
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
    print("         Plan-and-Solve Batch Test -- Cross-Sample Average ScoresSummary")
    print("=" * 70)

    for r in sorted(batch_avg["rounds"].keys()):
        info = batch_avg["rounds"][r]
        print(f"\n  [Scoring]  (Covering {info['count']}/{n} samples)")
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
                       f"pns_batch_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"generator": GENERATOR_MODEL, "judges": [JUDGE_MODEL_DS, JUDGE_MODEL_GPT], "samples": n},
            "batch_averages": batch_avg, "samples": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[Result saved] {out}")


if __name__ == "__main__":
    interactive_loop()

"""
Experiment_2/sot.py
Skeleton-of-Thought (SoT) Experimental Plan Generator — Ning et al. 2024
First generate skeleton outline -> parallel expand each skeleton point -> assemble and synthesize -> score -> revise skeleton and regenerate
Generator: DeepSeek V4 Pro  |  Judges: DeepSeek-chat + GPT-5.4

SoT vs Plan-and-Solve:
  Both define structure first then expand, but SoT's expansion phase is parallel (ThreadPool),
  and SoT's second round re-expands with feedback rather than continuing to write the next section
"""

import json
import os
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
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

MAX_ROUNDS = 2  # Max skeleton revision + re-expand rounds

# =========================================================
# Dataset Path
# =========================================================
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gen_lcot", "data", "train.jsonl"
)


# =========================================================
# SoT Prompts
# =========================================================

# Stage 1: Skeleton — generate concise skeleton outline
SKELETON_PROMPT = """You are a skeleton-of-thought planner. Given a research question,
generate a concise SKELETON outline — just the key points to expand later.

Research Question:
{question}

Generate a skeleton with 4-6 points. Each point should be a SINGLE SENTENCE
capturing a key aspect of the experimental plan.

Output JSON:
{{"skeleton": ["point1", "point2", ...]}}"""

# Stage 2: Expand — parallel expand each skeleton point
EXPAND_POINT_PROMPT = """You are expanding a skeleton point into a full section of an experimental plan.

Research Question:
{question}

Full Skeleton (for context of how this point fits):
{full_skeleton}

Point to expand:
{point}

Write a detailed, rigorous section expanding this single point.
{feedback}"""

# Stage 3: Assemble — assemble sections into a complete plan
ASSEMBLE_PROMPT = """You are assembling independently-written sections into one cohesive plan.

Research Question:
{question}

SECTIONS (written independently, need integration):
{sections}

Polish, merge, add transitions, fix inconsistencies between sections.
Output the COMPLETE unified experimental plan."""

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

class SoTState(TypedDict):
    """Skeleton-of-Thought state"""
    research_topic: str          # User's original question
    skeleton: List[str]          # Skeleton point list
    sections: List[Dict]         # Expanded sections [{index, point, content}]
    plan: str                    # Assembled full plan
    scores: List[Dict]           # Score history
    round_num: int               # Current round number
    feedback: str                # Feedback passed to next round
    final_report: str            # Final output report


# =========================================================
# Build SoT Pipeline
# =========================================================

def build_sot_graph() -> StateGraph:
    """
    Build the Skeleton-of-Thought state graph.

    Graph structure:
        __start__ -> skeleton -> expand -> assemble -> score -> [NEEDS_IMPROV?]
                                                              ↓ YES -> expand (with feedback)
                                                              ↓ NO  -> report -> END
    """

    # =====================
    # Node 1: Skeleton — generate skeleton outline
    # =====================
    def make_skeleton(state: SoTState) -> Dict:
        print("\n" + "=" * 60)
        print("[Stage 1] Skeleton — Generate skeleton outline (4-6 points)")
        print("=" * 60)

        question = state["research_topic"]
        raw = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "Generate a concise skeleton outline for the experimental plan.",
            SKELETON_PROMPT.format(question=question),
            temperature=0.3
        )

        # Parse JSON
        skeleton = []
        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                skeleton = json.loads(json_match.group(0)).get("skeleton", [])
        except json.JSONDecodeError:
            skeleton = [line.strip() for line in raw.split("\n") if len(line.strip()) > 20][:5]

        print(f"  -> {len(skeleton)} skeleton point(s):")
        for i, point in enumerate(skeleton):
            print(f"      [{i + 1}] {point[:80]}...")

        return {"skeleton": skeleton}

    # =====================
    # Node 2: Expand — parallel expand all skeleton points
    # =====================
    def expand_parallel(state: SoTState) -> Dict:
        round_num = state.get("round_num", 0) + 1
        skeleton = state.get("skeleton", [])
        question = state["research_topic"]

        # If there is feedback from previous round, add it as context
        fb = state.get("feedback", "")
        fb_text = f"\n\nFEEDBACK FROM PREVIOUS ROUND (apply these improvements):\n{fb}" if fb else ""

        print("\n" + "=" * 60)
        print(f"[Stage 2] Expand Round {round_num} — parallel expand {len(skeleton)} skeleton point(s)")
        print("=" * 60)

        skeleton_text = "\n".join(f"  [{i + 1}] {p}" for i, p in enumerate(skeleton))

        # ---- Expand a single skeleton point ----
        def expand_one(index: int, point: str) -> Dict:
            print(f"  [Expand] Skeleton point [{index + 1}]: {point[:50]}...")
            section = call_llm(
                deepseek_client, GENERATOR_MODEL,
                "Expand a single skeleton point into a detailed section.",
                EXPAND_POINT_PROMPT.format(
                    question=question,
                    full_skeleton=skeleton_text,
                    point=point,
                    feedback=fb_text,
                ),
                temperature=0.3
            )
            print(f"    [{index + 1}] -> {len(section)} chars")
            return {"index": index, "point": point, "content": section}

        # ---- Parallel expand (ThreadPool) ----
        sections = []
        if len(skeleton) <= 1:
            for i, p in enumerate(skeleton):
                sections.append(expand_one(i, p))
        else:
            with ThreadPoolExecutor(max_workers=min(len(skeleton), 4)) as pool:
                futures = [pool.submit(expand_one, i, p) for i, p in enumerate(skeleton)]
                for future in as_completed(futures):
                    sections.append(future.result())

        # Sort by original order
        sections.sort(key=lambda x: x["index"])

        return {"sections": sections, "round_num": round_num}

    # =====================
    # Node 3: Assemble — synthesize sections
    # =====================
    def assemble(state: SoTState) -> Dict:
        print("\n" + "=" * 60)
        print("[Stage 3] Assemble — synthesize sections into unified plan")
        print("=" * 60)

        question = state["research_topic"]
        sections = state.get("sections", [])

        sections_text = "\n\n---\n\n".join(
            f"## Section {s['index'] + 1}: {s['point']}\n{s['content']}"
            for s in sections
        )

        plan = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "Assemble independently-written sections into one cohesive plan.",
            ASSEMBLE_PROMPT.format(question=question, sections=sections_text),
            temperature=0.2
        )

        print(f"  -> Synthesized plan: {len(plan)} chars")
        return {"plan": plan}

    # =====================
    # Node 4: Dual-Judge Scoring
    # =====================
    def score_node(state: SoTState) -> Dict:
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

        scores = list(state.get("scores", []))
        scores.append({
            "round": round_num,
            "ds": scores_data["ds"],
            "gpt": scores_data["gpt"],
            "combined": c,
            "verdict": verdict,
        })

        # Generate feedback when score is insufficient
        feedback = ""
        if verdict == "NEEDS_IMPROVEMENT" and round_num < MAX_ROUNDS:
            feedback = (
                f"Previous round R={c['reliability']} I={c['innovation']}. "
                f"Improve depth, rigor, and detail in each expanded section. "
                f"Make sections longer and more comprehensive."
            )

        return {"scores": scores, "feedback": feedback}

    # =====================
    # Node 5: Final Report
    # =====================
    def report(state: SoTState) -> Dict:
        print("\n" + "=" * 60)
        print("[Report] Generate SoT Final Report")
        print("=" * 60)

        plan = state.get("plan", "")
        scores = state.get("scores", [])
        skeleton = state.get("skeleton", [])

        display = plan[:5000] + "\n\n...(truncated)" if len(plan) > 5000 else plan

        lines = [
            "=" * 70,
            "          Skeleton-of-Thought (SoT) Experimental Plan Report",
            "=" * 70,
            "",
            f"Research Topic: {state['research_topic'][:300]}",
            "",
            "-" * 70,
            f"Skeleton: {len(skeleton)} point(s) | Parallel Expand -> Assemble -> Score",
            "-" * 70,
            "",
            "Score History:",
        ]
        for s in scores:
            lines.append(
                f"  R{s['round']}: DS(R={s['ds'].get('reliability', 0)} I={s['ds'].get('innovation', 0)}) "
                f"GPT(R={s['gpt'].get('reliability', 0)} I={s['gpt'].get('innovation', 0)}) "
                f"-> Combined R={s['combined'].get('reliability', 0)} I={s['combined'].get('innovation', 0)}"
            )

        # Improvement delta
        if len(scores) >= 2:
            delta_r = round(scores[-1]["combined"]["reliability"] - scores[0]["combined"]["reliability"], 1)
            delta_i = round(scores[-1]["combined"]["innovation"] - scores[0]["combined"]["innovation"], 1)
            lines.append(f"\n  Skeleton Revision Improvement: Delta R = +{delta_r}  Delta I = +{delta_i}")

        lines += [
            "",
            "-" * 70,
            "Final Plan",
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
    def route_after_score(state: SoTState) -> Literal["expand", "report"]:
        round_num = state.get("round_num", 1)
        scores = state.get("scores", [])

        if round_num >= MAX_ROUNDS:
            print(f"  -> Max rounds reached ({MAX_ROUNDS}), output report")
            return "report"
        if scores and scores[-1]["verdict"] == "GOOD":
            print(f"  -> Score acceptable, output report")
            return "report"
        print(f"  -> Score insufficient, re-expanding with feedback...")
        return "expand"

    # =====================
    # Assemble Graph
    # =====================
    workflow = StateGraph(SoTState)

    workflow.add_node("skeleton", make_skeleton)
    workflow.add_node("expand", expand_parallel)
    workflow.add_node("assemble", assemble)
    workflow.add_node("score", score_node)
    workflow.add_node("report", report)

    workflow.set_entry_point("skeleton")
    workflow.add_edge("skeleton", "expand")
    workflow.add_edge("expand", "assemble")
    workflow.add_edge("assemble", "score")
    workflow.add_conditional_edges(
        "score", route_after_score,
        {"expand": "expand", "report": "report"},
    )
    workflow.add_edge("report", END)

    return workflow


# =========================================================
# Compile & Run
# =========================================================

def compile_agent():
    """Compile SoT Agent graph"""
    return build_sot_graph().compile(checkpointer=MemorySaver())


def run_agent(question: str, verbose: bool = True) -> Dict[str, Any]:
    """Run SoT Agent.
    Returns: {final_report, plan, scores}
    """
    app = compile_agent()
    initial: SoTState = {
        "research_topic": question,
        "skeleton": [],
        "sections": [],
        "plan": "",
        "scores": [],
        "round_num": 0,
        "feedback": "",
        "final_report": "",
    }
    config = {"configurable": {"thread_id": f"sot-{int(time.time())}"}}

    print(f"\n{'★' * 70}")
    print(f"SoT: {question[:200]}")
    print(f"{'★' * 70}")
    print("Pipeline: Skeleton -> ParallelExpand -> Assemble -> Score -> [Re-Expand]")

    final_state = None
    for event in app.stream(initial, config, stream_mode="values"):
        final_state = event

    if final_state is None:
        return {"final_report": "[ERROR]", "plan": "", "scores": []}

    return {
        "final_report": final_state.get("final_report", ""),
        "plan": final_state.get("plan", ""),
        "scores": final_state.get("scores", []),
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
║    Skeleton-of-Thought (SoT) Experimental Plan Generator    ║
║    Skeleton -> Parallel Expand -> Assemble -> Score -> Revise Skeleton  ║
║    Generator: DeepSeek V4 Pro | Judges: DS-chat + GPT-5.4   ║
╚══════════════════════════════════════════════════════════════╝

Commands:
  /run <question> -- Generate experimental plan for research question (SoT)
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
                               f"sot_result_{int(time.time())}.json")
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
    print(f"  SoT Batch Testing: {n} samples")
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
    print("         SoT Batch Testing — Cross-Sample Average Summary")
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
                       f"sot_batch_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"generator": GENERATOR_MODEL, "judges": [JUDGE_MODEL_DS, JUDGE_MODEL_GPT], "samples": n},
            "batch_averages": batch_avg,
            "samples": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[Result saved] {out}")


if __name__ == "__main__":
    interactive_loop()

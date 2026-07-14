"""
Experiment_2/tot.py
Tree-of-Thoughts (ToT) Experimental Plan Generator — Yao et al. 2023
BFS multi-path exploration -> review and prune -> keep top branches to expand further -> select global best
Generator: DeepSeek V4 Pro  |  Judges: DeepSeek-chat + GPT-5.4

ToT vs LATS:
  ToT is a simplified tree search: fixed BFS, fixed branch count, pure score-based selection (no UCT)
  LATS is an MCTS variant: has UCT exploration term, visit counts, backpropagation
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

# ---- Search hyperparameters ----
BRANCHES = 3    # Branches per level
MAX_DEPTH = 2   # Maximum search depth
KEEP_TOP = 2    # Keep top branches per level

# =========================================================
# Dataset Path
# =========================================================
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gen_lcot", "data", "train.jsonl"
)


# =========================================================
# ToT Prompts
# =========================================================

# Stage 1: Brainstorm — breadth-first generation of multiple different directions
TOT_BRAINSTORM = """You are exploring the solution space for an experimental plan.
Generate {branches} DISTINCT approaches.

Research Question:
{question}

Generate {branches} DIFFERENT high-level approaches (just the core idea + 1-2 sentence outline).
Make them genuinely different — try different paradigms, methodologies, angles.

Output JSON:
[
  {{"id": "A", "approach": "brief approach description", "core_idea": "the central novel idea"}},
  ...
]"""

# Stage 2: Expand — expand each branch into a full plan
TOT_EXPAND = """You are expanding a promising approach into a detailed experimental plan.

Research Question:
{question}

Approach to expand:
{approach}

Expand this into a complete, rigorous experimental plan including:
- Research Objective
- Methodology (detailed — model architecture, training, innovations)
- Experimental Design (datasets, baselines, ablation studies)
- Evaluation & Metrics
- Strengths & Weaknesses of this approach"""

# Stage 3: Evaluate — dual-judge scoring
TOT_EVALUATE = """You are a ToT evaluator. Rate the following plan on two dimensions.
Output ONLY:
Reliability: xx
Innovation: xx

Plan:
{plan}"""


# =========================================================
# Data & Utility Functions
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
                       "Rate this plan.", TOT_EVALUATE.format(plan=plan[:4000]),
                       temperature=0.0)
    ds = parse_score(ds_text)

    gpt_text = call_llm(gpt_client, JUDGE_MODEL_GPT,
                        "Rate this plan.", TOT_EVALUATE.format(plan=plan[:4000]),
                        temperature=0.0)
    gpt = parse_score(gpt_text)

    combined = {
        "reliability": round((ds["reliability"] + gpt["reliability"]) / 2, 1),
        "innovation": round((ds["innovation"] + gpt["innovation"]) / 2, 1),
    }
    return {"ds": ds, "gpt": gpt, "combined": combined}


# =========================================================
# LangGraph State Definition
# =========================================================

class ToTState(TypedDict):
    """ToT search state"""
    research_topic: str          # User's original question
    depth: int                   # Current search depth
    candidates: List[Dict]       # Current candidate nodes [{id, approach, plan, scores}]
    best_plan: str               # Global best plan text
    best_scores: Dict            # Global best scores
    scores: List[Dict]           # Score history of selected nodes per round
    final_report: str            # Final output report


# =========================================================
# Build ToT BFS Search Graph
# =========================================================

def build_tot_graph() -> StateGraph:
    """
    Build the ToT state graph.

    Graph structure:
        __start__ -> brainstorm -> expand_and_score -> [depth < MAX_DEPTH?]
                                                        ↓ YES -> expand_and_score (loop)
                                                        ↓ NO  -> report -> END
    """

    # =====================
    # Node 1: Brainstorm — breadth-first generation of initial branches
    # =====================
    def brainstorm(state: ToTState) -> Dict:
        print("\n" + "=" * 60)
        print(f"[ToT Depth 0] Brainstorm — breadth-first generate {BRANCHES} distinct approaches")
        print("=" * 60)

        question = state["research_topic"]
        raw = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "Brainstorm diverse approaches for the research question.",
            TOT_BRAINSTORM.format(question=question, branches=BRANCHES),
            temperature=0.6   # Higher temperature to encourage diversity
        )

        # Parse JSON
        candidates = []
        try:
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            if json_match:
                candidates = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

        # fallback: split by text chunks
        if not candidates:
            candidates = [
                {"id": f"T{i}", "approach": f"Approach {i}",
                 "core_idea": raw[i * 200:(i + 1) * 200]}
                for i in range(BRANCHES) if i * 200 < len(raw)
            ]

        candidates = candidates[:BRANCHES]
        print(f"  -> Generated {len(candidates)} root candidate(s):")
        for c in candidates:
            print(f"     {c.get('id', '?')}: {c.get('core_idea', c.get('approach', ''))[:80]}...")

        return {"candidates": candidates, "depth": 0}

    # =====================
    # Node 2: Expand + Score — expand all candidates + parallel scoring + pruning
    # =====================
    def expand_and_score(state: ToTState) -> Dict:
        depth = state.get("depth", 0) + 1
        candidates = state.get("candidates", [])
        question = state["research_topic"]

        print("\n" + "=" * 60)
        print(f"[ToT Depth {depth}] Expand {len(candidates)} branch(es) -> Score -> Keep top {KEEP_TOP}")
        print("=" * 60)

        # ---- Expand a single candidate into a full plan ----
        def expand_one(candidate: Dict) -> Dict:
            plan_id = candidate.get("id", "?")
            approach = candidate.get("approach", candidate.get("core_idea", ""))
            print(f"  [Expand] {plan_id}: {approach[:60]}...")

            plan = call_llm(
                deepseek_client, GENERATOR_MODEL,
                "Expand this approach into a full experimental plan.",
                TOT_EXPAND.format(question=question, approach=approach),
                temperature=0.3
            )

            scores = score_single_plan(plan)
            c = scores["combined"]
            print(f"    {plan_id} -> R={c['reliability']} I={c['innovation']}")

            return {**candidate, "plan": plan, "scores": scores}

        # ---- Parallel expand (ThreadPool) ----
        expanded = []
        if len(candidates) <= 1:
            for c in candidates:
                expanded.append(expand_one(c))
        else:
            with ThreadPoolExecutor(max_workers=min(len(candidates), 4)) as pool:
                futures = [pool.submit(expand_one, c) for c in candidates]
                for future in as_completed(futures):
                    expanded.append(future.result())

        # ---- Prune: sort by combined score, keep top K ----
        expanded.sort(
            key=lambda x: (
                x["scores"]["combined"]["reliability"] * 0.6 +
                x["scores"]["combined"]["innovation"] * 0.4
            ),
            reverse=True
        )
        keep = expanded[:KEEP_TOP]

        best = keep[0]
        best_scores = best["scores"]["combined"]
        print(f"  -> Best at this depth: {best['id']} R={best_scores['reliability']} I={best_scores['innovation']}")

        # ---- Update global best ----
        prev_best = state.get("best_scores", {})
        prev_combined = (
            prev_best.get("reliability", 0) * 0.6 +
            prev_best.get("innovation", 0) * 0.4
        ) if prev_best else -1
        cur_combined = best_scores["reliability"] * 0.6 + best_scores["innovation"] * 0.4

        new_best_plan = state.get("best_plan", "")
        new_best_scores = dict(state.get("best_scores", {})) if state.get("best_scores") else {}

        if cur_combined > prev_combined:
            new_best_plan = best["plan"]
            new_best_scores = best_scores
            print(f"  -> New global best!")

        # Record scores of selected nodes
        scores = list(state.get("scores", []))
        c = best["scores"]["combined"]
        scores.append({
            "round": depth,
            "ds": best["scores"]["ds"],
            "gpt": best["scores"]["gpt"],
            "combined": c,
            "verdict": "GOOD" if c["reliability"] >= 70 else "NEEDS_IMPROVEMENT",
        })

        return {
            "depth": depth,
            "candidates": keep,
            "scores": scores,
            "best_plan": new_best_plan,
            "best_scores": new_best_scores,
        }

    # =====================
    # Node 3: Final Report
    # =====================
    def report(state: ToTState) -> Dict:
        print("\n" + "=" * 60)
        print("[Report] Generate ToT Final Report")
        print("=" * 60)

        best_plan = state.get("best_plan", "")
        best_scores = state.get("best_scores", {})
        scores = state.get("scores", [])

        display = best_plan[:5000] + "\n\n...(truncated)" if len(best_plan) > 5000 else best_plan

        lines = [
            "=" * 70,
            "          Tree-of-Thoughts (ToT) Experimental Plan Report",
            "=" * 70,
            "",
            f"Research Topic: {state['research_topic'][:300]}",
            "",
            "-" * 70,
            f"Search Config: {BRANCHES} branches x {MAX_DEPTH} depth | Keep top {KEEP_TOP} per level",
            "-" * 70,
            "",
            "Selected Plan Score History (per depth):",
        ]
        for s in scores:
            lines.append(
                f"  Depth {s['round']}: "
                f"DS(R={s['ds'].get('reliability', 0)} I={s['ds'].get('innovation', 0)}) "
                f"GPT(R={s['gpt'].get('reliability', 0)} I={s['gpt'].get('innovation', 0)}) "
                f"-> Combined R={s['combined'].get('reliability', 0)} I={s['combined'].get('innovation', 0)}"
            )

        lines += [
            "",
            "-" * 70,
            f"Global Best Plan (R={best_scores.get('reliability', 'N/A')} I={best_scores.get('innovation', 'N/A')})",
            "-" * 70,
            "",
            display or "Failed to generate global best plan",
            "",
            "=" * 70,
        ]
        return {"final_report": "\n".join(lines)}

    # =====================
    # Routing: continue exploring or finish?
    # =====================
    def route_after_expand(state: ToTState) -> Literal["expand", "report"]:
        depth = state.get("depth", 0)
        if depth >= MAX_DEPTH:
            print(f"  -> Max depth reached ({MAX_DEPTH}), output report")
            return "report"
        print(f"  -> Continue to next depth (depth {depth}/{MAX_DEPTH})")
        return "expand"

    # =====================
    # Assemble Graph
    # =====================
    workflow = StateGraph(ToTState)

    workflow.add_node("brainstorm", brainstorm)
    workflow.add_node("expand", expand_and_score)
    workflow.add_node("report", report)

    workflow.set_entry_point("brainstorm")
    workflow.add_edge("brainstorm", "expand")
    workflow.add_conditional_edges(
        "expand", route_after_expand,
        {"expand": "expand", "report": "report"},
    )
    workflow.add_edge("report", END)

    return workflow


# =========================================================
# Compile & Run
# =========================================================

def compile_agent():
    """Compile ToT Agent graph"""
    return build_tot_graph().compile(checkpointer=MemorySaver())


def run_agent(question: str, verbose: bool = True) -> Dict[str, Any]:
    """Run ToT Agent to generate an experimental plan for the given research question.
    Returns: {final_report, best_plan, scores}
    """
    app = compile_agent()
    initial: ToTState = {
        "research_topic": question,
        "depth": 0,
        "candidates": [],
        "best_plan": "",
        "best_scores": {},
        "scores": [],
        "final_report": "",
    }
    config = {"configurable": {"thread_id": f"tot-{int(time.time())}"}}

    print(f"\n{'★' * 70}")
    print(f"ToT Search: {question[:200]}")
    print(f"{'★' * 70}")
    print(f"Config: B={BRANCHES} branches x D={MAX_DEPTH} depth | Keep={KEEP_TOP} per level")

    final_state = None
    for event in app.stream(initial, config, stream_mode="values"):
        final_state = event
        if verbose and event.get("candidates"):
            print(f"  [candidates] {len(event['candidates'])} node(s)")

    if final_state is None:
        return {"final_report": "[ERROR] Agent produced no output", "best_plan": "", "scores": []}

    return {
        "final_report": final_state.get("final_report", ""),
        "best_plan": final_state.get("best_plan", ""),
        "scores": final_state.get("scores", []),
    }


# =========================================================
# Cross-Sample Average Summary
# =========================================================

def _avg(lst: List[float]) -> float:
    """Safely compute average"""
    return round(sum(lst) / len(lst), 1) if lst else 0.0


def compute_batch_averages(all_results: List[Dict]) -> Dict:
    """Compute cross-sample averages for batch testing (grouped by round)"""
    by_round: Dict[int, Dict[str, List[float]]] = {}
    for result in all_results:
        for s in result.get("scores", []):
            r = s.get("round", 1)
            if r not in by_round:
                by_round[r] = {"rel": [], "inn": []}
            c = s.get("combined", {})
            by_round[r]["rel"].append(c.get("reliability", 0))
            by_round[r]["inn"].append(c.get("innovation", 0))

    rounds_summary = {}
    for ri in sorted(by_round.keys()):
        d = by_round[ri]
        rounds_summary[ri] = {
            "count": len(d["rel"]),
            "reliability_avg": _avg(d["rel"]),
            "innovation_avg": _avg(d["inn"]),
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
║    Tree-of-Thoughts (ToT) Experimental Plan Generator       ║
║    BFS Search: {BRANCHES} branches x {MAX_DEPTH} depth | Keep top {KEEP_TOP} per level        ║
║    Generator: DeepSeek V4 Pro | Judges: DS-chat + GPT-5.4   ║
╚══════════════════════════════════════════════════════════════╝

Commands:
  /run <question> -- Generate experimental plan for research question (ToT)
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
                               f"tot_result_{int(time.time())}.json")
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
    """Batch testing: use first N dataset samples, output cross-sample average scores per round"""
    dataset = load_dataset(sample_size=n)
    print(f"\n{'#' * 70}")
    print(f"  ToT Batch Testing: {n} samples | B={BRANCHES} D={MAX_DEPTH} Keep={KEEP_TOP}")
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
            print(f"  -> Depth {s['round']}: Combined R={s['combined'].get('reliability', 0)} I={s['combined'].get('innovation', 0)}")

    # Cross-sample averages
    batch_avg = compute_batch_averages(all_results)
    print("\n" + "=" * 70)
    print("         ToT Batch Testing — Cross-Sample Average Summary")
    print("=" * 70)

    for ri in sorted(batch_avg["rounds"].keys()):
        info = batch_avg["rounds"][ri]
        print(f"\n  [Depth {ri}]  (Covering {info['count']} selection(s))")
        print(f"    Selected Plan Average:  R = {info['reliability_avg']}  |  I = {info['innovation_avg']}")

    overall = batch_avg["overall"]
    print(f"\n  {'─' * 60}")
    print(f"  [Global Best Average]  (total {overall['total']} score(s))")
    print(f"    * Reliability = {overall['reliability_avg']}  |  Innovation = {overall['innovation_avg']}")
    print(f"\n{'=' * 70}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"tot_batch_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"breadth": BRANCHES, "max_depth": MAX_DEPTH, "keep_top": KEEP_TOP,
                       "generator": GENERATOR_MODEL, "judges": [JUDGE_MODEL_DS, JUDGE_MODEL_GPT],
                       "samples": n},
            "batch_averages": batch_avg,
            "samples": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[Result saved] {out}")


# =========================================================
# Main Entry
# =========================================================

if __name__ == "__main__":
    interactive_loop()

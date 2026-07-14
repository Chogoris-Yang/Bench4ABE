"""
Experiment_2/lats.py
LATS (Language Agent Tree Search) Experimental Plan Generator
Expand(generate candidates) -> Simulate(score) -> Select(best) -> Backpropagate -> Loop
Generator: DeepSeek V4 Pro  |  Judge: DeepSeek-chat + GPT-5.4
"""

import json
import os
import re
import time
import random
import math
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

BREADTH = 2            # Number of candidate plans per level
MAX_DEPTH = 3          # Maximum search depth
EXPLORATION_WEIGHT = 1.4  # UCT exploration weight

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gen_lcot", "data", "train.jsonl"
)


# =========================================================
# LATS Prompts
# =========================================================

ROOT_GENERATION_PROMPT = """You are an expert AI research scientist. Generate {breadth} DISTINCT experimental plans
for the following research question. Each plan should take a DIFFERENT approach — explore diverse methodologies,
experimental designs, or evaluation strategies.

Research Question:
{question}

For each of the {breadth} plans, include:
- A unique identifier (Plan A, Plan B, ...)
- The CORE IDEA that differentiates it from the others
- Research Objective
- Proposed Methodology
- Experimental Design
- Expected Strengths & Weaknesses

Output format (JSON array):
[
  {{"plan_id": "A", "core_idea": "what makes this unique", "plan": "complete plan text..."}},
  ...
]

Make the plans GENUINELY different — not just minor variations. Explore the solution space broadly."""


EXPAND_PROMPT = """You are an expert AI research scientist. You have a best-so-far experimental plan.
Your task: Generate {breadth} IMPROVED VARIATIONS of this plan, each taking a different refinement direction.

Current Best Plan:
---
{best_plan}
---

Scores: Reliability={reliability}, Innovation={innovation}

Generate {breadth} DISTINCT improvements. Each variation should:
- Address weaknesses in the original
- Push in a DIFFERENT direction (e.g., one more rigorous, one more innovative, one more practical)
- Keep the core strengths while fixing specific issues

Output format (JSON array):
[
  {{"plan_id": "V1", "improvement_direction": "e.g., stronger theoretical grounding", "plan": "improved plan text..."}},
  ...
]

Be creative — explore genuinely different ways to improve the plan."""


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
# Tree Node Structure
# =========================================================

class TreeNode:
    """LATS search tree node"""
    def __init__(self, plan_id: str, plan_text: str, parent=None, depth: int = 0):
        self.plan_id = plan_id
        self.plan_text = plan_text
        self.parent = parent
        self.depth = depth
        self.children: List[TreeNode] = []
        self.visits = 0
        self.scores: Dict = {}       # {reliability, innovation}
        self.total_score = 0.0       # Combined score (for UCT)
        self.mean_score = 0.0

    def uct_value(self, parent_visits: int, c: float = EXPLORATION_WEIGHT) -> float:
        """Compute UCT = Avg + Exploration term"""
        if self.visits == 0:
            return float('inf')  # Unvisited nodes prioritized
        exploitation = self.mean_score / 100.0
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def update(self, reliability: float, innovation: float):
        """Update node with new scores"""
        self.visits += 1
        combined = (reliability + innovation) / 2.0
        self.total_score += combined
        self.mean_score = self.total_score / self.visits
        self.scores = {"reliability": reliability, "innovation": innovation}


# =========================================================
# LangGraph State
# =========================================================

class LATSState(TypedDict):
    research_topic: str
    depth: int                         # Current search depth
    root_nodes: List[Dict]             # Level 0 candidates [{plan_id, plan_text}]
    tree_history: List[Dict]           # Search tree summary
    current_nodes: List[Dict]          # Current level candidates [{plan_id, plan_text, scores, depth}]
    best_plan: str                     # Global Best Plan
    best_scores: Dict                  # Global best scores
    final_report: str


# =========================================================
# Build LATS Pipeline
# =========================================================

def score_single_plan(plan_text: str) -> Dict:
    """Perform dual-judge scoring on a single plan"""
    ds_text = call_llm(deepseek_client, JUDGE_MODEL_DS, SCORING_PROMPT, plan_text, temperature=0.0)
    ds = parse_score(ds_text)
    gpt_text = call_llm(gpt_client, JUDGE_MODEL_GPT, SCORING_PROMPT, plan_text, temperature=0.0)
    gpt = parse_score(gpt_text)
    combined_rel = round((ds["reliability"] + gpt["reliability"]) / 2, 1)
    combined_inn = round((ds["innovation"] + gpt["innovation"]) / 2, 1)
    return {"ds": ds, "gpt": gpt, "combined": {"reliability": combined_rel, "innovation": combined_inn}}


def build_lats_graph() -> StateGraph:

    # =====================
    # Node 1: Root — breadth-first generation of initial candidates
    # =====================
    def root_expand(state: LATSState) -> Dict:
        print("\n" + "=" * 60)
        print(f"[Root] BFS generate {BREADTH} different initial plans")
        print("=" * 60)

        question = state["research_topic"]
        raw = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are a tree search explorer. Generate diverse initial plans.",
            ROOT_GENERATION_PROMPT.format(question=question, breadth=BREADTH),
            temperature=0.5  # High temperature to encourage diversity
        )

        root_nodes = []
        try:
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            if json_match:
                root_nodes = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            print("  [!] JSON parse failed, using fallback")
            root_nodes = [
                {"plan_id": "A", "core_idea": "Conventional approach",
                 "plan": raw[:2000] if raw else "Generate a standard experimental plan with established methods."},
            ]
            # Split long text into multiple candidates
            if len(root_nodes) == 1 and BREADTH > 1:
                parts = raw.split("\n\n")
                root_nodes = []
                for j, part in enumerate(parts[:BREADTH]):
                    if len(part.strip()) > 100:
                        root_nodes.append({"plan_id": chr(65+j), "core_idea": f"Variant {j+1}", "plan": part.strip()})

        print(f"  -> generated {len(root_nodes)} root candidates:")
        for rn in root_nodes[:BREADTH]:
            print(f"     Plan {rn.get('plan_id','?')}: {rn.get('core_idea', rn.get('plan',''))[:80]}...")

        return {"root_nodes": root_nodes[:BREADTH], "depth": 0, "current_nodes": root_nodes[:BREADTH]}

    # =====================
    # Node 2: Simulate — Parallel scoring of all current candidates
    # =====================
    def simulate(state: LATSState) -> Dict:
        current_nodes = state.get("current_nodes", [])
        depth = state.get("depth", 0)

        print("\n" + "=" * 60)
        print(f"[Simulate] Depth {depth} — Parallel scoring {len(current_nodes)} candidates")
        print("=" * 60)

        def score_node(node: Dict) -> Dict:
            pid = node.get("plan_id", "?")
            plan = node.get("plan", "")
            print(f"  [Scoring] Plan {pid} ({len(plan)} chars)...")
            scores = score_single_plan(plan)
            c = scores["combined"]
            print(f"    Plan {pid}: DS(R={scores['ds']['reliability']} I={scores['ds']['innovation']}) "
                  f"GPT(R={scores['gpt']['reliability']} I={scores['gpt']['innovation']}) "
                  f"-> Combined R={c['reliability']} I={c['innovation']}")
            return {**node, "scores": scores, "depth": depth}

        scored_nodes = []
        if len(current_nodes) <= 1:
            for node in current_nodes:
                scored_nodes.append(score_node(node))
        else:
            with ThreadPoolExecutor(max_workers=min(len(current_nodes), 4)) as pool:
                futures = [pool.submit(score_node, n) for n in current_nodes]
                for f in as_completed(futures):
                    scored_nodes.append(f.result())

        return {"current_nodes": scored_nodes}

    # =====================
    # Node 3: Select — UCT select best node
    # =====================
    def select(state: LATSState) -> Dict:
        current_nodes = state.get("current_nodes", [])
        depth = state.get("depth", 0)

        print("\n" + "=" * 60)
        print(f"[Select] Depth {depth} — from {len(current_nodes)} candidates select best")
        print("=" * 60)

        if not current_nodes:
            return {}

        # Compute each nodeCombined score and select
        best = None
        best_combined = -1

        for node in current_nodes:
            scores = node.get("scores", {})
            c = scores.get("combined", {})
            rel = c.get("reliability", 0)
            inn = c.get("innovation", 0)
            # Combined score: 60% reliability + 40% innovation (reliability-weighted)
            combined_score = rel * 0.6 + inn * 0.4

            print(f"    Plan {node.get('plan_id','?')}: R={rel} I={inn} -> Combined={combined_score:.1f}")

            if combined_score > best_combined:
                best_combined = combined_score
                best = node

        if best is None:
            return {}

        c = best["scores"]["combined"]
        print(f"  -> Selected Plan {best.get('plan_id','?')}: R={c['reliability']} I={c['innovation']}")

        # Update global best
        current_best_scores = state.get("best_scores", {})
        current_best_rel = current_best_scores.get("reliability", 0) if current_best_scores else 0
        current_best_inn = current_best_scores.get("innovation", 0) if current_best_scores else 0
        current_best_combined = current_best_rel * 0.6 + current_best_inn * 0.4

        new_best_plan = state.get("best_plan", "")
        new_best_scores = dict(current_best_scores) if current_best_scores else {}

        if best_combined > current_best_combined:
            new_best_plan = best.get("plan", "")
            new_best_scores = c
            print(f"  -> New global best!")

        # Record tree history
        tree_history = list(state.get("tree_history", []))
        tree_history.append({
            "depth": depth,
            "candidates": len(current_nodes),
            "selected": best.get("plan_id", "?"),
            "selected_scores": best.get("scores", {}).get("combined", {}),
            "best_so_far": new_best_scores,
        })

        return {
            "best_plan": new_best_plan,
            "best_scores": new_best_scores,
            "tree_history": tree_history,
        }

    # =====================
    # Node 4: Expand — Generate new candidates from best node
    # =====================
    def expand(state: LATSState) -> Dict:
        depth = state.get("depth", 0)
        best_plan = state.get("best_plan", "")
        best_scores = state.get("best_scores", {})
        question = state["research_topic"]

        new_depth = depth + 1
        print("\n" + "=" * 60)
        print(f"[Expand] Depth {depth} -> {new_depth} — Generate {BREADTH} improved plans from best node")
        print("=" * 60)

        if not best_plan:
            print("  [!] No best plan to expand")
            return {"depth": new_depth, "current_nodes": []}

        raw = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are exploring the solution tree. Generate diverse improvements.",
            EXPAND_PROMPT.format(
                breadth=BREADTH,
                best_plan=best_plan[:4000],
                reliability=best_scores.get("reliability", 0),
                innovation=best_scores.get("innovation", 0),
            ),
            temperature=0.5
        )

        children = []
        try:
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            if json_match:
                children = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            children = [{"plan_id": f"D{new_depth}V1", "improvement_direction": "improved variant",
                         "plan": raw[:2000] if raw else best_plan}]

        print(f"  -> Expanded {len(children)} improved candidates:")
        for ch in children[:BREADTH]:
            print(f"     {ch.get('plan_id','?')}: {ch.get('improvement_direction', ch.get('plan',''))[:80]}...")

        return {"depth": new_depth, "current_nodes": children[:BREADTH]}

    # =====================
    # Node 5: Report
    # =====================
    def report_node(state: LATSState) -> Dict:
        print("\n" + "=" * 60)
        print("[Report] Generate LATS Final Report")
        print("=" * 60)

        topic = state["research_topic"]
        best_plan = state.get("best_plan", "")
        best_scores = state.get("best_scores", {})
        tree_history = state.get("tree_history", [])
        root_nodes = state.get("root_nodes", [])

        display = best_plan[:5000] + "\n\n...(truncated)" if len(best_plan) > 5000 else best_plan

        lines = [
            "=" * 70,
            "          LATS (Language Agent Tree Search) Experiment Protocol Report",
            "=" * 70, "",
            f"Research Topic: {topic[:300]}", "",
            "-" * 70,
            f"Search Config: Breadth={BREADTH}, MaxDepth={MAX_DEPTH}, UCT_c={EXPLORATION_WEIGHT}",
            f"Search Workflow: Root(generated {len(root_nodes)} candidates) -> Simulate(Scoring) -> Select(best) -> Expand(refine)",
            "-" * 70,
        ]

        if tree_history:
            lines += ["", "Search Tree History:", ""]
            for th in tree_history:
                s = th.get("selected_scores", {})
                b = th.get("best_so_far", {})
                lines.append(
                    f"  depth {th['depth']}: {th['candidates']}candidates -> select {th['selected']}"
                    f"(R={s.get('reliability',0)} I={s.get('innovation',0)})"
                    f" | global best R={b.get('reliability',0)} I={b.get('innovation',0)}"
                )

        lines += ["", "-" * 70, "Global Best Plan", "-" * 70]
        if best_scores:
            lines.append(
                f"Combined R={best_scores.get('reliability',0)}/100 "
                f"I={best_scores.get('innovation',0)}/100"
            )
        lines += ["", display or "Failed to generate", "", "=" * 70]

        return {"final_report": "\n".join(lines)}

    # =====================
    # Routing: Continue search or end?
    # =====================
    def route_after_select(state: LATSState) -> Literal["expand", "report"]:
        depth = state.get("depth", 0)
        if depth >= MAX_DEPTH:
            print(f"  -> Max depth reached ({MAX_DEPTH}), output report")
            return "report"
        best_scores = state.get("best_scores", {})
        if best_scores.get("reliability", 0) >= 85 and best_scores.get("innovation", 0) >= 80:
            print(f"  -> Plan already excellent (R>85 & I>80), output report")
            return "report"
        print(f"  -> Continue search (Depth {depth}/{MAX_DEPTH})")
        return "expand"

    # =====================
    # Assemble Graph
    #   root -> simulate -> select -> [continue?]
    #                                    | YES -> expand -> simulate (loop)
    #                                    | NO  -> report -> END
    # =====================
    workflow = StateGraph(LATSState)

    workflow.add_node("root", root_expand)
    workflow.add_node("simulate", simulate)
    workflow.add_node("select", select)
    workflow.add_node("expand", expand)
    workflow.add_node("report", report_node)

    workflow.set_entry_point("root")
    workflow.add_edge("root", "simulate")
    workflow.add_edge("simulate", "select")
    workflow.add_conditional_edges("select", route_after_select,
                                   {"expand": "expand", "report": "report"})
    workflow.add_edge("expand", "simulate")
    workflow.add_edge("report", END)

    return workflow


# =========================================================
# Compile & Run
# =========================================================

def compile_agent():
    return build_lats_graph().compile(checkpointer=MemorySaver())


def run_agent(question: str, verbose: bool = True) -> Dict[str, Any]:
    app = compile_agent()
    initial: LATSState = {
        "research_topic": question, "depth": 0, "root_nodes": [],
        "tree_history": [], "current_nodes": [],
        "best_plan": "", "best_scores": {}, "final_report": "",
    }
    config = {"configurable": {"thread_id": f"lats-{int(time.time())}"}}

    print(f"\n{'*'*70}")
    print(f"LATS: {question[:200]}")
    print(f"{'*'*70}")
    print(f"B={BREADTH} D={MAX_DEPTH}  Root -> [Simulate -> Select -> Expand] x {MAX_DEPTH}")

    final_state = None
    for event in app.stream(initial, config, stream_mode="values"):
        final_state = event
        if verbose:
            for key in ["current_nodes", "tree_history", "best_plan"]:
                val = event.get(key, "")
                if isinstance(val, list) and val:
                    print(f"  [{key}] {len(val)}  items")
                elif isinstance(val, str) and len(val) > 50:
                    print(f"  [{key}] {len(val)} chars")

    if final_state is None:
        return {"final_report": "[ERROR]", "best_plan": "", "tree_history": []}

    return {
        "final_report": final_state.get("final_report", ""),
        "best_plan": final_state.get("best_plan", ""),
        "best_scores": final_state.get("best_scores", {}),
        "tree_history": final_state.get("tree_history", []),
    }


# =========================================================
# Cross-Sample Average Scores
# =========================================================

def _avg(lst: List[float]) -> float:
    return round(sum(lst) / len(lst), 1) if lst else 0.0


def compute_batch_averages(all_results: List[Dict]) -> Dict:
    rels, inns = [], []
    for r in all_results:
        scores = r.get("best_scores", {})
        if scores:
            rels.append(scores.get("reliability", 0))
            inns.append(scores.get("innovation", 0))

    # Statistics by depth
    by_depth: Dict[int, Dict[str, List[float]]] = {}
    for r in all_results:
        for h in r.get("tree_history", []):
            d = h.get("depth", 0)
            if d not in by_depth:
                by_depth[d] = {"rel": [], "inn": []}
            s = h.get("selected_scores", {})
            by_depth[d]["rel"].append(s.get("reliability", 0))
            by_depth[d]["inn"].append(s.get("innovation", 0))

    depths_summary = {}
    for d in sorted(by_depth.keys()):
        dd = by_depth[d]
        depths_summary[d] = {"count": len(dd["rel"]),
                             "reliability_avg": _avg(dd["rel"]),
                             "innovation_avg": _avg(dd["inn"])}

    return {
        "depths": depths_summary,
        "overall": {
            "total_samples": len(rels),
            "reliability_avg": _avg(rels),
            "innovation_avg": _avg(inns),
        },
    }


# =========================================================
# CLI
# =========================================================

def interactive_loop():
    print(rf"""
╔══════════════════════════════════════════════════════════════╗
║    LATS Experiment Protocol Generator                                      ║
║    Root({BREADTH}branches) -> [Simulate -> Select -> Expand] x {MAX_DEPTH}  ║
║    Generator: DeepSeek V4 Pro | Judges: DS-chat + GPT-5.4          ║
╚══════════════════════════════════════════════════════════════╝

Commands:
  /run <question>    -- Generate experiment plan for research question (LATS)
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
                               f"lats_result_{int(time.time())}.json")
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
    print(f"  LATS Batch Test：{n} samples (B={BREADTH}, D={MAX_DEPTH})")
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
            "report": result["final_report"],
            "best_scores": result.get("best_scores", {}),
            "tree_history": result.get("tree_history", []),
        })
        bs = result.get("best_scores", {})
        th = result.get("tree_history", [])
        print(f"  -> global best: R={bs.get('reliability','?')} I={bs.get('innovation','?')} "
              f"(search {len(th)} levels, total evaluations {sum(h.get('candidates',0) for h in th)} times)")

    batch_avg = compute_batch_averages(all_results)
    print("\n" + "=" * 70)
    print("         LATS Batch Test -- Cross-Sample Average ScoresSummary")
    print("=" * 70)

    for d in sorted(batch_avg["depths"].keys()):
        info = batch_avg["depths"][d]
        print(f"\n  [depth {d}]  (Covering {info['count']} selections)")
        print(f"    Avg of Selected Plans:  R = {info['reliability_avg']}  |  I = {info['innovation_avg']}")

    overall = batch_avg["overall"]
    print(f"\n  {'-'*60}")
    print(f"  [Global Best Avg]  (Total {overall['total_samples']} samples)")
    print(f"    * Reliability = {overall['reliability_avg']}  |  Innovation = {overall['innovation_avg']}")
    print(f"\n{'='*70}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"lats_batch_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"generator": GENERATOR_MODEL, "judges": [JUDGE_MODEL_DS, JUDGE_MODEL_GPT],
                       "samples": n, "breadth": BREADTH, "max_depth": MAX_DEPTH},
            "batch_averages": batch_avg, "samples": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[Result saved] {out}")


if __name__ == "__main__":
    interactive_loop()

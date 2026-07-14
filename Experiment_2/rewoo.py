"""
Experiment_2/rewoo.py
ReWOO (Reasoning WithOut Observation) experimental plan generator
First create a complete reasoning plan (with placeholders) -> parallel evidence collection -> fill evidence to generate final plan
Generator: DeepSeek V4 Pro  |  Judges: DeepSeek-chat + GPT-5.4
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

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gen_lcot", "data", "train.jsonl"
)


# =========================================================
# ReWOO Prompts
# =========================================================

# Phase 1: Planner — generate reasoning plan (with placeholders), no observation performed
PLANNER_PROMPT = """You are an expert research planner using the ReWOO (Reasoning WithOut Observation) method.

Research Question:
{question}

Your task: Create a detailed reasoning PLAN for designing a complete experimental plan.
The plan should identify what EVIDENCE and INFORMATION is needed at each step,
using placeholders like #E{{num}} to mark where evidence will be inserted later.

Step 1: Identify what knowledge/evidence is needed:
- What prior methods should be referenced? -> #E1
- What datasets are appropriate? -> #E2
- What evaluation metrics are standard? -> #E3
- What baselines should be compared? -> #E4
- What are the key challenges? -> #E5

Step 2: Outline the reasoning steps to go from evidence -> experimental plan.

Step 3: For each piece of evidence needed, craft a specific SEARCH QUERY or sub-question that
would retrieve that evidence.

Output format (JSON):
{{
  "evidence_needed": [
    {{"id": "E1", "description": "what evidence this is", "query": "specific search query to find this evidence"}},
    {{"id": "E2", ...}},
    ...
  ],
  "reasoning_plan": [
    "Step 1: Using #E1 and #E2, define the research objective...",
    "Step 2: Based on #E3 and #E4, design the experimental setup...",
    "Step 3: With #E5, address potential challenges...",
    ...
  ]
}}

Make the queries concrete and searchable. The reasoning plan should have 4-6 steps."""


# Phase 2: Worker — parallel evidence collection (here: search dataset + call LLM to answer sub-questions)
WORKER_SEARCH_PROMPT = """You are a research assistant gathering evidence for an experimental plan.

Original Research Question:
{question}

The planner needs the following evidence: "{description}"

Search Query: {query}

Search the knowledge base (the research dataset) for relevant information.
Return the most relevant findings that address this specific evidence need.
Be concise but thorough — this evidence will be plugged into a reasoning plan."""


# Phase 3: Solver — fill gathered evidence into the reasoning plan to generate final plan
SOLVER_PROMPT = """You are an expert AI research scientist. The planner has created a reasoning plan
with placeholders, and the worker has gathered all the needed evidence. Now SOLVE the original problem.

Original Research Question:
{question}

REASONING PLAN (with placeholders):
---
{reasoning_plan}
---

GATHERED EVIDENCE:
---
{evidence}
---

Instructions:
1. Replace each placeholder (#E1, #E2, ...) with the corresponding evidence
2. Follow the reasoning plan to generate a complete experimental plan
3. If any evidence is insufficient, use your best judgment to fill gaps
4. Integrate everything into a unified, coherent document

Generate a complete experimental plan including:
- Research Objective
- Related Work & Background (from evidence)
- Proposed Methodology
- Experimental Design & Datasets (from evidence)
- Evaluation Metrics & Baselines (from evidence)
- Expected Outcomes & Limitations

The plan should be comprehensive, rigorous, and ready for execution."""


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


def extract_assistant_content(sample: Dict) -> str:
    for msg in sample.get("messages", []):
        if msg["role"] == "assistant":
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
# Dataset Search (replicate search_dataset logic)
# =========================================================

_dataset_cache: Optional[List[Dict]] = None


def get_dataset() -> List[Dict]:
    global _dataset_cache
    if _dataset_cache is None:
        _dataset_cache = load_dataset()
    return _dataset_cache


def search_dataset_local(query: str, top_k: int = 3) -> str:
    """Search relevant cases in dataset, return formatted evidence text"""
    dataset = get_dataset()
    query_lower = query.lower()
    keywords = query_lower.split()
    scored = []

    for i, sample in enumerate(dataset):
        user_content = extract_user_content(sample).lower()
        assistant_content = extract_assistant_content(sample).lower()
        score = sum(
            user_content.count(kw) * 2 + assistant_content.count(kw)
            for kw in keywords
        )
        if score > 0:
            scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, idx in scored[:top_k]:
        sample = dataset[idx]
        assistant_text = extract_assistant_content(sample)
        if len(assistant_text) > 600:
            assistant_text = assistant_text[:600] + "..."
        results.append(f"[Relevance: {score}] {assistant_text}")

    if not results:
        return "No directly relevant case studies found. Use general best practices."

    return "\n\n---\n\n".join(results)


# =========================================================
# LangGraph State
# =========================================================

class ReWOOState(TypedDict):
    research_topic: str
    evidence_needed: List[Dict]       # [{id, description, query}]
    reasoning_plan: List[str]         # reasoning steps with #E placeholders
    evidence_results: List[Dict]      # [{id, description, evidence}]
    plan: str                         # final solved plan
    scores: List[Dict]
    final_report: str


# =========================================================
# Build ReWOO Pipeline
# =========================================================

def build_rewoo_graph() -> StateGraph:

    # =====================
    # Node 1: Planner — create reasoning plan + placeholders
    # =====================
    def planner_node(state: ReWOOState) -> Dict:
        print("\n" + "=" * 60)
        print("[Phase 1] Planner — create complete reasoning plan + evidence placeholders (no observation yet)")
        print("=" * 60)

        question = state["research_topic"]
        raw = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are a ReWOO planner. Create a reasoning plan with evidence placeholders.",
            PLANNER_PROMPT.format(question=question),
            temperature=0.3
        )

        evidence_needed = []
        reasoning_plan = []

        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                evidence_needed = data.get("evidence_needed", [])
                reasoning_plan = data.get("reasoning_plan", [])
        except json.JSONDecodeError:
            print("  [!] JSON parse failed, using defaults")
            evidence_needed = [
                {"id": "E1", "description": "Prior methods and approaches in this domain",
                 "query": "What are the state-of-the-art methods for this problem?"},
                {"id": "E2", "description": "Standard datasets and benchmarks",
                 "query": "What datasets are commonly used to evaluate approaches in this area?"},
                {"id": "E3", "description": "Evaluation metrics and best practices",
                 "query": "What evaluation metrics are appropriate for this type of research?"},
                {"id": "E4", "description": "Common challenges and failure modes",
                 "query": "What are the key challenges and limitations in this research area?"},
            ]
            reasoning_plan = [
                "Step 1: Using evidence #E1, establish the research context and identify the gap.",
                "Step 2: Using evidence #E2, select appropriate datasets and benchmarks for evaluation.",
                "Step 3: Using evidence #E3, define the evaluation protocol and success criteria.",
                "Step 4: Using evidence #E4, anticipate challenges and design mitigations.",
                "Step 5: Synthesize all evidence into a complete experimental plan.",
            ]

        print(f"  Evidence needed: {len(evidence_needed)} items")
        for e in evidence_needed:
            print(f"    {e['id']}: {e['description'][:60]}... (query: {e.get('query', '')[:50]}...)")
        print(f"  Reasoning plan: {len(reasoning_plan)} steps")

        return {"evidence_needed": evidence_needed, "reasoning_plan": reasoning_plan}

    # =====================
    # Node 2: Worker — parallel gather all evidence
    # =====================
    def worker_node(state: ReWOOState) -> Dict:
        evidence_needed = state.get("evidence_needed", [])
        question = state["research_topic"]

        print("\n" + "=" * 60)
        print(f"[Phase 2] Worker — parallel gather {len(evidence_needed)} evidence items (no observation overhead)")
        print("=" * 60)

        def gather_single(ev: Dict) -> Dict:
            """Gather single evidence item: first search dataset, then use LLM for targeted answer"""
            eid = ev["id"]
            desc = ev.get("description", "")
            query = ev.get("query", desc)

            print(f"  [{eid}] Gathering: {desc[:60]}...")

            # 1. Search dataset
            ds_results = search_dataset_local(query, top_k=2)

            # 2. Summarize with LLM
            summary = call_llm(
                deepseek_client, GENERATOR_MODEL,
                "You are gathering specific evidence for an experimental plan design.",
                WORKER_SEARCH_PROMPT.format(
                    question=question, description=desc, query=query,
                ),
                temperature=0.2
            )

            evidence_text = f"""Evidence from Dataset Search:
{ds_results}

LLM Analysis:
{summary}"""

            print(f"  [{eid}] Done ({len(evidence_text)} chars)")
            return {"id": eid, "description": desc, "evidence": evidence_text}

        evidence_results = []

        # If only 1 item, call directly; else ThreadPool
        if len(evidence_needed) <= 1:
            for ev in evidence_needed:
                evidence_results.append(gather_single(ev))
        else:
            with ThreadPoolExecutor(max_workers=min(len(evidence_needed), 4)) as pool:
                futures = {pool.submit(gather_single, ev): ev["id"] for ev in evidence_needed}
                for future in as_completed(futures):
                    evidence_results.append(future.result())
            # Sort by E1, E2, ...
            evidence_results.sort(key=lambda x: x["id"])

        print(f"\n  -> Collection complete: {len(evidence_results)} evidence items")
        return {"evidence_results": evidence_results}

    # =====================
    # Node 3: Solver — fill evidence into plan, generate final plan
    # =====================
    def solver_node(state: ReWOOState) -> Dict:
        print("\n" + "=" * 60)
        print("[Phase 3] Solver — fill evidence into reasoning plan, generate final plan")
        print("=" * 60)

        question = state["research_topic"]
        reasoning_plan = state.get("reasoning_plan", [])
        evidence_results = state.get("evidence_results", [])

        # Format reasoning plan
        plan_text = "\n".join(f"  {step}" for step in reasoning_plan)

        # Format evidence
        evidence_text = "\n\n---\n\n".join(
            f"## Evidence {e['id']}: {e['description']}\n{e['evidence']}"
            for e in evidence_results
        )

        plan = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are a ReWOO solver. Fill in the reasoning plan with gathered evidence.",
            SOLVER_PROMPT.format(
                question=question,
                reasoning_plan=plan_text,
                evidence=evidence_text,
            ),
            temperature=0.3
        )

        print(f"  -> Final plan: {len(plan)} chars")
        return {"plan": plan}

    # =====================
    # Node 4: Scoring
    # =====================
    def score_node(state: ReWOOState) -> Dict:
        print("\n" + "=" * 60)
        print("[Score] Dual-Judge Evaluation")
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
    def report_node(state: ReWOOState) -> Dict:
        print("\n" + "=" * 60)
        print("[Report] Generating ReWOO final report")
        print("=" * 60)

        topic = state["research_topic"]
        plan = state.get("plan", "")
        scores = state.get("scores", [])
        evidence_needed = state.get("evidence_needed", [])
        evidence_results = state.get("evidence_results", [])
        reasoning_plan = state.get("reasoning_plan", [])

        display_plan = plan[:5000] + "\n\n...(truncated)" if len(plan) > 5000 else plan

        lines = [
            "=" * 70,
            "          ReWOO Experimental Plan Report",
            "=" * 70, "",
            f"Research Topic: {topic[:300]}", "",
            "-" * 70,
            "ReWOO Pipeline: Planner(reasoning plan + placeholders) -> Worker(parallel evidence collect) -> Solver(fill evidence + generate)",
            "-" * 70, "",
            f"Evidence placeholders: {len(evidence_needed)} items",
        ]
        for e in evidence_needed:
            ev_result = next((r for r in evidence_results if r["id"] == e["id"]), None)
            ev_len = len(ev_result["evidence"]) if ev_result else 0
            lines.append(f"  {e['id']}: {e['description'][:60]}... ({ev_len} chars)")

        lines += [
            "", "Reasoning Plan (with #E placeholders):",
        ]
        for step in reasoning_plan:
            lines.append(f"  {step[:120]}")

        lines += ["", "-" * 70, "Final Plan", "-" * 70, display_plan or "Failed to generate"]

        if scores:
            lines += ["", "-" * 70, "Quality Assessment", "-" * 70]
            for s in scores:
                c = s["combined"]
                lines.append(
                    f"  DS(R={s['ds'].get('reliability',0)} I={s['ds'].get('innovation',0)}) "
                    f"GPT(R={s['gpt'].get('reliability',0)} I={s['gpt'].get('innovation',0)}) "
                    f"-> Combined R={c.get('reliability',0)} I={c.get('innovation',0)} [{s.get('verdict','')}]"
                )

        lines += ["", f"Worker parallel evidence collection: {len(evidence_needed)} items parallelizable", "=" * 70]
        return {"final_report": "\n".join(lines)}

    # =====================
    # Assemble graph (linear DAG, Worker internally parallel)
    # =====================
    workflow = StateGraph(ReWOOState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("worker", worker_node)
    workflow.add_node("solver", solver_node)
    workflow.add_node("score", score_node)
    workflow.add_node("report", report_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "worker")
    workflow.add_edge("worker", "solver")
    workflow.add_edge("solver", "score")
    workflow.add_edge("score", "report")
    workflow.add_edge("report", END)

    return workflow


# =========================================================
# Compile & Run
# =========================================================

def compile_agent():
    return build_rewoo_graph().compile(checkpointer=MemorySaver())


def run_agent(question: str, verbose: bool = True) -> Dict[str, Any]:
    app = compile_agent()
    initial: ReWOOState = {
        "research_topic": question, "evidence_needed": [], "reasoning_plan": [],
        "evidence_results": [], "plan": "", "scores": [], "final_report": "",
    }
    config = {"configurable": {"thread_id": f"rewoo-{int(time.time())}"}}

    print(f"\n{'*'*70}")
    print(f"ReWOO: {question[:200]}")
    print(f"{'*'*70}")
    print("Pipeline: Planner(plan+placeholders) -> Worker(parallel evidence gather) -> Solver(fill+generate) -> Score -> Report")

    final_state = None
    for event in app.stream(initial, config, stream_mode="values"):
        final_state = event
        if verbose:
            for key in ["evidence_needed", "evidence_results", "plan"]:
                val = event.get(key, "")
                if isinstance(val, list) and val:
                    print(f"  [{key}] {len(val)} items")
                elif isinstance(val, str) and len(val) > 50:
                    print(f"  [{key}] {len(val)} chars")

    if final_state is None:
        return {"final_report": "[ERROR]", "scores": [], "plan": ""}

    return {
        "final_report": final_state.get("final_report", ""),
        "plan": final_state.get("plan", ""),
        "scores": final_state.get("scores", []),
        "evidence_results": final_state.get("evidence_results", []),
    }


# =========================================================
# Cross-Sample Average
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
║    ReWOO Experimental Plan Generator                        ║
║    Pipeline: Planner -> Worker(parallel) -> Solver -> Score ║
║    Generator: DeepSeek V4 Pro | Judges: DS-chat + GPT-5.4   ║
╚══════════════════════════════════════════════════════════════╝

Commands:
  /run <question>  -- Generate experimental plan for research question (ReWOO)
  /batch [N]       -- Batch testing (default 3 samples)
  /demo            -- Use built-in demo
  /quit            -- Exit
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
                               f"rewoo_result_{int(time.time())}.json")
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
    print(f"  ReWOO batch testing: {n} samples")
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
    print("         ReWOO Batch Testing -- Cross-Sample Average Summary")
    print("=" * 70)

    for r in sorted(batch_avg["rounds"].keys()):
        info = batch_avg["rounds"][r]
        print(f"\n  [Score]  (Covering {info['count']}/{n} samples)")
        print(f"    DeepSeek Judges:  R = {info['ds_reliability_avg']}  |  I = {info['ds_innovation_avg']}")
        print(f"    GPT Judges:       R = {info['gpt_reliability_avg']}  |  I = {info['gpt_innovation_avg']}")
        print(f"    * Combined Avg:  R = {info['combined_reliability_avg']}  |  I = {info['combined_innovation_avg']}")

    overall = batch_avg["overall"]
    print(f"\n  {'-'*60}")
    print(f"  [Overall Total]  (total {overall['total_scores']} score(s))")
    print(f"    DeepSeek Judges:  R = {overall['ds_reliability_avg']}  |  I = {overall['ds_innovation_avg']}")
    print(f"    GPT Judges:       R = {overall['gpt_reliability_avg']}  |  I = {overall['gpt_innovation_avg']}")
    print(f"    * Combined Avg:  R = {overall['combined_reliability_avg']}  |  I = {overall['combined_innovation_avg']}")
    print(f"\n{'='*70}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"rewoo_batch_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"generator": GENERATOR_MODEL, "judges": [JUDGE_MODEL_DS, JUDGE_MODEL_GPT], "samples": n},
            "batch_averages": batch_avg, "samples": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[Result saved] {out}")


if __name__ == "__main__":
    interactive_loop()

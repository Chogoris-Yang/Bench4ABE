"""
Experiment_2/least_to_most.py
Least-to-Most Experimental Plan Generator
Problem decomposition -> Solve sub-problems easy to hard -> Accumulate and synthesize final plan
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
# Least-to-Most Prompts
# =========================================================

# Stage 1: Problem Decomposition — list sub-problems from easy to hard
DECOMPOSE_PROMPT = """You are an expert at breaking down complex research problems into simpler sub-problems.

Research Question:
{question}

Your task: Decompose this research question into 3-5 sub-questions, ordered from EASIEST to HARDEST.
Each sub-question should be a self-contained, answerable component of the overall problem.

Rules:
1. Order by difficulty — the first should be the most straightforward, the last the most challenging
2. Each sub-question must build upon the previous ones
3. Together they must cover ALL aspects needed to design a complete experimental plan

Output format (JSON array):
[
  {{"id": 1, "difficulty": "easy", "question": "...", "domain": "problem_definition|dataset|method|experiment|evaluation"}},
  {{"id": 2, "difficulty": "medium", "question": "...", "domain": "..."}},
  ...
]

Explain your decomposition briefly, then output the JSON array."""


# Stage 2: Solve the simplest sub-problem (no context)
SOLVE_FIRST_PROMPT = """You are an expert AI research scientist.

Original Research Question:
{question}

You are solving the FIRST (easiest) sub-problem:

Sub-Problem {sub_id}: {sub_question}

Since this is the first step, focus on foundational aspects:
- Problem definition and scope
- Key concepts and terminology
- What exists in the literature

Provide a thorough, well-structured answer. This will be used as context for solving harder sub-problems."""


# Stage 3-N: Solve subsequent sub-problems (accumulated context)
SOLVE_NEXT_PROMPT = """You are an expert AI research scientist.

Original Research Question:
{question}

You have already solved these sub-problems:
---
{previous_solutions}
---

Now solve the NEXT sub-problem:

Sub-Problem {sub_id}: {sub_question}

Build on the previous solutions. Do not repeat what was already established.
Focus on what's NEW in this sub-problem and how it connects to previous work.

Provide a thorough, well-structured answer."""


# Final stage: Synthesize all sub-problem solutions -> Complete experimental plan
SYNTHESIZE_PROMPT = """You are an expert AI research scientist.

Original Research Question:
{question}

Below are solutions to sub-problems, solved from easiest to hardest:

---
{all_solutions}
---

Your task: SYNTHESIZE these into one complete, cohesive experimental plan.

Do NOT simply concatenate the solutions. Integrate them into a single flowing document:

1. Research Objective (from sub-problems about problem definition)
2. Related Work & Background (from foundational sub-problems)
3. Proposed Methodology (from method sub-problems)
4. Experimental Design (from experiment sub-problems)
5. Evaluation Plan (from evaluation sub-problems)
6. Expected Outcomes & Limitations

The final plan should be comprehensive, rigorous, and ready for execution.
Ensure internal consistency — every component should reference and support the others."""


# Scoring
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

class LeastToMostState(TypedDict):
    research_topic: str
    sub_questions: List[Dict]       # [{id, difficulty, question, domain}]
    solutions: List[Dict]           # [{sub_id, question, answer}]
    current_sub_index: int          # current sub-problem index being solved
    synthesized_plan: str           # final synthesized plan
    scores: List[Dict]              # scoring history
    final_report: str


# =========================================================
# Build Least-to-Most Pipeline
# =========================================================

def build_least_to_most_graph() -> StateGraph:

    # =====================
    # Node 1: Decompose the problem
    # =====================
    def decompose(state: LeastToMostState) -> Dict:
        print("\n" + "=" * 60)
        print("[Stage 1] Decomposition — break down the research question from easy to hard")
        print("=" * 60)

        question = state["research_topic"]
        raw = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are an expert at decomposing complex problems. Output the JSON array of sub-questions.",
            DECOMPOSE_PROMPT.format(question=question),
            temperature=0.2
        )

        # Parse JSON
        sub_questions = []
        try:
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            if json_match:
                sub_questions = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            # Fallback: if parsing fails, create default decomposition
            print("  [!] JSON parse failed, using default decomposition")
            sub_questions = [
                {"id": 1, "difficulty": "easy", "question": "What is the core research problem and its scope?", "domain": "problem_definition"},
                {"id": 2, "difficulty": "medium", "question": "What datasets and resources are needed for this research?", "domain": "dataset"},
                {"id": 3, "difficulty": "medium", "question": "What methodology or approach should be used?", "domain": "method"},
                {"id": 4, "difficulty": "hard", "question": "How should experiments be designed to validate the approach?", "domain": "experiment"},
                {"id": 5, "difficulty": "hard", "question": "What evaluation metrics and benchmarks should be used?", "domain": "evaluation"},
            ]

        print(f"  -> Decomposed into {len(sub_questions)} sub-problems:")
        for sq in sub_questions:
            print(f"     [{sq['difficulty']}] Q{sq['id']}: {sq['question'][:80]}...")

        return {"sub_questions": sub_questions, "current_sub_index": 0}

    # =====================
    # Node 2: Solve the current sub-problem
    # =====================
    def solve_current(state: LeastToMostState) -> Dict:
        sub_questions = state.get("sub_questions", [])
        current_idx = state.get("current_sub_index", 0)
        solutions = list(state.get("solutions", []))

        if current_idx >= len(sub_questions):
            return {}

        sq = sub_questions[current_idx]
        print("\n" + "=" * 60)
        print(f"[Stage 2] Solving sub-problem {sq['id']}/{len(sub_questions)} [{sq['difficulty']}]")
        print(f"        {sq['question'][:100]}")
        print("=" * 60)

        question = state["research_topic"]

        if current_idx == 0:
            # First sub-problem: no context
            result = call_llm(
                deepseek_client, GENERATOR_MODEL,
                "You are an expert research scientist solving the foundational sub-problem.",
                SOLVE_FIRST_PROMPT.format(
                    question=question,
                    sub_id=sq["id"],
                    sub_question=sq["question"],
                ),
                temperature=0.3
            )
        else:
            # Sub-problems 2+: accumulate all previous answers
            prev_text = "\n\n---\n\n".join(
                f"[Sub-Problem {s['sub_id']}]: {s['question']}\n\nSolution: {s['answer']}"
                for s in solutions
            )
            result = call_llm(
                deepseek_client, GENERATOR_MODEL,
                "You are an expert research scientist building on previous solutions.",
                SOLVE_NEXT_PROMPT.format(
                    question=question,
                    previous_solutions=prev_text,
                    sub_id=sq["id"],
                    sub_question=sq["question"],
                ),
                temperature=0.3
            )

        solutions.append({
            "sub_id": sq["id"],
            "question": sq["question"],
            "difficulty": sq["difficulty"],
            "answer": result,
        })

        next_idx = current_idx + 1
        print(f"  -> Q{sq['id']} answer: {len(result)} chars  ({len(sub_questions) - next_idx} sub-problems remaining)")

        return {"solutions": solutions, "current_sub_index": next_idx}

    # =====================
    # Node 3: Synthesis
    # =====================
    def synthesize(state: LeastToMostState) -> Dict:
        print("\n" + "=" * 60)
        print("[Stage 3] Synthesis — merge all sub-problem solutions into a complete experimental plan")
        print("=" * 60)

        question = state["research_topic"]
        solutions = state.get("solutions", [])

        all_solutions_text = "\n\n---\n\n".join(
            f"## Sub-Problem {s['sub_id']} [{s['difficulty']}]: {s['question']}\n\n{s['answer']}"
            for s in solutions
        )

        plan = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are an expert synthesizing sub-problem solutions into a unified experimental plan.",
            SYNTHESIZE_PROMPT.format(
                question=question,
                all_solutions=all_solutions_text,
            ),
            temperature=0.2
        )

        print(f"  -> Synthesized plan: {len(plan)} chars")
        return {"synthesized_plan": plan}

    # =====================
    # Node 4: Dual-judge scoring
    # =====================
    def score_node(state: LeastToMostState) -> Dict:
        print("\n" + "=" * 60)
        print("[Scoring] Dual-judge evaluation of synthesized plan")
        print("=" * 60)

        plan = state.get("synthesized_plan", "")

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

        scores = [{
            "round": 1,
            "ds": ds_scores,
            "gpt": gpt_scores,
            "combined": combined,
            "verdict": verdict,
        }]

        print(f"  Combined -> R={combined['reliability']}, I={combined['innovation']} -> {verdict}")

        return {"scores": scores}

    # =====================
    # Node 5: Final report
    # =====================
    def report_node(state: LeastToMostState) -> Dict:
        print("\n" + "=" * 60)
        print("[Report] Generating Least-to-Most final report")
        print("=" * 60)

        topic = state["research_topic"]
        plan = state.get("synthesized_plan", "")
        scores = state.get("scores", [])
        sub_questions = state.get("sub_questions", [])
        solutions = state.get("solutions", [])

        display_plan = plan[:5000] + "\n\n...(truncated)" if len(plan) > 5000 else plan

        lines = [
            "=" * 70,
            "          Least-to-Most Experimental Plan Report",
            "=" * 70,
            "",
            f"Research Topic: {topic[:300]}",
            "",
            "-" * 70,
            "Pipeline: Decomposition(" + f"{len(sub_questions)} sub-problems) -> Solve sub-problems -> Synthesis",
            "-" * 70,
        ]

        # Sub-problem decomposition
        lines += ["", "Sub-problem Decomposition (Easy -> Hard):"]
        for sq in sub_questions:
            sol = next((s for s in solutions if s["sub_id"] == sq["id"]), None)
            ans_len = len(sol["answer"]) if sol else 0
            lines.append(f"  [{sq['difficulty']:6s}] Q{sq['id']}: {sq['question'][:80]}... ({ans_len} chars)")

        # Final plan
        lines += ["", "-" * 70, "Final Synthesized Plan", "-" * 70, display_plan or "Failed to generate"]

        # Scores
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
    # Route: Are there more sub-problems to solve?
    # =====================
    def route_after_solve(state: LeastToMostState) -> Literal["solve", "synthesize"]:
        current_idx = state.get("current_sub_index", 0)
        sub_questions = state.get("sub_questions", [])

        if current_idx < len(sub_questions):
            next_sq = sub_questions[current_idx]
            print(f"  -> Continue solving Q{next_sq['id']} [{next_sq['difficulty']}]")
            return "solve"
        else:
            print(f"  -> All {len(sub_questions)} sub-problems solved, entering synthesis")
            return "synthesize"

    # =====================
    # Assemble the graph
    #   decompose -> solve -> [more sub-problems?]
    #                 ^        ↓ NO
    #                 +-- YES - -> synthesize -> score -> report -> END
    # =====================
    workflow = StateGraph(LeastToMostState)

    workflow.add_node("decompose", decompose)
    workflow.add_node("solve", solve_current)
    workflow.add_node("synthesize", synthesize)
    workflow.add_node("score", score_node)
    workflow.add_node("report", report_node)

    workflow.set_entry_point("decompose")
    workflow.add_edge("decompose", "solve")

    workflow.add_conditional_edges(
        "solve", route_after_solve,
        {"solve": "solve", "synthesize": "synthesize"},
    )
    workflow.add_edge("synthesize", "score")
    workflow.add_edge("score", "report")
    workflow.add_edge("report", END)

    return workflow


# =========================================================
# Compile & Run
# =========================================================

def compile_agent():
    workflow = build_least_to_most_graph()
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def run_agent(question: str, verbose: bool = True) -> Dict[str, Any]:
    app = compile_agent()

    initial_state: LeastToMostState = {
        "research_topic": question,
        "sub_questions": [],
        "solutions": [],
        "current_sub_index": 0,
        "synthesized_plan": "",
        "scores": [],
        "final_report": "",
    }

    config = {"configurable": {"thread_id": f"l2m-{int(time.time())}"}}

    print(f"\n{'★'*70}")
    print(f"Least-to-Most Generation: {question[:200]}")
    print(f"{'★'*70}")
    print("Pipeline: Decompose -> Solve sub-problems easy to hard -> Synthesize -> Score -> Report")

    final_state = None
    for event in app.stream(initial_state, config, stream_mode="values"):
        final_state = event
        if verbose:
            for key in ["sub_questions", "solutions", "synthesized_plan"]:
                val = event.get(key, "")
                if isinstance(val, list) and val:
                    print(f"  [{key}] {len(val)} items")
                elif isinstance(val, str) and len(val) > 50:
                    print(f"  [{key}] {len(val)} chars")

    if final_state is None:
        return {"final_report": "[ERROR]", "scores": [], "solutions": []}

    return {
        "final_report": final_state.get("final_report", ""),
        "plan": final_state.get("synthesized_plan", ""),
        "scores": final_state.get("scores", []),
        "sub_questions": final_state.get("sub_questions", []),
        "solutions": final_state.get("solutions", []),
    }


# =========================================================
# Cross-Sample Average Summary
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
║    Least-to-Most Experimental Plan Generator                ║
║    Pipeline: Decompose -> Solve Easy-to-Hard -> Synthesize -> Score ║
║    Generator: DeepSeek V4 Pro | Judges: DS-chat + GPT-5.4   ║
╚══════════════════════════════════════════════════════════════╝

Commands:
  /run <question>  — Generate experimental plan for research question (Least-to-Most)
  /batch [N]       — Batch testing (default 3 samples)
  /demo            — Use built-in demo
  /quit            — Exit
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
                                       f"l2m_result_{int(time.time())}.json")
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
    print(f"  Least-to-Most Batch Testing: {n} samples")
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
    print("         Least-to-Most Batch Test — Cross-Sample Average Summary")
    print("=" * 70)

    for r in sorted(batch_avg["rounds"].keys()):
        info = batch_avg["rounds"][r]
        print(f"\n  [Scores]  (Covering {info['count']}/{n} samples)")
        print(f"    DeepSeek Judges:  R = {info['ds_reliability_avg']}  |  I = {info['ds_innovation_avg']}")
        print(f"    GPT Judges:       R = {info['gpt_reliability_avg']}  |  I = {info['gpt_innovation_avg']}")
        print(f"    * Combined Avg:  R = {info['combined_reliability_avg']}  |  I = {info['combined_innovation_avg']}")

    overall = batch_avg["overall"]
    print(f"\n  {'-'*60}")
    print(f"  [Overall Total]  ({overall['total_scores']} scores total)")
    print(f"    DeepSeek Judges:  R = {overall['ds_reliability_avg']}  |  I = {overall['ds_innovation_avg']}")
    print(f"    GPT Judges:       R = {overall['gpt_reliability_avg']}  |  I = {overall['gpt_innovation_avg']}")
    print(f"    * Combined Avg:  R = {overall['combined_reliability_avg']}  |  I = {overall['combined_innovation_avg']}")
    print(f"\n{'='*70}")

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f"l2m_batch_{int(time.time())}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"generator": GENERATOR_MODEL, "judges": [JUDGE_MODEL_DS, JUDGE_MODEL_GPT], "samples": n},
            "batch_averages": batch_avg, "samples": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[Result saved] {output_path}")


if __name__ == "__main__":
    interactive_loop()

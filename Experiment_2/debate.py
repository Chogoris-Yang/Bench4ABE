"""
Experiment_2/debate.py
Debate (Multi-Agent Debate) experimental plan generator -- Du et al. 2023
Agent A (conservative, rigor-focused) vs Agent B (innovative, bold) -> mutual critique -> each improves -> synthesize best
Generator Model: DeepSeek V4 Pro  |  Judge Model: DeepSeek-chat + GPT-5.4

Debate vs Self-Refine:
  Self-Refine is single-model self-introspection
  Debate has two Agents with different personas opposing each other, with stronger driving force from diverse perspectives
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

MAX_ROUNDS = 2  # Max debate rounds

# =========================================================
# Dataset Path
# =========================================================
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gen_lcot", "data", "train.jsonl"
)


# =========================================================
# Debate Prompts
# =========================================================

# Agent A: conservative, rigor-focused researcher -- emphasizes methodological reliability and reproducibility
AGENT_A_PROMPT = """You are Research Agent A (conservative, rigor-focused).
Generate an experimental plan emphasizing methodological soundness, reproducibility,
and thorough validation.

Research Question:
{question}

Generate a complete experimental plan. Your approach should prioritize:
- Solid theoretical grounding and formal justification
- Rigorous experimental design with extensive ablation studies
- Conservative but reliable claims
- Clear reproducibility protocol (hyperparameters, seeds, hardware)

{opponent_feedback}"""

# Agent B: innovative researcher -- emphasizes novelty and cross-disciplinary insights
AGENT_B_PROMPT = """You are Research Agent B (creative, innovation-focused).
Generate an experimental plan emphasizing novelty, cross-disciplinary insights,
and bold contributions.

Research Question:
{question}

Generate a complete experimental plan. Your approach should prioritize:
- Novel methodology or unconventional approach
- Creative evaluation strategies
- Bold, high-impact claims with clear innovation
- Cross-domain analogies and insights

{opponent_feedback}"""

# Mutual Critique
CRITIQUE_PROMPT = """You are critiquing an opponent's experimental plan in a structured research debate.
Your goal is to identify genuine weaknesses, not just disagree.

Your own approach (for reference): {my_approach_summary}

Opponent's plan to critique:
{opponent_plan}

Identify:
1. The 2-3 most CRITICAL weaknesses in the opponent's plan
2. What YOUR approach does better on each weakness
3. Specific, actionable suggestions for the opponent

Be rigorous but constructive. Output a structured critique."""

# Revise own plans
REVISE_PROMPT = """You are revising your experimental plan after receiving critique from an opponent
in a research debate.

Your original plan:
{my_plan}

Opponent's critique of your plan:
{critique}

Revise your plan:
- Address all valid criticisms
- Strengthen weak points
- But KEEP your core philosophy (don't just copy the opponent's approach)
- Output the COMPLETE revised plan."""

# Synthesize advantages of both plans
SYNTHESIZE_PROMPT = """You are synthesizing two debated experimental plans into one optimal plan.

Plan A (rigor-focused, conservative):
{plan_a}

Plan B (innovation-focused, creative):
{plan_b}

Research Question:
{question}

Combine the best of both:
- Take Plan A's methodological rigor, ablation design, reproducibility
- Take Plan B's novel ideas, creative evaluation, bold contributions
- Create ONE superior, unified experimental plan that is BOTH rigorous AND innovative.

Output the complete synthesized plan."""

# Scoring
SCORING_PROMPT = """Rate: Reliability: xx\nInnovation: xx"""


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
    """Parse Reliability and Innovation scores from scoring text"""
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
    """Perform dual-judge scoring on a single plan, return {ds, gpt, combined}"""
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

class DebateState(TypedDict):
    """Multi-Agent Debate state"""
    research_topic: str          # User original question
    plan_a: str                  # Agent A's plan (rigor-focused)
    plan_b: str                  # Agent B's plan (innovation-focused)
    critique_a_to_b: str         # A's critique of B
    critique_b_to_a: str         # B's critique of A
    revised_a: str               # A's revised plan
    revised_b: str               # B's revised plan
    synthesized_plan: str        # Synthesized final plan from both revised plans
    round_num: int               # Current debate round
    scores: List[Dict]           # Score History
    final_report: str            # Final Output Report


# =========================================================
# Build Multi-Agent Debate Pipeline
# =========================================================

def build_debate_graph() -> StateGraph:
    """
    Build Multi-Agent Debate state graph.

    Graph structure:
        __start__ -> generate_both -> critique -> revise -> synthesize -> score
                       ^                                               |
                       +--- [NEEDS_IMPROV & rounds left] --------------+
                                                      | OK
                                                    report -> END

    Each generate/critique/revise phase internally has two parallel calls (Agent A + Agent B)
    """

    # =====================
    # Node 1: Generate Both -- two Agents each generate their own plan
    # =====================
    def generate_both(state: DebateState) -> Dict:
        round_num = state.get("round_num", 0) + 1
        question = state["research_topic"]

        # If second round, carry opponent's previous critique
        fb_a = state.get("critique_b_to_a", "")
        fb_b = state.get("critique_a_to_b", "")
        fb_a_text = f"\n\nOPPONENT'S LAST CRITIQUE OF YOUR PLAN (address these issues):\n{fb_a}" if fb_a else ""
        fb_b_text = f"\n\nOPPONENT'S LAST CRITIQUE OF YOUR PLAN (address these issues):\n{fb_b}" if fb_b else ""

        print("\n" + "=" * 60)
        print(f"[Debate Round {round_num}] Agent A (rigor) & Agent B (innovation) each generate their plan")
        print("=" * 60)

        # ---- Parallel generation ----
        def gen_role(label: str, prompt_template: str, fb_text: str) -> str:
            print(f"  [{label}] Generating...")
            temp = 0.3 if label == "Agent A" else 0.4  # B slightly higher temp to encourage innovation
            return call_llm(
                deepseek_client, GENERATOR_MODEL,
                f"You are {label}, a research scientist with a distinct philosophy.",
                prompt_template.format(question=question, opponent_feedback=fb_text),
                temperature=temp
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            future_a = pool.submit(gen_role, "Agent A", AGENT_A_PROMPT, fb_a_text)
            future_b = pool.submit(gen_role, "Agent B", AGENT_B_PROMPT, fb_b_text)
            plan_a = future_a.result()
            plan_b = future_b.result()

        print(f"  -> Plan A (rigor): {len(plan_a)} chars  |  Plan B (innovation): {len(plan_b)} chars")
        return {"plan_a": plan_a, "plan_b": plan_b, "round_num": round_num}

    # =====================
    # Node 2: Critique -- mutual critique
    # =====================
    def critique(state: DebateState) -> Dict:
        plan_a = state.get("plan_a", "")
        plan_b = state.get("plan_b", "")

        print("\n" + "=" * 60)
        print("[Critique] Agent A critiques Plan B  |  Agent B critiques Plan A")
        print("=" * 60)

        # ---- Parallel critique ----
        def critic_one(label: str, my_plan: str, opp_plan: str) -> str:
            summary = my_plan[:300]  # Use own plan summary as reference
            print(f"  [{label}] Critiquing opponent...")
            return call_llm(
                deepseek_client, GENERATOR_MODEL,
                f"You are {label} critiquing your debate opponent.",
                CRITIQUE_PROMPT.format(
                    my_approach_summary=summary,
                    opponent_plan=opp_plan[:3000],
                ),
                temperature=0.2
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            future_ca = pool.submit(critic_one, "Agent A", plan_a, plan_b)
            future_cb = pool.submit(critic_one, "Agent B", plan_b, plan_a)
            critique_a_to_b = future_ca.result()
            critique_b_to_a = future_cb.result()

        print(f"  -> A's critique of B: {len(critique_a_to_b)} chars")
        print(f"  -> B's critique of A: {len(critique_b_to_a)} chars")

        return {"critique_a_to_b": critique_a_to_b, "critique_b_to_a": critique_b_to_a}

    # =====================
    # Node 3: Revise -- each revises own plan based on opponent's critique
    # =====================
    def revise(state: DebateState) -> Dict:
        plan_a = state.get("plan_a", "")
        plan_b = state.get("plan_b", "")
        critique_b_to_a = state.get("critique_b_to_a", "")
        critique_a_to_b = state.get("critique_a_to_b", "")

        print("\n" + "=" * 60)
        print("[Revise] Each revises own plan based on opponent's critique")
        print("=" * 60)

        # ---- Parallel revision ----
        def revise_one(label: str, plan: str, critique: str) -> str:
            print(f"  [{label}] Revising plan...")
            return call_llm(
                deepseek_client, GENERATOR_MODEL,
                f"You are {label} revising your plan after debate feedback.",
                REVISE_PROMPT.format(my_plan=plan, critique=critique),
                temperature=0.3
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            future_ra = pool.submit(revise_one, "Agent A", plan_a, critique_b_to_a)
            future_rb = pool.submit(revise_one, "Agent B", plan_b, critique_a_to_b)
            revised_a = future_ra.result()
            revised_b = future_rb.result()

        print(f"  -> Revised A: {len(revised_a)} chars  |  Revised B: {len(revised_b)} chars")
        return {"revised_a": revised_a, "revised_b": revised_b}

    # =====================
    # Node 4: Synthesize -- synthesize both revised plans
    # =====================
    def synthesize(state: DebateState) -> Dict:
        ra = state.get("revised_a", "") or state.get("plan_a", "")
        rb = state.get("revised_b", "") or state.get("plan_b", "")
        question = state["research_topic"]

        print("\n" + "=" * 60)
        print("[Synthesize] Synthesize advantages of both debated plans")
        print("=" * 60)

        synthesized = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "Synthesize two debated plans into one superior plan.",
            SYNTHESIZE_PROMPT.format(
                question=question,
                plan_a=ra[:3000],
                plan_b=rb[:3000],
            ),
            temperature=0.2
        )

        print(f"  -> Synthesized plan: {len(synthesized)} chars")
        return {"synthesized_plan": synthesized}

    # =====================
    # Node 5: Dual-Judge Scoring
    # =====================
    def score_node(state: DebateState) -> Dict:
        round_num = state.get("round_num", 1)
        plan = state.get("synthesized_plan", "")

        print("\n" + "=" * 60)
        print(f"[Score] Round {round_num} debate, evaluate synthesized plan")
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

        return {"scores": scores}

    # =====================
    # Node 6: Final Report
    # =====================
    def report(state: DebateState) -> Dict:
        print("\n" + "=" * 60)
        print("[Report] Generate Debate Final Report")
        print("=" * 60)

        synthesized = state.get("synthesized_plan", "")
        scores = state.get("scores", [])
        round_num = state.get("round_num", 1)

        display = synthesized[:5000] + "\n\n...(truncated)" if len(synthesized) > 5000 else synthesized

        lines = [
            "=" * 70,
            "          Debate (Multi-Agent Debate) Experimental Plan Report",
            "=" * 70,
            "",
            f"Research Topic: {state['research_topic'][:300]}",
            "",
            "-" * 70,
            f"Agent A: conservative, rigor-focused (methodological reliability priority)",
            f"Agent B: innovative, bold (novelty and impact priority)",
            f"Debate Rounds: {round_num}  |  Workflow: generate -> mutual critique -> revise -> synthesize -> score",
            "-" * 70,
            "",
            "Score History (Synthesized Plan):",
        ]
        for s in scores:
            lines.append(
                f"  R{s['round']}: DS(R={s['ds'].get('reliability', 0)} I={s['ds'].get('innovation', 0)}) "
                f"GPT(R={s['gpt'].get('reliability', 0)} I={s['gpt'].get('innovation', 0)}) "
                f"-> Combined R={s['combined'].get('reliability', 0)} I={s['combined'].get('innovation', 0)}"
            )

        # Improvement Delta
        if len(scores) >= 2:
            delta_r = round(scores[-1]["combined"]["reliability"] - scores[0]["combined"]["reliability"], 1)
            delta_i = round(scores[-1]["combined"]["innovation"] - scores[0]["combined"]["innovation"], 1)
            lines.append(f"\n  Debate Improvement: Delta R = +{delta_r}  Delta I = +{delta_i}")

        lines += [
            "",
            "-" * 70,
            "Final Synthesized Plan (rigor + innovation)",
            "-" * 70,
            "",
            display or "Failed to generate plan",
            "",
            "=" * 70,
        ]
        return {"final_report": "\n".join(lines)}

    # =====================
    # Routing Condition: continue debate or finish?
    # =====================
    def route_after_score(state: DebateState) -> Literal["generate", "report"]:
        round_num = state.get("round_num", 1)
        scores = state.get("scores", [])

        if round_num >= MAX_ROUNDS:
            print(f"  -> Max debate rounds reached ({MAX_ROUNDS}), output report")
            return "report"
        if scores and scores[-1]["verdict"] == "GOOD":
            print(f"  -> Synthesized plan score acceptable, output report")
            return "report"
        print(f"  -> Score insufficient, entering next debate round with opponent critique...")
        return "generate"

    # =====================
    # Assemble Graph
    # =====================
    workflow = StateGraph(DebateState)

    workflow.add_node("generate", generate_both)
    workflow.add_node("critique", critique)
    workflow.add_node("revise", revise)
    workflow.add_node("synthesize", synthesize)
    workflow.add_node("score", score_node)
    workflow.add_node("report", report)

    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "critique")
    workflow.add_edge("critique", "revise")
    workflow.add_edge("revise", "synthesize")
    workflow.add_edge("synthesize", "score")
    workflow.add_conditional_edges(
        "score", route_after_score,
        {"generate": "generate", "report": "report"},
    )
    workflow.add_edge("report", END)

    return workflow


# =========================================================
# Compile & Run
# =========================================================

def compile_agent():
    """Compile Debate Agent graph"""
    return build_debate_graph().compile(checkpointer=MemorySaver())


def run_agent(question: str, verbose: bool = True) -> Dict[str, Any]:
    """Run Multi-Agent Debate.
    Returns: {final_report, plan, scores}
    """
    app = compile_agent()
    initial: DebateState = {
        "research_topic": question,
        "plan_a": "",
        "plan_b": "",
        "critique_a_to_b": "",
        "critique_b_to_a": "",
        "revised_a": "",
        "revised_b": "",
        "synthesized_plan": "",
        "round_num": 0,
        "scores": [],
        "final_report": "",
    }
    config = {"configurable": {"thread_id": f"debate-{int(time.time())}"}}

    print(f"\n{'★' * 70}")
    print(f"Debate: {question[:200]}")
    print(f"{'★' * 70}")
    print("Workflow: A+B generate -> mutual critique -> revise -> synthesize -> score -> [re-debate]")

    final_state = None
    for event in app.stream(initial, config, stream_mode="values"):
        final_state = event

    if final_state is None:
        return {"final_report": "[ERROR]", "plan": "", "scores": []}

    return {
        "final_report": final_state.get("final_report", ""),
        "plan": final_state.get("synthesized_plan", ""),
        "scores": final_state.get("scores", []),
    }


# =========================================================
# Cross-Sample Average Scores
# =========================================================

def _avg(lst: List[float]) -> float:
    return round(sum(lst) / len(lst), 1) if lst else 0.0


def compute_batch_averages(all_results: List[Dict]) -> Dict:
    """Compute cross-sample average scores for batch testing (grouped by round)"""
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
║    Debate (Multi-Agent Debate) Experimental Plan Generator    ║
║    Agent A(rigor) vs Agent B(innovation) -> critique -> revise -> synthesize ║
║    Generator: DeepSeek V4 Pro | Judges: DS-chat + GPT-5.4     ║
╚══════════════════════════════════════════════════════════════╝

Commands:
  /run <question>    -- Generate experiment plan for research question (Multi-Agent Debate)
  /batch [N]     -- Batch Test (default 3)
  /demo          -- use built-in demo
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
                               f"debate_result_{int(time.time())}.json")
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
    """Batch Test"""
    dataset = load_dataset(sample_size=n)
    print(f"\n{'#' * 70}")
    print(f"  Debate Batch Test：{n} samples")
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
    print("         Debate Batch Test — Cross-Sample Average ScoresSummary")
    print("=" * 70)

    for ri in sorted(batch_avg["rounds"].keys()):
        info = batch_avg["rounds"][ri]
        print(f"\n  [R{ri}]  (Covering {info['count']} scoring events)")
        print(f"    * Reliability = {info['reliability_avg']}  |  Innovation = {info['innovation_avg']}")

    overall = batch_avg["overall"]
    print(f"\n  {'─' * 60}")
    print(f"  [Overall Total]  (Total {overall['total']} scoring events)")
    print(f"    * Reliability = {overall['reliability_avg']}  |  Innovation = {overall['innovation_avg']}")
    print(f"\n{'=' * 70}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"debate_batch_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"generator": GENERATOR_MODEL, "judges": [JUDGE_MODEL_DS, JUDGE_MODEL_GPT], "samples": n},
            "batch_averages": batch_avg,
            "samples": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[ResultSaved] {out}")


if __name__ == "__main__":
    interactive_loop()

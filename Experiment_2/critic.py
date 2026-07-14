"""
Experiment_2/critic.py
CRITIC experimental plan generator -- Gou et al. 2024
Generate -> external tool verification (dataset search) -> self-correct based on verification results -> regenerate
Generator Model: DeepSeek V4 Pro  |  Judge Model: DeepSeek-chat + GPT-5.4

CRITIC vs Self-Refine:
  Self-Refine is pure introspective self-critique (no external info)
  CRITIC introduces external tools (dataset search) as verification signals, critique backed by evidence
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

MAX_ROUNDS = 2  # Max verification + correction rounds

# =========================================================
# Dataset Path
# =========================================================
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gen_lcot", "data", "train.jsonl"
)


# =========================================================
# CRITIC Prompts
# =========================================================

# Phase 1: Generate -- generate experimental plan (may carry previous verification feedback)
GENERATE_PROMPT = """You are an AI research scientist. Generate a complete experimental plan
for the following research question. Be rigorous and innovative.

Research Question:
{question}
{feedback}"""

# Phase 2: CRITIC Verify -- external tool verification (search dataset for evidence)
CRITIC_VERIFY_PROMPT = """You are a CRITIC verifier. Check this experimental plan against
external evidence from the research dataset (similar published cases).

PLAN TO VERIFY:
{plan}

EXTERNAL EVIDENCE (similar cases from research dataset):
{evidence}

Identify:
1. FACTUAL ERRORS: claims in the plan contradicted by evidence
2. MISSING ELEMENTS: what the evidence suggests should be included but isn't
3. IMPROVEMENT OPPORTUNITIES: where evidence shows better approaches exist

Output JSON:
{{
  "has_errors": true/false,
  "error_list": ["specific error 1", ...],
  "missing": ["missing element 1", ...],
  "improvements": ["improvement opportunity 1", ...]
}}"""

# Phase 3: Correct -- correct plan based on verification results
CORRECT_PROMPT = """You are correcting an experimental plan based on external verification results.

ORIGINAL PLAN:
{plan}

VERIFICATION RESULTS (from external evidence):
{verification}

Rewrite the plan, fixing ALL identified errors, adding all missing elements,
and incorporating all improvements. Output the COMPLETE corrected plan."""

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


def extract_assistant_content(sample: Dict) -> str:
    """Extract assistant answer from sample"""
    for msg in sample.get("messages", []):
        if msg["role"] == "assistant":
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


# =========================================================
# External Tool: dataset search verification
# =========================================================

_dataset_cache: Optional[List[Dict]] = None


def get_dataset() -> List[Dict]:
    """Get cached global dataset"""
    global _dataset_cache
    if _dataset_cache is None:
        _dataset_cache = load_dataset()
    return _dataset_cache


def search_evidence(query: str, top_k: int = 3) -> str:
    """
    CRITIC's core external tool: search relevant evidence in dataset
    Used to verify whether generated plan is well-founded or has omissions
    """
    dataset = get_dataset()
    query_lower = query.lower()
    keywords = query_lower.split()
    scored = []

    for i, sample in enumerate(dataset):
        user_text = extract_user_content(sample).lower()
        assistant_text = extract_assistant_content(sample).lower()
        score = sum(user_text.count(k) * 2 + assistant_text.count(k) for k in keywords)
        if score > 0:
            scored.append((score, i))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for sc, idx in scored[:top_k]:
        assistant_text = extract_assistant_content(dataset[idx])
        results.append(assistant_text[:600] if len(assistant_text) > 600 else assistant_text)

    return "\n---\n".join(results) if results else "No relevant evidence found."


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

class CRITICState(TypedDict):
    """CRITIC state"""
    research_topic: str          # User original question
    plan: str                    # Current plan
    evidence: str                # Evidence retrieved by external tool
    verification: Dict           # Verification results {has_errors, error_list, missing, improvements}
    round_num: int               # Current round number
    scores: List[Dict]           # Score History
    final_report: str            # Final Output Report


# =========================================================
# Build CRITIC Pipeline
# =========================================================

def build_critic_graph() -> StateGraph:
    """
    Build CRITIC state graph.

    Graph structure:
        __start__ -> generate -> verify(external tool) -> correct -> score
                       ^                                      |
                       +--- [NEEDS_IMPROV & rounds left] -----+
                                           | OK
                                         report -> END
    """

    # =====================
    # Node 1: Generate -- generate plan
    # =====================
    def generate(state: CRITICState) -> Dict:
        round_num = state.get("round_num", 0) + 1
        question = state["research_topic"]

        # If second round, use previous verification results as feedback
        prev_verification = state.get("verification", {})
        fb = ""
        if prev_verification:
            fb = (
                f"\n\nCRITICAL - Previous plan had these issues verified against external evidence:\n"
                f"{json.dumps(prev_verification, indent=2)}\n"
                f"Fix ALL issues in this new plan."
            )

        print("\n" + "=" * 60)
        print(f"[Generate] Round {round_num} generation")
        print("=" * 60)

        plan = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "Generate a rigorous experimental plan.",
            GENERATE_PROMPT.format(question=question, feedback=fb),
            temperature=0.3
        )

        print(f"  -> Plan: {len(plan)} chars")
        return {"plan": plan, "round_num": round_num}

    # =====================
    # Node 2: CRITIC Verify -- external tool verification
    # =====================
    def verify(state: CRITICState) -> Dict:
        plan = state.get("plan", "")
        question = state["research_topic"]

        print("\n" + "=" * 60)
        print("[CRITIC Verify] External tool verification -- search dataset as evidence")
        print("=" * 60)

        # Call external tool to search for evidence
        evidence = search_evidence(question, top_k=3)
        evidence_count = evidence.count("---") + 1
        print(f"  -> Retrieved {evidence_count} relevant evidence items ({len(evidence)} chars)")

        # LLM compares plan against evidence
        raw = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "Verify this plan against external evidence from the research dataset.",
            CRITIC_VERIFY_PROMPT.format(
                plan=plan[:3000],
                evidence=evidence[:2000],
            ),
            temperature=0.1
        )

        verification = {}
        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                verification = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            verification = {
                "has_errors": True,
                "error_list": ["Verification parsing failed"],
                "missing": [],
                "improvements": [],
            }

        errors = len(verification.get("error_list", []))
        missing = len(verification.get("missing", []))
        improvements = len(verification.get("improvements", []))
        print(f"  -> Found: {errors} error(s), {missing} missing, {improvements} improvement opportunities")

        return {"evidence": evidence, "verification": verification}

    # =====================
    # Node 3: Correct -- correct plan based on verification
    # =====================
    def correct(state: CRITICState) -> Dict:
        verification = state.get("verification", {})
        plan = state.get("plan", "")

        print("\n" + "=" * 60)
        print("[Correct] Correct plan based on external verification results")
        print("=" * 60)

        # If no issues found, skip correction
        if not verification.get("has_errors") and not verification.get("missing"):
            print("  -> External verification found no issues, skipping correction")
            return {}

        corrected = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "Correct the experimental plan based on external verification findings.",
            CORRECT_PROMPT.format(
                plan=plan,
                verification=json.dumps(verification, indent=2),
            ),
            temperature=0.3
        )

        print(f"  -> Corrected plan: {len(corrected)} chars")
        return {"plan": corrected}

    # =====================
    # Node 4: Dual-Judge Scoring
    # =====================
    def score_node(state: CRITICState) -> Dict:
        round_num = state.get("round_num", 1)
        plan = state.get("plan", "")

        print("\n" + "=" * 60)
        print(f"[Score] Round {round_num} evaluation (after CRITIC verification)")
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
    # Node 5: Final Report
    # =====================
    def report(state: CRITICState) -> Dict:
        print("\n" + "=" * 60)
        print("[Report] Generate CRITIC final report")
        print("=" * 60)

        plan = state.get("plan", "")
        scores = state.get("scores", [])
        verification = state.get("verification", {})

        display = plan[:5000] + "\n\n...(truncated)" if len(plan) > 5000 else plan

        lines = [
            "=" * 70,
            "          CRITIC Experimental Plan Report",
            "=" * 70,
            "",
            f"Research Topic: {state['research_topic'][:300]}",
            "",
            "-" * 70,
            "CRITIC Workflow: generate -> external verification (dataset search) -> correct -> score",
            f"External verification findings: {len(verification.get('error_list', []))} error(s), "
            f"{len(verification.get('missing', []))} missing, "
            f"{len(verification.get('improvements', []))} improvement opportunities",
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

        # Improvement Delta
        if len(scores) >= 2:
            delta_r = round(scores[-1]["combined"]["reliability"] - scores[0]["combined"]["reliability"], 1)
            delta_i = round(scores[-1]["combined"]["innovation"] - scores[0]["combined"]["innovation"], 1)
            lines.append(f"\n  External verification + correction improvement: Delta R = +{delta_r}  Delta I = +{delta_i}")

        lines += [
            "",
            "-" * 70,
            "Final Plan (corrected by external verification)",
            "-" * 70,
            "",
            display or "Failed to generate plan",
            "",
            "=" * 70,
        ]
        return {"final_report": "\n".join(lines)}

    # =====================
    # Routing Condition
    # =====================
    def route_after_score(state: CRITICState) -> Literal["generate", "report"]:
        round_num = state.get("round_num", 1)
        scores = state.get("scores", [])

        if round_num >= MAX_ROUNDS:
            print(f"  -> Max rounds reached ({MAX_ROUNDS}), output report")
            return "report"
        if scores and scores[-1]["verdict"] == "GOOD":
            print(f"  -> Score acceptable, output report")
            return "report"
        print(f"  -> Score insufficient, regenerating with verification results...")
        return "generate"

    # =====================
    # Assemble Graph
    # =====================
    workflow = StateGraph(CRITICState)

    workflow.add_node("generate", generate)
    workflow.add_node("verify", verify)
    workflow.add_node("correct", correct)
    workflow.add_node("score", score_node)
    workflow.add_node("report", report)

    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "verify")
    workflow.add_edge("verify", "correct")
    workflow.add_edge("correct", "score")
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
    """Compile CRITIC Agent graph"""
    return build_critic_graph().compile(checkpointer=MemorySaver())


def run_agent(question: str, verbose: bool = True) -> Dict[str, Any]:
    """Run CRITIC Agent.
    Returns: {final_report, plan, scores}
    """
    app = compile_agent()
    initial: CRITICState = {
        "research_topic": question,
        "plan": "",
        "evidence": "",
        "verification": {},
        "round_num": 0,
        "scores": [],
        "final_report": "",
    }
    config = {"configurable": {"thread_id": f"critic-{int(time.time())}"}}

    print(f"\n{'★' * 70}")
    print(f"CRITIC: {question[:200]}")
    print(f"{'★' * 70}")
    print("Workflow: Generate -> Verify (external dataset search) -> Correct -> Score -> [Loop]")

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
║    CRITIC Experimental Plan Generator                          ║
║    generate -> external verify (dataset search) -> correct -> score -> regenerate ║
║    Generator: DeepSeek V4 Pro | Judges: DS-chat + GPT-5.4     ║
╚══════════════════════════════════════════════════════════════╝

Commands:
  /run <question>    -- Generate experiment plan for research question (CRITIC)
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
                               f"critic_result_{int(time.time())}.json")
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
    print(f"  CRITIC Batch Test：{n} samples")
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
    print("         CRITIC Batch Test — Cross-Sample Average ScoresSummary")
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
                       f"critic_batch_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"generator": GENERATOR_MODEL, "judges": [JUDGE_MODEL_DS, JUDGE_MODEL_GPT], "samples": n},
            "batch_averages": batch_avg,
            "samples": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[ResultSaved] {out}")


if __name__ == "__main__":
    interactive_loop()

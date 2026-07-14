"""
Experiment_5/pipeline_runner.py — 4-Stage × 3-Round Pipeline.
Each question runs 3 full iterations, all stages dual-judge scored.
Output: all plans, code, scores preserved per round.
"""

import os, sys, json, time, re
from typing import Dict, List

_EXP5_DIR = os.path.dirname(os.path.abspath(__file__))
if _EXP5_DIR not in sys.path:
    sys.path.insert(0, _EXP5_DIR)

from pipeline_components import (
    call_llm, generator_client, judge_client,
    GENERATOR_MODEL, JUDGE_MODEL_1, JUDGE_MODEL_2,
    dual_judge_score, parse_score, parse_generic_score,
    load_bio_test_questions, extract_question,
    get_iteration_memory, get_prompt_memory,
    ScoreAccumulator, _avg,
)
from stages.stage1_retrieval import run_stage1
from config_exp5 import (
    BIO_TEST_PATH, RETRIEVAL_MODE, REWARD_WEIGHTS,
    GENERATED_CODE_DIR,
)

TOTAL_ROUNDS = 3
_AVOID_PAST = "Review the PAST ISSUES above. DO NOT repeat the same mistakes."

# =========================================================
# Judge Critique Generator — produces detailed feedback from scores
# =========================================================

CRITIQUE_PLAN = """You are an expert scientific reviewer. Based on the dual-judge scores below,
write a DETAILED, ACTIONABLE critique of this experimental plan.

Research Question: {question}

JUDGE SCORES:
- {judge1}: Reliability={j1_rel}, Innovation={j1_inn}
- {judge2}: Reliability={j2_rel}, Innovation={j2_inn}
- Combined: Reliability={comb_rel}, Innovation={comb_inn}

PLAN (excerpt):
---
{plan_excerpt}
---

Write a structured critique covering:
1. SPECIFIC WEAKNESSES that caused low scores (cite exact sections or missing components)
2. CONCRETE SUGGESTIONS for improvement (what to add, change, or remove)
3. WHAT WAS DONE WELL (preserve these strengths in the next iteration)
4. ONE KEY INSIGHT that the next round MUST incorporate

Be direct and specific. The next iteration will use this critique to improve."""

CRITIQUE_PROMPT = """You are an expert prompt engineer. Analyze why this prompt produced suboptimal results.

Question: {question}
Stage2 Plan Scores: Reliability={comb_rel}, Innovation={comb_inn}

PROMPT USED:
---
{prompt_text}
---

Write a detailed critique:
1. What the prompt LACKED (missing instructions, vague phrasing)
2. What INSTRUCTIONS would have produced a better plan
3. CONCRETE PROMPT REWRITE SUGGESTIONS

Be specific — the next iteration's prompt will incorporate this feedback."""

CRITIQUE_CODE = """You are a senior code reviewer. Critique this generated experiment code.

Question: {question}
Stage4 Code Scores: Reliability={comb_rel}, Innovation={comb_inn}

CODE EXCERPT:
---
{code_excerpt}
---

Write a detailed critique:
1. CODE QUALITY ISSUES (bugs, missing imports, logic errors, incomplete sections)
2. MISSING FUNCTIONALITY (what the code should do but doesn't)
3. IMPROVEMENTS for the next iteration

Be specific about lines/modules that need fixing."""

# =========================================================
# Plan generation prompts (Round 1 vs Round 2+)
# =========================================================

GEN_INITIAL = """You are an expert AI research scientist. Generate a complete experimental plan.

Research Question: {question}
Retrieved Knowledge: {context}

Include: 1.Research Objective 2.Proposed Methodology 3.Experimental Design 4.Datasets & Metrics 5.Limitations.
Be rigorous and innovative. Output the complete plan."""

GEN_REFINE = """You are refining an experimental plan based on dual-judge feedback and detailed critiques.

Research Question: {question}
Retrieved Knowledge: {context}

PREVIOUS PLAN (Round {prev_round}):
---
{prev_plan}
---

JUDGE SCORES (Round {prev_round}):
- {judge1}: Reliability={j1_rel}, Innovation={j1_inn}
- {judge2}: Reliability={j2_rel}, Innovation={j2_inn}
- Combined: Reliability={comb_rel}, Innovation={comb_inn}

DETAILED JUDGE CRITIQUES FROM ALL PAST ROUNDS:
---
{past_iteration_issues}
---

PROMPT IMPROVEMENT HISTORY:
---
{past_prompt_issues}
---

CRITICAL INSTRUCTIONS:
1. Read EVERY critique above carefully — each one identifies a specific failure
2. Address ALL weaknesses raised by the judges in past rounds
3. {avoid_msg}
4. Produce a SIGNIFICANTLY improved plan — not minor edits
5. If innovation scores are low, try a fundamentally different methodological approach
6. If reliability scores are low, add validation steps, formal definitions, and reproducibility measures

Generate the improved experimental plan. Be specific about WHAT changed and WHY."""

# =========================================================
# Stage 3: Prompt Analysis (per round)
# =========================================================

PROMPT_ANALYSIS = """Analyze the generation prompt used to produce this experimental plan.

Research Question: {question}
Plan Reliability={comb_rel}, Innovation={comb_inn}

PROMPT USED:
---
{prompt_used}
---

Past prompt issues:
{past_prompt_issues}

Identify prompt weaknesses that caused low scores and propose an improved prompt.
Output:
Analysis: (what went wrong with the prompt)
Improved_Prompt: (the revised complete prompt)"""

# =========================================================
# Stage 4: Code Generation (per round)
# =========================================================

CODE_GEN = """You are an expert bioinformatics software engineer. Generate COMPLETE, RUNNABLE Python code
implementing the core methodology from the experimental plan below.

EXPERIMENTAL PLAN:
---
{plan}
---

Research Question: {question}

Generate MULTIPLE Python files:
1. main.py — entry point, data loading, pipeline orchestration
2. model.py — core model architecture
3. train.py — training loop with config
4. utils.py — helper functions

Output as:
===FILE:main.py===
(code)
===FILE:model.py===
(code)
===FILE:train.py===
(code)
===FILE:utils.py===
(code)

Make code self-contained, documented, and runnable."""


# =========================================================
# Single Question Pipeline
# =========================================================

def run_full_pipeline(question: str, q_idx: int, mode: str = None) -> Dict:
    if mode is None:
        mode = RETRIEVAL_MODE

    print(f"\n{'#'*70}\n  Q{q_idx+1}: {question[:120]}\n{'#'*70}")

    iter_mem = get_iteration_memory()
    prompt_mem = get_prompt_memory()

    # ═══ Stage 1: Retrieval (once, reused all rounds) ═══
    print(f"\n[Stage 1] Retrieval ({mode})...")
    s1 = run_stage1(question, mode=mode)
    context = s1["context"]
    # Dual-judge score retrieval quality
    s1_scores = dual_judge_score(
        f"Research Question: {question}\n\nRetrieved Content:\n{context[:5000]}",
        label="retrieval", verbose=True)

    print(f"  Stage1 scores: R={s1_scores['combined']['reliability']} I={s1_scores['combined']['innovation']}")

    # ═══ Rounds 1-3 ═══
    rounds_data = []
    current_prompt = GEN_INITIAL
    current_plan = ""

    for r in range(1, TOTAL_ROUNDS + 1):
        print(f"\n{'='*50}\n  ROUND {r}/{TOTAL_ROUNDS}\n{'='*50}")

        round_entry = {"round": r}

        # ── Stage 2: Plan Generation ──
        print(f"  [Stage2] Generating plan...")
        past_iter = iter_mem.get_summary()
        past_prompt = prompt_mem.get_summary()

        if r == 1:
            plan = call_llm(generator_client, GENERATOR_MODEL,
                            "Expert AI research scientist.",
                            GEN_INITIAL.format(question=question, context=context[:5000]),
                            temperature=0.3, max_tokens=8192)
        else:
            prev = rounds_data[-1]
            s2_prev = prev["stage2_scores"]
            plan = call_llm(generator_client, GENERATOR_MODEL,
                            "Expert refining plan based on judge feedback.",
                            GEN_REFINE.format(
                                question=question, context=context[:4000],
                                prev_round=r-1, prev_plan=current_plan[:6000],
                                judge1=JUDGE_MODEL_1, judge2=JUDGE_MODEL_2,
                                j1_rel=s2_prev["judge1"].get("reliability",0),
                                j1_inn=s2_prev["judge1"].get("innovation",0),
                                j2_rel=s2_prev["judge2"].get("reliability",0),
                                j2_inn=s2_prev["judge2"].get("innovation",0),
                                comb_rel=s2_prev["combined"].get("reliability",0),
                                comb_inn=s2_prev["combined"].get("innovation",0),
                                past_iteration_issues=past_iter[:2000],
                                past_prompt_issues=past_prompt[:2000],
                                avoid_msg=_AVOID_PAST,
                            ), temperature=0.35, max_tokens=8192)

        current_plan = plan
        round_entry["plan"] = plan

        # Judge the plan
        s2_scores = dual_judge_score(plan, label=f"plan_r{r}", verbose=True)
        round_entry["stage2_scores"] = s2_scores
        s2_c = s2_scores["combined"]
        print(f"  Stage2: R={s2_c['reliability']} I={s2_c['innovation']} reward={s2_scores['reward']}")

        # ── Generate detailed judge critique → write to iteration_issues.md ──
        critique_text = call_llm(generator_client, GENERATOR_MODEL,
            "Expert reviewer writing detailed critique for iterative improvement.",
            CRITIQUE_PLAN.format(
                question=question,
                judge1=JUDGE_MODEL_1, judge2=JUDGE_MODEL_2,
                j1_rel=s2_scores["judge1"].get("reliability",0),
                j1_inn=s2_scores["judge1"].get("innovation",0),
                j2_rel=s2_scores["judge2"].get("reliability",0),
                j2_inn=s2_scores["judge2"].get("innovation",0),
                comb_rel=s2_c["reliability"], comb_inn=s2_c["innovation"],
                plan_excerpt=plan[:4000],
            ), temperature=0.3, max_tokens=2048)
        iter_mem.append(question, f"[Round {r}] R={s2_c['reliability']} I={s2_c['innovation']}", critique_text[:800])
        round_entry["stage2_critique"] = critique_text
        print(f"  Stage2 critique: {len(critique_text)} chars")

        # ── Stage 3: Prompt Analysis ──
        print(f"  [Stage3] Prompt analysis...")
        past_prompt2 = prompt_mem.get_summary()
        prompt_raw = call_llm(generator_client, GENERATOR_MODEL,
                              "Expert prompt engineer.",
                              PROMPT_ANALYSIS.format(
                                  question=question, comb_rel=s2_c["reliability"],
                                  comb_inn=s2_c["innovation"],
                                  prompt_used=current_prompt[:3000],
                                  past_prompt_issues=past_prompt2[:1500],
                              ), temperature=0.25, max_tokens=4096)

        # Extract improved prompt
        m = re.search(r"Improved_Prompt\s*:\s*\n?(.+?)(?:\n\s*(?:\n|$)|$)", prompt_raw, re.DOTALL | re.IGNORECASE)
        if m:
            current_prompt = m.group(1).strip()
            round_entry["optimized_prompt"] = current_prompt
        else:
            round_entry["optimized_prompt"] = current_prompt

        # Dual-judge score the prompt
        s3_scores = dual_judge_score(
            f"Research: {question}\nPrompt:\n{current_prompt[:5000]}",
            label=f"prompt_r{r}", verbose=True)
        round_entry["stage3_scores"] = s3_scores

        # ── Generate prompt critique → write to prompt_issues.md ──
        p_critique = call_llm(generator_client, GENERATOR_MODEL,
            "Expert prompt engineer critiquing prompt quality.",
            CRITIQUE_PROMPT.format(
                question=question,
                comb_rel=s3_scores["combined"]["reliability"],
                comb_inn=s3_scores["combined"]["innovation"],
                prompt_text=current_prompt[:3000],
            ), temperature=0.3, max_tokens=1536)
        prompt_mem.append(question, f"[Round {r}] Prompt R={s3_scores['combined']['reliability']} I={s3_scores['combined']['innovation']}", p_critique[:800])
        round_entry["stage3_critique"] = p_critique

        # ── Stage 4: Code Generation ──
        print(f"  [Stage4] Code generation...")
        code_raw = call_llm(generator_client, GENERATOR_MODEL,
                            "Expert bioinformatics engineer. Output code files.",
                            CODE_GEN.format(plan=plan[:8000], question=question),
                            temperature=0.2, max_tokens=8192)

        # Parse multi-file output
        code_files = _parse_code_files(code_raw)
        code_dir = _save_code_files(q_idx, r, code_files)
        round_entry["code_dir"] = code_dir
        round_entry["code_files"] = list(code_files.keys())

        # Dual-judge score the code
        code_text_for_scoring = "\n\n".join(f"=== {fn} ===\n{fc[:2000]}" for fn, fc in list(code_files.items())[:2])
        s4_scores = dual_judge_score(
            f"Research: {question}\nCode:\n{code_text_for_scoring[:6000]}",
            label=f"code_r{r}", verbose=True)
        round_entry["stage4_scores"] = s4_scores
        print(f"  Stage4: R={s4_scores['combined']['reliability']} I={s4_scores['combined']['innovation']}")

        # ── Generate code critique → write to iteration_issues.md ──
        c_critique = call_llm(generator_client, GENERATOR_MODEL,
            "Senior code reviewer critiquing experiment code.",
            CRITIQUE_CODE.format(
                question=question,
                comb_rel=s4_scores["combined"]["reliability"],
                comb_inn=s4_scores["combined"]["innovation"],
                code_excerpt=code_text_for_scoring[:4000],
            ), temperature=0.3, max_tokens=1536)
        iter_mem.append(question, f"[Round {r}] Code R={s4_scores['combined']['reliability']} I={s4_scores['combined']['innovation']}", c_critique[:800])
        round_entry["stage4_critique"] = c_critique

        rounds_data.append(round_entry)

    # ═══ Build result ═══
    # Compute cross-round averages
    s2_rewards = [rd["stage2_scores"]["reward"] for rd in rounds_data]
    s3_rewards = [rd["stage3_scores"]["reward"] for rd in rounds_data]
    s4_rewards = [rd["stage4_scores"]["reward"] for rd in rounds_data]

    result = {
        "question_index": q_idx,
        "question": question,
        "retrieval_mode": mode,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage1": {
            "mode": s1["mode"], "doc_count": s1["doc_count"],
            "scores": s1_scores,
            "retrieved_docs": [{"source": d.get("source","?"),
                                "title": d.get("title", d.get("filename",""))[:120]}
                               for d in s1.get("retrieved_docs", [])],
        },
        "rounds": rounds_data,
        "summary": {
            "total_rounds": TOTAL_ROUNDS,
            "stage1_avg_reward": s1_scores["reward"],
            "stage2_avg_reward": _avg(s2_rewards),
            "stage3_avg_reward": _avg(s3_rewards),
            "stage4_avg_reward": _avg(s4_rewards),
            "stage2_best_reward": max(s2_rewards),
            "stage2_final": rounds_data[-1]["stage2_scores"]["combined"],
            "stage2_rewards": s2_rewards,
            "stage3_rewards": s3_rewards,
            "stage4_rewards": s4_rewards,
        },
    }
    return result


def _parse_code_files(raw: str) -> Dict[str, str]:
    """Parse ===FILE:name=== blocks from LLM output."""
    files = {}
    pattern = r'===FILE:(.+?)===\s*\n(.*?)(?=\n===FILE:|\Z)'
    for m in re.finditer(pattern, raw, re.DOTALL):
        fname = m.group(1).strip()
        code = m.group(2).strip()
        if code:
            files[fname] = code
    if not files:
        # Fallback: entire output as main.py
        clean = raw
        clean = re.sub(r'^```(?:python)?\s*\n?', '', clean, flags=re.MULTILINE)
        clean = re.sub(r'\n?```\s*$', '', clean, flags=re.MULTILINE)
        files["main.py"] = clean.strip()
    return files


def _save_code_files(q_idx: int, round_num: int, files: Dict[str, str]) -> str:
    dir_name = f"q{q_idx+1:02d}_r{round_num}"
    dir_path = os.path.join(GENERATED_CODE_DIR, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    for fname, code in files.items():
        fpath = os.path.join(dir_path, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(f"# Q{q_idx+1} Round {round_num} — {fname}\n")
            f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(code)
    print(f"    [Code] {len(files)} files → {dir_path}")
    return dir_path


# =========================================================
# Batch: All 10 Questions
# =========================================================

def run_all_questions(mode: str = None):
    if mode is None:
        mode = RETRIEVAL_MODE

    print(f"\n{'#'*70}\n  EXPERIMENT 5 — 4-STAGE × 3-ROUND PIPELINE ({mode})\n{'#'*70}")

    entries = load_bio_test_questions(BIO_TEST_PATH, n=10)
    questions = [extract_question(e) for e in entries]

    all_results = []
    acc = ScoreAccumulator()

    for i, q in enumerate(questions):
        result = run_full_pipeline(q, i, mode=mode)
        all_results.append(result)

        s = result["summary"]
        acc.add("stage1_reward", s["stage1_avg_reward"])
        acc.add("stage2_avg_reward", s["stage2_avg_reward"])
        acc.add("stage3_avg_reward", s["stage3_avg_reward"])
        acc.add("stage4_avg_reward", s["stage4_avg_reward"])
        acc.add("stage2_best_reward", s["stage2_best_reward"])
        acc.add("stage2_final_R", s["stage2_final"].get("reliability",0))
        acc.add("stage2_final_I", s["stage2_final"].get("innovation",0))

        out_path = os.path.join(_EXP5_DIR, f"q{i+1:02d}_result.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n[Saved] Q{i+1} → {out_path}")

    # Summary
    summary = acc.summary()
    print(f"\n{'#'*70}\n  FINAL SUMMARY (10 questions)\n{'#'*70}")
    for k, v in summary.items():
        print(f"  {k:<25}: avg={v['avg']:.1f}  [{v['min']:.0f}-{v['max']:.0f}]")

    final = {
        "experiment": "Experiment_5", "mode": mode,
        "generator": GENERATOR_MODEL, "judges": [JUDGE_MODEL_1, JUDGE_MODEL_2],
        "total_rounds_per_question": TOTAL_ROUNDS, "questions": 10,
        "summary": summary,
        "per_question": [{
            "q": r["question"][:120],
            "s1": r["summary"]["stage1_avg_reward"],
            "s2_best": r["summary"]["stage2_best_reward"],
            "s2_final_R": r["summary"]["stage2_final"]["reliability"],
            "s2_final_I": r["summary"]["stage2_final"]["innovation"],
            "s3_avg": r["summary"]["stage3_avg_reward"],
            "s4_avg": r["summary"]["stage4_avg_reward"],
        } for r in all_results],
    }
    sp = os.path.join(_EXP5_DIR, f"pipeline_summary_{int(time.time())}.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[Saved] Summary → {sp}")
    return all_results


if __name__ == "__main__":
    mode = RETRIEVAL_MODE
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("pubmed","pdf","hybrid"):
        mode = sys.argv[1].lower()
    run_all_questions(mode=mode)

"""
Experiment_5/stages/stage2_iteration.py — Stage 2: Plan Self-Iteration.
Generates an initial experimental plan, then iteratively self-improves it.
Issues encountered are logged to memory/iteration_issues.md for persistence.
"""

import os
import time
from typing import Dict, List

from pipeline_components import (
    call_llm, generator_client, GENERATOR_MODEL,
    dual_judge_score, get_iteration_memory, parse_score,
)
from config_exp5 import (
    TOTAL_ROUNDS,
)


PLAN_GEN_PROMPT = """You are an expert AI research scientist. Generate a complete experimental plan.

Research Question:
{question}

Retrieved Context (from Stage 1):
{context}

Past iteration issues to avoid:
{past_issues}

Generate a comprehensive experimental plan:
1. Research Objective
2. Proposed Methodology (detailed)
3. Experimental Design
4. Datasets & Evaluation Metrics
5. Expected Outcomes & Limitations

Be rigorous and innovative. Output the complete plan."""


PLAN_REFINE_PROMPT = """You are refining an experimental plan based on judge feedback.

CURRENT PLAN:
---
{plan}
---

JUDGE FEEDBACK:
Judge 1 (MiniMax-M2.5): Reliability={j1_rel}, Innovation={j1_inn}
Judge 2 (GPT-5.4): Reliability={j2_rel}, Innovation={j2_inn}
Combined: Reliability={comb_rel}, Innovation={comb_inn}

Past iteration issues to avoid:
{past_issues}

Weaknesses to address:
- If Reliability < 70: strengthen experimental rigor, add validation steps
- If Innovation < 60: incorporate novel methodology, cross-disciplinary approaches
- If both are low: fundamentally rethink the approach

Generate the IMPROVED plan. Be specific about what changed and why."""


def run_stage2(question: str, context: str, verbose: bool = True) -> Dict:
    """
    Stage 2: Plan Self-Iteration.

    1. Generate initial plan using retrieved context
    2. Dual-judge score it
    3. If below threshold, refine with feedback → re-score
    4. Log issues to iteration_issues.md
    5. Return best plan + scores + iteration history

    Returns:
        {plan, scores, iteration_history, issues_found}
    """
    print(f"\n{'='*60}")
    print(f"[Stage 2] Plan Self-Iteration (max {TOTAL_ROUNDS} rounds)")
    print(f"{'='*60}")

    memory = get_iteration_memory()
    past_issues = memory.get_summary()

    iteration_history = []
    issues_found = []
    best_plan = ""
    best_scores = {}
    best_reward = 0.0

    for round_idx in range(1, TOTAL_ROUNDS + 1):
        print(f"\n  --- Round {round_idx}/{TOTAL_ROUNDS} ---")

        if round_idx == 1:
            # Initial plan
            print(f"  [Generate] Initial plan...")
            plan = call_llm(
                generator_client, GENERATOR_MODEL,
                "You are an expert AI research scientist.",
                PLAN_GEN_PROMPT.format(
                    question=question, context=context[:5000],
                    past_issues=past_issues[:2000],
                ),
                temperature=0.3, max_tokens=8192,
            )
        else:
            # Refine based on previous scores
            prev = iteration_history[-1]
            scores = prev.get("scores", {})
            j1 = scores.get("judge1", {})
            j2 = scores.get("judge2", {})
            c = scores.get("combined", {})

            # Detect issues
            issues = []
            if c.get("reliability", 0) < 70:
                issues.append(f"Low reliability ({c.get('reliability')}): need more rigorous validation")
            if c.get("innovation", 0) < 60:
                issues.append(f"Low innovation ({c.get('innovation')}): need novel methodology")
            if len(plan) > 8000:
                issues.append(f"Plan too long ({len(plan)} chars): prune verbose sections")

            for iss in issues:
                memory.append(question, iss, f"Round {round_idx} refinement")
                issues_found.append({"round": round_idx, "issue": iss})
                print(f"  [Issue] {iss}")

            print(f"  [Refine] Based on judge feedback...")
            past_issues = memory.get_summary()
            plan = call_llm(
                generator_client, GENERATOR_MODEL,
                "You are refining an experimental plan based on reviewer feedback.",
                PLAN_REFINE_PROMPT.format(
                    plan=plan[:6000],
                    j1_rel=j1.get("reliability", 0), j1_inn=j1.get("innovation", 0),
                    j2_rel=j2.get("reliability", 0), j2_inn=j2.get("innovation", 0),
                    comb_rel=c.get("reliability", 0), comb_inn=c.get("innovation", 0),
                    past_issues=past_issues[:2000],
                ),
                temperature=0.3, max_tokens=8192,
            )

        # Score
        print(f"  [Scoring] Round {round_idx}...")
        scores = dual_judge_score(plan, label=f"plan_r{round_idx}", verbose=verbose)
        reward = scores["reward"]

        entry = {
            "round": round_idx,
            "plan": plan,
            "scores": scores,
            "reward": reward,
        }
        iteration_history.append(entry)

        print(f"  → Round {round_idx}: R={scores['combined']['reliability']} I={scores['combined']['innovation']} reward={reward}")

        # Track best
        if reward > best_reward or not best_plan:
            best_plan = plan
            best_scores = scores
            best_reward = reward

        # Stop if good enough
        c = scores["combined"]
        if c["reliability"] >= 70 and c["innovation"] >= 60:
            print(f"  → Plan exceeds threshold, stopping iteration")
            break

    print(f"\n  [Stage 2] Best: R={best_scores.get('combined',{}).get('reliability',0)} I={best_scores.get('combined',{}).get('innovation',0)} reward={best_reward}")

    return {
        "best_plan": best_plan,
        "best_scores": best_scores,
        "best_reward": best_reward,
        "iteration_history": iteration_history,
        "issues_found": issues_found,
        "rounds_completed": len(iteration_history),
    }

"""
Experiment_5/stages/stage3_prompt.py — Stage 3: Prompt Iteration.
Analyzes the generation prompt used in Stage 2 and iteratively improves it.
Issues are logged to memory/prompt_issues.md for persistence.
"""

import time
from typing import Dict, List

from pipeline_components import (
    call_llm, generator_client, GENERATOR_MODEL,
    dual_judge_score, get_prompt_memory, parse_generic_score,
)
from config_exp5 import (
    TOTAL_ROUNDS,
)


PROMPT_ANALYZE = """You are an expert prompt engineer. Analyze the following generation prompt
and identify its strengths and weaknesses for generating scientific experimental plans.

ORIGINAL PROMPT USED FOR GENERATION:
---
{prompt}
---

RESULTING PLAN SCORES:
Judge 1 (MiniMax-M2.5): R={j1_rel} I={j1_inn}
Judge 2 (GPT-5.4): R={j2_rel} I={j2_inn}
Combined: R={comb_rel} I={comb_inn}

Past prompt issues to avoid:
{past_issues}

IDENTIFY:
1. What aspects of the prompt led to good results (preserve these)
2. What weaknesses caused low scores (fix these)
3. What missing instructions would improve the plan
4. A revised prompt that addresses all issues

Output format:
Analysis: (your analysis)
Revised_Prompt: (the complete revised prompt)
Score: (estimate how much this will improve, 0-100)"""


PROMPT_SCORING = """Evaluate this prompt for generating scientific experimental plans.

Prompt:
{prompt}

Evaluate:
1. Specificity (0-100): How specific and detailed are the instructions?
2. Coverage (0-100): Does it cover all necessary aspects (method, experiment, evaluation)?

Output:
Specificity: xx
Coverage: xx"""


def run_stage3(question: str, plan: str, plan_scores: Dict, stage2_prompt: str = "",
               verbose: bool = True) -> Dict:
    """
    Stage 3: Prompt Iteration.

    1. Start with the prompt used in Stage 2
    2. Analyze it against plan scores
    3. Iteratively improve the prompt
    4. Log issues to prompt_issues.md
    5. Return the optimized prompt + scores

    Returns:
        {optimized_prompt, scores, prompt_history, issues_found}
    """
    print(f"\n{'='*60}")
    print(f"[Stage 3] Prompt Iteration (max {TOTAL_ROUNDS} rounds)")
    print(f"{'='*60}")

    memory = get_prompt_memory()
    past_issues = memory.get_summary()

    # Fallback: use a default prompt template if none provided
    if not stage2_prompt:
        stage2_prompt = (
            f"You are an expert AI research scientist. Generate a complete experimental plan "
            f"for the following research question, including objective, methodology, experimental design, "
            f"datasets, evaluation metrics, and limitations.\n\nQuestion: {question}"
        )

    current_prompt = stage2_prompt
    prompt_history = []
    issues_found = []
    best_prompt = current_prompt
    best_prompt_score = 0.0

    j1 = plan_scores.get("judge1", {})
    j2 = plan_scores.get("judge2", {})
    c = plan_scores.get("combined", {})

    for round_idx in range(1, TOTAL_ROUNDS + 1):
        print(f"\n  --- Round {round_idx}/{TOTAL_ROUNDS} ---")

        # Analyze current prompt
        print(f"  [Analyze] Prompt quality...")
        analysis = call_llm(
            generator_client, GENERATOR_MODEL,
            "You are an expert prompt engineer analyzing and improving prompts.",
            PROMPT_ANALYZE.format(
                prompt=current_prompt[:4000],
                j1_rel=j1.get("reliability", 0), j1_inn=j1.get("innovation", 0),
                j2_rel=j2.get("reliability", 0), j2_inn=j2.get("innovation", 0),
                comb_rel=c.get("reliability", 0), comb_inn=c.get("innovation", 0),
                past_issues=past_issues[:2000],
            ),
            temperature=0.3, max_tokens=4096,
        )

        # Extract revised prompt
        import re
        revised_match = re.search(r"Revised_Prompt\s*:\s*\n?(.+?)(?:\n\s*\n|\Z)", analysis, re.DOTALL | re.IGNORECASE)
        if revised_match:
            current_prompt = revised_match.group(1).strip()
            print(f"    → Revised prompt ({len(current_prompt)} chars)")
        else:
            # Check for issues and log them
            issue = f"Round {round_idx}: Could not extract revised prompt from analysis"
            memory.append(question, issue, "Manual review needed")
            issues_found.append({"round": round_idx, "issue": issue})
            print(f"    [Issue] {issue}")
            if round_idx > 1:
                break  # Don't keep failing

        # Score the prompt
        print(f"  [Scoring] Prompt quality...")
        score_text = call_llm(
            generator_client, GENERATOR_MODEL,
            "You evaluate prompt quality for scientific content generation.",
            PROMPT_SCORING.format(prompt=current_prompt[:4000]),
            temperature=0.0, max_tokens=256,
        )
        scores = parse_generic_score(score_text, "Specificity", "Coverage")
        prompt_quality = round((scores.get("specificity", 0) + scores.get("coverage", 0)) / 2, 1)

        entry = {"round": round_idx, "prompt": current_prompt, "scores": scores, "quality": prompt_quality}
        prompt_history.append(entry)
        print(f"    Specificity={scores.get('specificity',0)} Coverage={scores.get('coverage',0)} → {prompt_quality}")

        # Track best
        if prompt_quality > best_prompt_score:
            best_prompt = current_prompt
            best_prompt_score = prompt_quality

        # Log issues
        if scores.get("specificity", 0) < 50:
            iss = f"Low prompt specificity ({scores.get('specificity')})"
            memory.append(question, iss, f"Round {round_idx}: added detailed instructions")
            issues_found.append({"round": round_idx, "issue": iss})

        if prompt_quality >= 70.0:
            print(f"  → Prompt exceeds quality threshold, stopping")
            break

    # Test the best prompt by generating a sample plan
    print(f"\n  [Validate] Testing optimized prompt...")
    sample_plan = call_llm(
        generator_client, GENERATOR_MODEL,
        "You are an expert AI research scientist. Follow the prompt instructions carefully.",
        best_prompt[:5000],
        temperature=0.3, max_tokens=4096,
    )
    validation_scores = dual_judge_score(sample_plan, label="prompt_validation", verbose=verbose)

    print(f"\n  [Stage 3] Best prompt quality: {best_prompt_score}, Validation reward: {validation_scores['reward']}")

    return {
        "optimized_prompt": best_prompt,
        "prompt_quality": best_prompt_score,
        "validation_scores": validation_scores,
        "validation_reward": validation_scores["reward"],
        "prompt_history": prompt_history,
        "issues_found": issues_found,
        "rounds_completed": len(prompt_history),
    }

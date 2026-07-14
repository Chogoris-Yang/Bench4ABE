"""
Experiment_6/runner.py — RL-enhanced pipeline runner.
Runs all 10 questions, supports strategy selection and comparison.
"""

import os, sys, json, time
from typing import Dict, List

_EXP6_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXP6_DIR)

from config_exp6 import (
    GENERATOR_MODEL, RL_MODEL, JUDGE_MODEL_1, JUDGE_MODEL_2,
    RL_STRATEGY, TOTAL_ROUNDS, BIO_TEST_PATH,
)
from pipeline_components import (
    load_bio_test_questions, extract_question,
    ScoreAccumulator, _avg,
)
from rl_pipeline import run_rl_pipeline


def run_all(strategy: str = None):
    if strategy is None:
        strategy = RL_STRATEGY

    print(f"\n{'#'*70}")
    print(f"  EXPERIMENT 6 — RL-ENHANCED 4-STAGE PIPELINE")
    print(f"  Generator: {GENERATOR_MODEL}")
    print(f"  RL Optimizer: {RL_MODEL} ({strategy})")
    print(f"  Judges: {JUDGE_MODEL_1} + {JUDGE_MODEL_2}")
    print(f"  Rounds: {TOTAL_ROUNDS} | Questions: 10")
    print(f"{'#'*70}")

    entries = load_bio_test_questions(BIO_TEST_PATH, n=10)
    questions = [extract_question(e) for e in entries]

    all_results, acc = [], ScoreAccumulator()

    for i, q in enumerate(questions):
        result = run_rl_pipeline(q, i, strategy=strategy)
        all_results.append(result)

        s = result["summary"]
        acc.add("final_reward", s["final_reward"])
        acc.add("s1_reward", s["s1_reward"])
        acc.add("s2_final_R", s["s2_final"]["reliability"])
        acc.add("s2_final_I", s["s2_final"]["innovation"])
        acc.add("s3_final_R", s["s3_final"]["reliability"])
        acc.add("s4_final_R", s["s4_final"]["reliability"])

        out = os.path.join(_EXP6_DIR, f"q{i+1:02d}_result.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n[Saved] Q{i+1} → {out}")

    # Summary
    sm = acc.summary()
    print(f"\n{'#'*70}\n  FINAL — {strategy.upper()} (10 questions)\n{'#'*70}")
    for k, v in sm.items():
        print(f"  {k:<22}: avg={v['avg']:.1f} [{v['min']:.0f}-{v['max']:.0f}]")

    final = {
        "experiment": "Experiment_6", "strategy": strategy,
        "generator": GENERATOR_MODEL, "rl_model": RL_MODEL,
        "judges": [JUDGE_MODEL_1, JUDGE_MODEL_2],
        "rounds": TOTAL_ROUNDS, "questions": 10,
        "summary": sm,
        "per_question": [{
            "q": r["question"][:120],
            "final_reward": r["summary"]["final_reward"],
            "s2_final_R": r["summary"]["s2_final"]["reliability"],
            "s2_final_I": r["summary"]["s2_final"]["innovation"],
            "reward_trajectory": r["summary"]["rewards"],
        } for r in all_results],
    }
    sp = os.path.join(_EXP6_DIR, f"summary_{strategy}_{int(time.time())}.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[Saved] Summary → {sp}")
    return all_results


if __name__ == "__main__":
    strategy = RL_STRATEGY
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("ppo", "grpo", "direct"):
            strategy = arg
            print(f"  Strategy: {strategy}")
        elif arg in ("compare", "all"):
            # Run all 3 strategies and compare
            results = {}
            for s in ("ppo", "grpo", "direct"):
                results[s] = run_all(strategy=s)
            # Comparison
            print(f"\n{'#'*70}\n  STRATEGY COMPARISON\n{'#'*70}")
            print(f"  {'Strategy':<10} {'Final R':<10} {'Final I':<10} {'Avg Reward':<12}")
            for s, res in results.items():
                avg_r = _avg([r["summary"]["final_reward"] for r in res])
                avg_ri = _avg([r["summary"]["s2_final"]["reliability"] for r in res])
                avg_ii = _avg([r["summary"]["s2_final"]["innovation"] for r in res])
                print(f"  {s:<10} {avg_ri:<10.1f} {avg_ii:<10.1f} {avg_r:<12.1f}")
            sys.exit(0)
        else:
            print(f"Unknown: {arg}. Options: ppo | grpo | direct | compare")
            sys.exit(1)

    run_all(strategy=strategy)

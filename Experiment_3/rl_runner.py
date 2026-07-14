"""
Experiment_3/rl_runner.py — Unified runner for all 5 RL strategies.

Supports:
  /run <question>             — Run a single strategy on a research question
  /batch [N]                  — Batch test across N questions
  /compare [N]                — Compare ALL 5 strategies on N questions
  /strategy <name>            — Switch active strategy (ppo/grpo/gspo/dapo/odp)
  /demo                       — Demo with a built-in example
  /quit                       — Exit

Usage examples:
  python rl_runner.py
  python rl_runner.py --batch 5 --strategy grpo
  python rl_runner.py --compare 3
"""

import json
import os
import re
import time
import sys
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from rl_components import (
    load_dataset, load_jsonl, extract_user_content, extract_clean_question,
    _avg,
)
from ppo_agent import PPOAgent
from grpo_agent import GRPOAgent
from gspo_agent import GSPOAgent
from dapo_agent import DAPOAgent
from odp_agent import ODPAgent

# Default dataset — auto-detect: prefer bio_test.jsonl in Experiment_3, fallback to train.jsonl
_DEFAULT_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bio_test.jsonl")
if not os.path.exists(_DEFAULT_DATASET):
    from config import DATASET_PATH as _TRAIN_PATH
    _DEFAULT_DATASET = _TRAIN_PATH


# =========================================================
# Strategy Registry
# =========================================================

STRATEGIES = {
    "ppo":  {"name": "PPO",  "agent": PPOAgent,  "desc": "Proximal Policy Optimization — clipped surrogate objective"},
    "grpo": {"name": "GRPO", "agent": GRPOAgent, "desc": "Group Relative Policy Optimization — no Critic model, group-relative advantage"},
    "gspo": {"name": "GSPO", "agent": GSPOAgent, "desc": "Group Sampling Policy Optimization — sequential elitism + directed mutation"},
    "dapo": {"name": "DAPO", "agent": DAPOAgent, "desc": "Decoupled Alignment — asymmetric clipping + dynamic filter + token-level"},
    "odp":  {"name": "ODP",  "agent": ODPAgent,  "desc": "Online Distillation from Preferences — teacher GPT-5.4 → student DeepSeek"},
}


# =========================================================
# Single Strategy Runner
# =========================================================

def run_single_strategy(
    question: str,
    strategy_key: str = "ppo",
    verbose: bool = True,
) -> Dict:
    """Run a single RL strategy on one research question."""
    if strategy_key not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_key}. Options: {list(STRATEGIES.keys())}")

    info = STRATEGIES[strategy_key]
    agent = info["agent"]()
    result = agent.run(question, verbose=verbose)
    result["strategy"] = info["name"]
    result["strategy_key"] = strategy_key
    result["question"] = question
    return result


# =========================================================
# Batch Test Runner (Single Strategy)
# =========================================================

def batch_test_single(
    strategy_key: str = "ppo",
    n: int = 5,
    verbose: bool = True,
    dataset_path: str = None,
) -> Dict:
    """Batch test a single RL strategy on N questions from the dataset."""
    if dataset_path is None:
        dataset_path = _DEFAULT_DATASET
    info = STRATEGIES[strategy_key]
    dataset = load_jsonl(dataset_path, sample_size=n)

    print(f"\n{'#'*70}")
    print(f"  Batch Test: {info['name']} ({strategy_key})")
    print(f"  Dataset: {dataset_path}")
    print(f"  Samples: {n}  |  Generator: DeepSeek V4 Pro  |  Judges: DS-chat + GPT-5.4")
    print(f"  {info['desc']}")
    print(f"{'#'*70}")

    all_results = []
    for i, sample in enumerate(dataset):
        user_q = extract_user_content(sample)
        question = extract_clean_question(user_q)

        print(f"\n{'#'*60}")
        print(f"  Sample {i+1}/{n}: {question[:100]}...")
        print(f"{'#'*60}")

        result = run_single_strategy(question, strategy_key, verbose=verbose)
        result["sample_index"] = i

        best_scores = result.get("best_scores", {})
        print(f"  → Sample {i+1}: R={best_scores.get('reliability','?')} "
              f"I={best_scores.get('innovation','?')} reward={result.get('best_reward','?')}")

        all_results.append(result)

    # Compute aggregate metrics
    metrics = _compute_batch_summary(all_results, info["name"])

    print("\n")
    print("=" * 70)
    print(f"   {info['name']} Batch Summary ({n} samples)")
    print("=" * 70)
    print(f"   Avg Reward:      {metrics['avg_reward']}")
    print(f"   Best Reward:     {metrics['best_reward']}")
    print(f"   Reward Range:    [{metrics['min_reward']}, {metrics['max_reward']}]")
    print(f"   --- 6-SCORE Cross-Sample Averages ---")
    print(f"   DS  Avg:    R={metrics.get('cross_ds_avg_reliability', 0):.1f}  I={metrics.get('cross_ds_avg_innovation', 0):.1f}")
    print(f"   GPT Avg:    R={metrics.get('cross_gpt_avg_reliability', 0):.1f}  I={metrics.get('cross_gpt_avg_innovation', 0):.1f}")
    print(f"   Comb Avg:   R={metrics.get('cross_combined_avg_reliability', 0):.1f}  I={metrics.get('cross_combined_avg_innovation', 0):.1f}")
    print("=" * 70)

    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, f"{strategy_key}_batch_{int(time.time())}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "strategy": info["name"],
            "strategy_key": strategy_key,
            "config": vars(STRATEGIES[strategy_key]["agent"]()) if hasattr(STRATEGIES[strategy_key]["agent"](), '__dict__') else {},
            "metrics": metrics,
            "samples": all_results,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[Results saved] {output_path}")

    return {"metrics": metrics, "results": all_results}


# =========================================================
# Compare All Strategies
# =========================================================

def compare_strategies(n: int = 3, dataset_path: str = None) -> Dict:
    """
    Run ALL 5 RL strategies on the same N questions and compare results.
    This is the most valuable mode — it produces a comparative analysis.

    Args:
        n: Number of questions to sample from the dataset
        dataset_path: Path to JSONL file (default: auto-detect bio_test.jsonl or train.jsonl)
    """
    if dataset_path is None:
        dataset_path = _DEFAULT_DATASET
    dataset = load_jsonl(dataset_path, sample_size=n)

    print(f"\n{'#'*70}")
    print(f"  COMPARATIVE STUDY: 5 RL Strategies on {n} Research Questions")
    print(f"{'#'*70}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Strategies: PPO | GRPO | GSPO | DAPO | ODP")
    print(f"  Generator: DeepSeek V4 Pro  |  Judges: DS-chat + GPT-5.4")
    print(f"{'#'*70}")

    # Results organized by strategy → samples
    all_strategy_results: Dict[str, List[Dict]] = {key: [] for key in STRATEGIES}

    for i, sample in enumerate(dataset):
        user_q = extract_user_content(sample)
        question = extract_clean_question(user_q)

        print(f"\n{'#'*60}")
        print(f"  Question {i+1}/{n}: {question[:120]}...")
        print(f"{'#'*60}")

        # Run each strategy on this question (sequential to avoid API rate limits)
        for key, info in STRATEGIES.items():
            print(f"\n  --- {info['name']} ({key}) ---")
            try:
                result = run_single_strategy(question, key, verbose=False)
                result["sample_index"] = i
                all_strategy_results[key].append(result)

                best_scores = result.get("best_scores", {})
                print(f"  {info['name']}: R={best_scores.get('reliability','?')} "
                      f"I={best_scores.get('innovation','?')} "
                      f"reward={result.get('best_reward','?')}")
            except Exception as e:
                print(f"  [!] {info['name']} ERROR: {e}")
                all_strategy_results[key].append({
                    "strategy": info["name"],
                    "strategy_key": key,
                    "sample_index": i,
                    "question": question,
                    "error": str(e),
                    "best_reward": 0,
                    "best_scores": {},
                })

    # Compute per-strategy metrics
    comparison = {}
    for key, results in all_strategy_results.items():
        valid_results = [r for r in results if "error" not in r]
        comparison[key] = _compute_batch_summary(valid_results, STRATEGIES[key]["name"])

    # Rank strategies
    rankings = sorted(comparison.items(), key=lambda x: x[1].get("avg_reward", 0), reverse=True)

    # Print comparison report
    print("\n\n")
    print("=" * 80)
    print("  5-STRATEGY COMPARATIVE ANALYSIS")
    print("=" * 80)
    print()
    print(f"  {'Rank':<6} {'Strategy':<8} {'Avg Reward':<12} {'DS(R/I)':<18} {'GPT(R/I)':<18} {'Comb(R/I)':<18}")
    print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*18} {'-'*18} {'-'*18}")

    for rank, (key, metrics) in enumerate(rankings, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f" {rank}."
        ds_r = metrics.get("cross_ds_avg_reliability", metrics.get("avg_reliability", 0))
        ds_i = metrics.get("cross_ds_avg_innovation", metrics.get("avg_innovation", 0))
        gpt_r = metrics.get("cross_gpt_avg_reliability", 0)
        gpt_i = metrics.get("cross_gpt_avg_innovation", 0)
        comb_r = metrics.get("cross_combined_avg_reliability", metrics.get("avg_reliability", 0))
        comb_i = metrics.get("cross_combined_avg_innovation", metrics.get("avg_innovation", 0))
        print(f"  {medal:<6} {STRATEGIES[key]['name']:<8} "
              f"{metrics['avg_reward']:<12.1f} "
              f"R={ds_r:.1f} I={ds_i:.1f}     "
              f"R={gpt_r:.1f} I={gpt_i:.1f}     "
              f"R={comb_r:.1f} I={comb_i:.1f}")

    # 6-SCORE DETAILED BREAKDOWN
    print(f"\n  {'='*70}")
    print(f"  6-SCORE DETAILED BREAKDOWN (per-judge averages across samples)")
    print(f"  {'='*70}")
    print(f"  {'Strategy':<8} {'DS Avg Rel':<13} {'DS Avg Inn':<13} {'GPT Avg Rel':<13} {'GPT Avg Inn':<13} {'Comb Avg Rel':<13} {'Comb Avg Inn':<13}")
    print(f"  {'-'*8} {'-'*13} {'-'*13} {'-'*13} {'-'*13} {'-'*13} {'-'*13}")

    for key, metrics in comparison.items():
        print(f"  {STRATEGIES[key]['name']:<8} "
              f"{metrics.get('cross_ds_avg_reliability', 0):<13.1f} "
              f"{metrics.get('cross_ds_avg_innovation', 0):<13.1f} "
              f"{metrics.get('cross_gpt_avg_reliability', 0):<13.1f} "
              f"{metrics.get('cross_gpt_avg_innovation', 0):<13.1f} "
              f"{metrics.get('cross_combined_avg_reliability', 0):<13.1f} "
              f"{metrics.get('cross_combined_avg_innovation', 0):<13.1f}")

    print(f"\n  {'='*70}")
    print(f"  Analysis Summary")
    print(f"  {'='*70}")

    if len(rankings) >= 2:
        best_key, best_metrics = rankings[0]
        worst_key, worst_metrics = rankings[-1]
        gap = best_metrics["avg_reward"] - worst_metrics["avg_reward"]
        print(f"\n  Winner: {STRATEGIES[best_key]['name']} ({best_metrics['avg_reward']:.1f})")
        print(f"  Gap between best and worst: {gap:.1f} points")
        print(f"  Winner's edge: {STRATEGIES[best_key]['desc']}")

    # Per-question strategy breakdown
    print(f"\n  {'='*70}")
    print(f"  Per-Question Strategy Ranking")
    print(f"  {'='*70}")

    for i in range(n):
        q_scores = {}
        for key, results in all_strategy_results.items():
            if i < len(results) and "error" not in results[i]:
                q_scores[key] = results[i].get("best_reward", 0)

        q_ranked = sorted(q_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Q{i+1}: {dataset[i].get('messages', [{}])[0].get('content', '?')[:80]}")
        for rk, (key, score) in enumerate(q_ranked, 1):
            print(f"    {rk}. {STRATEGIES[key]['name']}: {score:.1f}")

    print(f"\n{'='*80}")

    # Save comparative results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, f"compare_all_{int(time.time())}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "sample_count": n,
                "strategies": list(STRATEGIES.keys()),
                "generator": "DeepSeek V4 Pro",
                "judges": ["DeepSeek-chat", "GPT-5.4"],
            },
            "rankings": [
                {"rank": i+1, "strategy": STRATEGIES[k]["name"], "metrics": m}
                for i, (k, m) in enumerate(rankings)
            ],
            "per_strategy": comparison,
            "per_question": [
                {
                    "question": extract_clean_question(extract_user_content(dataset[j])),
                    "strategies": {
                        k: all_strategy_results[k][j].get("best_reward", 0)
                        for k in STRATEGIES if j < len(all_strategy_results[k])
                    }
                }
                for j in range(n)
            ],
            "all_results": all_strategy_results,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[Comparative results saved] {output_path}")

    return {"rankings": rankings, "comparison": comparison, "per_strategy": all_strategy_results}


# =========================================================
# Helper: Compute Batch Summary Metrics
# =========================================================

def _compute_batch_summary(results: List[Dict], strategy_name: str) -> Dict:
    """Compute aggregate metrics from batch results."""
    rewards = []
    reliabilities = []
    innovations = []
    iterations = []

    for r in results:
        rewards.append(r.get("best_reward", 0))
        bs = r.get("best_scores", {})
        reliabilities.append(bs.get("reliability", 0) if isinstance(bs, dict) else 0)
        innovations.append(bs.get("innovation", 0) if isinstance(bs, dict) else 0)

        # Count iterations from various possible keys
        iters = (
            r.get("epochs_completed", 0) or
            r.get("generations_completed", 0) or
            r.get("rounds_completed", 0) or
            r.get("steps_completed", 0) or
            0
        )
        iterations.append(iters)

    return {
        "strategy": strategy_name,
        "sample_count": len(rewards),
        "avg_reward": _avg(rewards),
        "avg_reliability": _avg(reliabilities),
        "avg_innovation": _avg(innovations),
        "best_reward": max(rewards) if rewards else 0,
        "min_reward": min(rewards) if rewards else 0,
        "max_reward": max(rewards) if rewards else 0,
        "avg_iterations": _avg(iterations),
        "all_rewards": rewards,
    }


# =========================================================
# Interactive CLI
# =========================================================

BANNER = r"""
╔══════════════════════════════════════════════════════════════════════╗
║     RL-Based Experiment Protocol Generator                          ║
║     5 Strategies: PPO | GRPO | GSPO | DAPO | ODP                    ║
║     Generator: DeepSeek V4 Pro  |  Judges: DS-chat + GPT-5.4       ║
╚══════════════════════════════════════════════════════════════════════╝

Commands:
  /run <question>         — Run current strategy on a research question
  /batch [N]              — Batch test current strategy on N questions (default 3)
  /compare [N]            — Compare ALL 5 strategies on N questions (default 3)
  /strategy <name>        — Switch strategy (ppo/grpo/gspo/dapo/odp)
  /strategies             — List all available strategies
  /dataset [path]         — Set or show dataset path (default: bio_test.jsonl)
  /demo                   — Run demo with built-in example
  /quit                   — Exit

Current strategy: {current_strategy}
Dataset: {current_dataset}
"""


def interactive_loop():
    """Interactive CLI for RL-based experiment generation."""
    current_strategy = "ppo"
    current_dataset = _DEFAULT_DATASET

    print(BANNER.format(
        current_strategy=current_strategy.upper(),
        current_dataset=current_dataset,
    ))

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

        # /strategy <name>
        if user_input.lower().startswith("/strategy "):
            parts = user_input.split()
            if len(parts) >= 2:
                key = parts[1].lower()
                if key in STRATEGIES:
                    current_strategy = key
                    print(f"  Switched to {STRATEGIES[key]['name']}: {STRATEGIES[key]['desc']}")
                else:
                    print(f"  Unknown strategy: {key}. Options: {list(STRATEGIES.keys())}")
            continue

        # /strategies
        if user_input.lower() == "/strategies":
            print("\n  Available RL Strategies:")
            print(f"  {'Key':<8} {'Name':<8} {'Description'}")
            print(f"  {'-'*8} {'-'*8} {'-'*50}")
            for key, info in STRATEGIES.items():
                print(f"  {key:<8} {info['name']:<8} {info['desc']}")
            continue

        # /dataset [path]
        if user_input.lower().startswith("/dataset"):
            parts = user_input.split(maxsplit=1)
            if len(parts) >= 2:
                new_path = parts[1].strip()
                if os.path.exists(new_path):
                    current_dataset = new_path
                    print(f"  Dataset set to: {current_dataset}")
                else:
                    print(f"  [!] File not found: {new_path}")
            else:
                # Show current dataset info
                print(f"  Current dataset: {current_dataset}")
                if os.path.exists(current_dataset):
                    n_total = sum(1 for _ in open(current_dataset, "r", encoding="utf-8"))
                    print(f"  Total entries:   {n_total}")
                # Also list JSONL files in Experiment_3
                exp_dir = os.path.dirname(os.path.abspath(__file__))
                jsonl_files = [f for f in os.listdir(exp_dir) if f.endswith('.jsonl')]
                if jsonl_files:
                    print(f"  Available .jsonl files: {', '.join(jsonl_files)}")
            continue

        # /demo
        if user_input.lower() == "/demo":
            user_input = (
                "/run Design an experiment to evaluate whether large language models "
                "can self-improve through recursive self-critique without external feedback"
            )

        # /run <question>
        if user_input.lower().startswith("/run "):
            question = user_input[5:].strip()
            print(f"\n  Strategy: {STRATEGIES[current_strategy]['name']}")
            print(f"  Question: {question[:200]}")

            result = run_single_strategy(question, current_strategy)

            print("\n" + result.get("final_report", "[No report generated]"))

            # Save
            output_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(output_dir, f"{current_strategy}_result_{int(time.time())}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n[Result saved] {output_path}")

        # /batch [N]
        elif user_input.lower().startswith("/batch"):
            parts = user_input.split()
            n = int(parts[1]) if len(parts) > 1 else 3
            batch_test_single(current_strategy, n=n, dataset_path=current_dataset)

        # /compare [N]
        elif user_input.lower().startswith("/compare"):
            parts = user_input.split()
            n = int(parts[1]) if len(parts) > 1 else 3
            compare_strategies(n=n, dataset_path=current_dataset)

        else:
            print("  Unknown command. Use /run, /batch, /compare, /strategy, /strategies, /dataset, /demo, or /quit.")


# =========================================================
# Direct Python API
# =========================================================

def generate_experiment(
    task_descriptions: List[str],
    strategy: str = "ppo",
    verbose: bool = True,
) -> List[Dict]:
    """
    Generate experiment protocols for a list of task descriptions.

    Args:
        task_descriptions: List of research question strings
        strategy: One of 'ppo', 'grpo', 'gspo', 'dapo', 'odp'
        verbose: Whether to print progress

    Returns:
        List of result dicts, one per task description
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Options: {list(STRATEGIES.keys())}")

    results = []
    for i, task in enumerate(task_descriptions):
        print(f"\n[{i+1}/{len(task_descriptions)}] Processing: {task[:100]}...")
        result = run_single_strategy(task, strategy, verbose=verbose)
        result["task_index"] = i
        result["task_description"] = task
        results.append(result)

    return results


def compare_strategies_on_tasks(
    task_descriptions: List[str],
    verbose: bool = True,
) -> Dict:
    """
    Compare all 5 RL strategies on a given set of task descriptions.

    Returns a dict with per-strategy results and rankings.
    """
    all_strategy_results: Dict[str, List[Dict]] = {key: [] for key in STRATEGIES}

    for i, task in enumerate(task_descriptions):
        print(f"\n[{i+1}/{len(task_descriptions)}] Processing: {task[:100]}...")

        for key, info in STRATEGIES.items():
            try:
                result = run_single_strategy(task, key, verbose=verbose)
                result["task_index"] = i
                all_strategy_results[key].append(result)
            except Exception as e:
                print(f"  [!] {info['name']} ERROR on task {i+1}: {e}")
                all_strategy_results[key].append({
                    "strategy": info["name"],
                    "task_index": i,
                    "task_description": task,
                    "error": str(e),
                })

    comparison = {}
    for key, results in all_strategy_results.items():
        valid = [r for r in results if "error" not in r]
        comparison[key] = _compute_batch_summary(valid, STRATEGIES[key]["name"])

    rankings = sorted(comparison.items(), key=lambda x: x[1].get("avg_reward", 0), reverse=True)

    return {
        "rankings": [
            {"rank": i+1, "strategy": STRATEGIES[k]["name"], "metrics": m}
            for i, (k, m) in enumerate(rankings)
        ],
        "per_strategy": comparison,
        "per_task": [
            {
                "task": task_descriptions[j],
                "strategies": {
                    k: all_strategy_results[k][j].get("best_reward", 0)
                    for k in STRATEGIES if j < len(all_strategy_results[k])
                }
            }
            for j in range(len(task_descriptions))
        ],
        "all_results": all_strategy_results,
    }


# =========================================================
# Main Entry Point
# =========================================================

def _parse_args(argv: List[str]) -> Dict[str, Any]:
    """Parse CLI arguments."""
    opts = {
        "dataset": _DEFAULT_DATASET,
        "strategy": "ppo",
        "n": 3,
    }
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--dataset" and i + 1 < len(argv):
            opts["dataset"] = argv[i + 1]; i += 2
        elif arg == "--strategy" and i + 1 < len(argv):
            opts["strategy"] = argv[i + 1].lower(); i += 2
        elif arg in ("--compare", "--batch") and i + 1 < len(argv) and argv[i + 1].isdigit():
            opts["n"] = int(argv[i + 1]); i += 2
        elif arg == "--run" and i + 1 < len(argv):
            opts["question"] = argv[i + 1]; i += 2
        elif arg == "--compare":
            opts["mode"] = "compare"; i += 1
        elif arg == "--batch":
            opts["mode"] = "batch"; i += 1
        elif arg == "--run":
            opts["mode"] = "run"
            if "question" not in opts:
                opts["question"] = "Design an experiment for drug-target interaction prediction"
            i += 1
        else:
            i += 1
    return opts


if __name__ == "__main__":
    if len(sys.argv) > 1:
        opts = _parse_args(sys.argv[1:])
        mode = opts.get("mode", "interactive")
        dataset_path = opts.get("dataset", _DEFAULT_DATASET)
        strategy = opts.get("strategy", "ppo")
        n = opts.get("n", 3)

        if mode == "compare":
            compare_strategies(n=n, dataset_path=dataset_path)

        elif mode == "batch":
            batch_test_single(strategy, n=n, dataset_path=dataset_path)

        elif mode == "run":
            question = opts.get("question", "Design an experiment for drug-target interaction prediction")
            result = run_single_strategy(question, strategy)
            print("\n" + result.get("final_report", ""))

        else:
            print("Usage: python rl_runner.py [--run <question>] [--batch N] [--compare N] [--strategy <name>] [--dataset <path>]")
            print(f"Available strategies: {list(STRATEGIES.keys())}")
            print(f"Default dataset: {_DEFAULT_DATASET}")
    else:
        interactive_loop()

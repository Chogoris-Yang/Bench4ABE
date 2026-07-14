"""
Experiment_4/rag_runner.py — RAG-enhanced RL experiment protocol generation.
Runs sub-experiments 4.1, 4.2, 4.3 and generates comparative analysis.

Sub-experiments:
  4.1 — Retrieval Strategy Ablation (Dense/Sparse/Hybrid × Papers/Prior/All)
  4.2 — RAG × RL Cross Experiment (5 strategies × with/without RAG)
  4.3 — RAG Integration Depth (pre_gen/post_gen/iterative/full_pipeline)

Usage:
  python rag_runner.py                          # Interactive menu
  python rag_runner.py --exp 4.1                # Retrieval ablation
  python rag_runner.py --exp 4.2                # RAG×RL cross experiment
  python rag_runner.py --exp 4.3                # Integration depth
  python rag_runner.py --exp all                # Run all three
"""

import os
import sys
import json
import time
import re
import importlib
from typing import Dict, List, Any

# Import Experiment_3 for baseline (no-RAG) results
# Case-insensitive lookup: try both "experiment_3" and "Experiment_3"
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXP3_DIR = None
for _candidate in ("experiment_3", "Experiment_3", "EXPERIMENT_3"):
    _path = os.path.join(_PARENT_DIR, _candidate)
    if os.path.isdir(_path):
        _EXP3_DIR = _path
        break
if _EXP3_DIR is None:
    raise FileNotFoundError(
        f"Cannot find Experiment_3 directory under {_PARENT_DIR}. "
        f"Tried: experiment_3, Experiment_3"
    )
if _EXP3_DIR not in sys.path:
    sys.path.insert(0, _EXP3_DIR)
print(f"  [Import] Experiment_3 path: {_EXP3_DIR}")

from rl_components import load_jsonl, extract_clean_question, extract_user_content, _avg

from rag_components import (
    RAGWrapper, KnowledgeSource,
    create_rag_agent, build_retrieved_context, build_prior_protocols_from_jsonl,
)
from retrieval.retriever import get_retriever, DocumentStore
from config_rag import (
    EXP41_STRATEGY, EXP41_N_QUESTIONS, EXP41_RETRIEVAL_METHODS, EXP41_KNOWLEDGE_SOURCES,
    EXP42_N_QUESTIONS, EXP42_STRATEGIES, EXP42_MODES,
    EXP43_STRATEGY, EXP43_N_QUESTIONS, EXP43_MODES,
    BIO_TEST_PATH,
)

# =========================================================
# Runner Utilities
# =========================================================

def _load_questions(n: int, path: str = BIO_TEST_PATH) -> List[str]:
    """Load N cleaned questions from a JSONL dataset."""
    entries = load_jsonl(path, sample_size=n)
    questions = []
    for entry in entries:
        user_q = extract_user_content(entry)
        clean = extract_clean_question(user_q)
        questions.append(clean)
    return questions


def _compute_6score_delta(rag_result: Dict, baseline_result: Dict) -> Dict:
    """Compute the 6-score difference between RAG and baseline."""
    rag_six = rag_result.get("six_scores", {})
    base_six = baseline_result.get("six_scores", {})

    delta = {}
    for key in ["ds_avg_reliability", "ds_avg_innovation",
                "gpt_avg_reliability", "gpt_avg_innovation",
                "combined_avg_reliability", "combined_avg_innovation"]:
        delta[key] = round(rag_six.get(key, 0) - base_six.get(key, 0), 1)

    delta["reward"] = round(rag_result.get("best_reward", 0) - baseline_result.get("best_reward", 0), 1)
    return delta


def _print_header(title: str, width: int = 75):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


# =========================================================
# Exp4.1: Retrieval Strategy Ablation
# =========================================================

def run_exp41_retrieval_ablation():
    """
    Fixed: GRPO strategy, 3 questions
    Vary: retrieval method (dense/sparse/hybrid) × knowledge source (papers/prior/all)
    Reports: 6-score for each combination → best configuration.
    """
    questions = _load_questions(EXP41_N_QUESTIONS)
    prior_protocols = build_prior_protocols_from_jsonl(BIO_TEST_PATH, n=EXP41_N_QUESTIONS)

    _print_header("Experiment 4.1: Retrieval Strategy Ablation")
    print(f"  Fixed: {EXP41_STRATEGY.upper()} on {EXP41_N_QUESTIONS} questions")
    print(f"  Grid: {len(EXP41_RETRIEVAL_METHODS)} methods × {len(EXP41_KNOWLEDGE_SOURCES)} sources = {len(EXP41_RETRIEVAL_METHODS)*len(EXP41_KNOWLEDGE_SOURCES)} combinations")

    results = {}
    details = {}  # Per-question full results (including plans)
    for source_name in EXP41_KNOWLEDGE_SOURCES:
        for method in EXP41_RETRIEVAL_METHODS:
            combo_key = f"{method}×{source_name}"
            _print_header(f"4.1 [{combo_key}]", 65)
            print(f"  Method: {method}  |  Source: {source_name}")

            combo_results = []
            for i, q in enumerate(questions):
                print(f"\n  Q{i+1}/{len(questions)}: {q[:100]}...")

                try:
                    agent = create_rag_agent(
                        EXP41_STRATEGY,
                        retriever=method,
                        mode="pre_gen",
                        prior_protocols=prior_protocols if source_name in ("prior", "all") else None,
                    )

                    # Override knowledge source based on config
                    if source_name == "papers":
                        agent.knowledge_source.build_store_from_sources = lambda q, src=None, pp=None: (
                            KnowledgeSource().pubmed.search_for_task(q) or
                            DocumentStore()
                        )
                        # Actually, let's just handle this by building the right store
                        store = DocumentStore()
                        papers = KnowledgeSource().pubmed.search_for_task(q)
                        if papers:
                            store.add_pubmed_papers(papers)
                        agent.last_retrieved_docs = agent.retriever.retrieve(q, store, top_k=5)
                        agent.last_retrieved_context = build_retrieved_context(agent.last_retrieved_docs)
                        result = agent._run_pre_gen(q)

                    elif source_name == "prior":
                        store = DocumentStore()
                        if prior_protocols:
                            store.add_prior_protocols(prior_protocols)
                        agent.last_retrieved_docs = agent.retriever.retrieve(q, store, top_k=5)
                        agent.last_retrieved_context = build_retrieved_context(agent.last_retrieved_docs)
                        result = agent._run_pre_gen(q)

                    else:  # "all"
                        result = agent.run(q)

                    combo_results.append(result)
                    print(f"    → reward={result.get('best_reward', 0):.1f}")

                except Exception as e:
                    print(f"    [!] Error: {e}")
                    combo_results.append({"best_reward": 0, "best_scores": {}, "six_scores": {}, "error": str(e)})

            # Aggregate
            rewards = [r.get("best_reward", 0) for r in combo_results]
            valid = [r for r in combo_results if "error" not in r]
            six_agg = {}
            if valid:
                for k in ["combined_avg_reliability", "combined_avg_innovation"]:
                    vals = [r.get("six_scores", {}).get(k, 0) for r in valid]
                    six_agg[k] = _avg(vals)

            results[combo_key] = {
                "avg_reward": _avg(rewards),
                "max_reward": max(rewards) if rewards else 0,
                "six_scores_avg": six_agg,
                "valid_count": len(valid),
            }
            details[combo_key] = combo_results  # Save full per-question results with plans

    # Print 4.1 summary
    _print_header("4.1 Results: Retrieval Strategy Ranking")
    ranked = sorted(results.items(), key=lambda x: x[1]["avg_reward"], reverse=True)
    print(f"\n  {'Rank':<5} {'Config':<22} {'Avg Reward':<12} {'Max':<8} {'Comb R/I':<15} {'Valid':<7}")
    print(f"  {'-'*5} {'-'*22} {'-'*12} {'-'*8} {'-'*15} {'-'*7}")
    for rank, (key, metrics) in enumerate(ranked, 1):
        six = metrics["six_scores_avg"]
        print(f"  {rank:<5} {key:<22} {metrics['avg_reward']:<12.1f} {metrics['max_reward']:<8.1f} "
              f"R={six.get('combined_avg_reliability',0):.1f} I={six.get('combined_avg_innovation',0):.1f}   "
              f"{metrics['valid_count']}/{EXP41_N_QUESTIONS}")

    best_config = ranked[0][0]
    print(f"\n  Best configuration: {best_config}")
    print(f"  Recommended for Experiment 4.2 + 4.3")

    # Save
    _save_results("exp41_retrieval_ablation", {
        "grid_results": results,
        "ranking": ranked,
        "best": best_config,
        "details": details,  # Per-question plans + scores
    })
    return {"best_config": best_config, "results": results}


# =========================================================
# Exp4.2: RAG × RL Cross Experiment
# =========================================================

def run_exp42_rag_rl_cross(best_config: str = None):
    """
    5 strategies × 2 modes (with/without RAG) on 10 questions.
    Reuses Experiment_3 "without RAG" results if available.
    """
    questions = _load_questions(EXP42_N_QUESTIONS)
    prior_protocols = build_prior_protocols_from_jsonl(BIO_TEST_PATH, n=EXP42_N_QUESTIONS)

    # Parse best config from 4.1 (e.g., "hybrid×all")
    if best_config:
        parts = best_config.split("×")
        best_method, best_source = parts[0], parts[1] if len(parts) > 1 else "all"
    else:
        best_method, best_source = "hybrid", "all"

    _print_header("Experiment 4.2: RAG × RL Cross Experiment")
    print(f"  Strategies: {EXP42_STRATEGIES}")
    print(f"  Retrieval: {best_method} × {best_source}")
    print(f"  Questions: {EXP42_N_QUESTIONS}")
    print(f"  Condition: RAG vs No-RAG for all 5 strategies")

    all_results = {}  # strategy → {rag: [...], no_rag: [...]}

    for strategy_key in EXP42_STRATEGIES:
        _print_header(f"4.2 [{strategy_key.upper()}] RAG vs No-RAG")

        rag_results = []
        no_rag_results = []

        for i, q in enumerate(questions):
            print(f"\n  Q{i+1}/{len(questions)}: {q[:100]}...")

            # ── RAG condition ──
            print(f"    [{strategy_key}] RAG ({best_method}×{best_source})...")
            try:
                agent = create_rag_agent(
                    strategy_key, retriever=best_method, mode="pre_gen",
                    prior_protocols=prior_protocols if best_source in ("prior", "all") else None,
                )

                if best_source == "papers":
                    store = DocumentStore()
                    papers = KnowledgeSource().pubmed.search_for_task(q)
                    if papers:
                        store.add_pubmed_papers(papers)
                    agent.last_retrieved_docs = agent.retriever.retrieve(q, store)
                    agent.last_retrieved_context = build_retrieved_context(agent.last_retrieved_docs)
                    rag_result = agent._run_pre_gen(q)
                elif best_source == "prior":
                    store = DocumentStore()
                    if prior_protocols:
                        store.add_prior_protocols(prior_protocols)
                    agent.last_retrieved_docs = agent.retriever.retrieve(q, store)
                    agent.last_retrieved_context = build_retrieved_context(agent.last_retrieved_docs)
                    rag_result = agent._run_pre_gen(q)
                else:
                    rag_result = agent.run(q)

                rag_results.append(rag_result)
                rag_reward = rag_result.get("best_reward", 0)
            except Exception as e:
                print(f"    [!] RAG error: {e}")
                rag_results.append({"best_reward": 0, "six_scores": {}, "error": str(e)})
                rag_reward = 0

            # ── No-RAG condition (fresh base agent, no RAG at all) ──
            print(f"    [{strategy_key}] No-RAG...")
            try:
                mod = importlib.import_module(f"{strategy_key}_agent")
                cls = getattr(mod, f"{strategy_key.upper()}Agent")
                base_agent = cls()
                base_result = base_agent.run(q, verbose=False)

                no_rag_results.append(base_result)
                base_reward = base_result.get("best_reward", 0)
            except Exception as e:
                print(f"    [!] No-RAG error: {e}")
                no_rag_results.append({"best_reward": 0, "six_scores": {}, "error": str(e)})
                base_reward = 0

            delta = rag_reward - base_reward
            print(f"    Δ = RAG({rag_reward:.1f}) - NoRAG({base_reward:.1f}) = {delta:+.1f}")

        all_results[strategy_key] = {"rag": rag_results, "no_rag": no_rag_results}

    # Summary table
    _print_header("4.2 Results: RAG × RL Cross Comparison")
    print(f"\n  {'Strategy':<8} {'No-RAG R/I':<16} {'RAG R/I':<16} {'Δ Reward':<12} {'Δ Comb Rel':<13} {'Δ Comb Inn':<13} {'Winner':<8}")
    print(f"  {'-'*8} {'-'*16} {'-'*16} {'-'*12} {'-'*13} {'-'*13} {'-'*8}")

    summary = {}
    for sk in EXP42_STRATEGIES:
        res = all_results[sk]
        rag_valid = [r for r in res["rag"] if "error" not in r]
        no_rag_valid = [r for r in res["no_rag"] if "error" not in r]

        rag_avg_r = _avg([r.get("best_reward", 0) for r in rag_valid])
        no_rag_avg_r = _avg([r.get("best_reward", 0) for r in no_rag_valid])
        delta_reward = round(rag_avg_r - no_rag_avg_r, 1)

        rag_six = {}
        no_rag_six = {}
        if rag_valid:
            for k in ["combined_avg_reliability", "combined_avg_innovation"]:
                rag_six[k] = _avg([r.get("six_scores", {}).get(k, 0) for r in rag_valid])
        if no_rag_valid:
            for k in ["combined_avg_reliability", "combined_avg_innovation"]:
                no_rag_six[k] = _avg([r.get("six_scores", {}).get(k, 0) for r in no_rag_valid])

        delta_rel = round(rag_six.get("combined_avg_reliability", 0) - no_rag_six.get("combined_avg_reliability", 0), 1)
        delta_inn = round(rag_six.get("combined_avg_innovation", 0) - no_rag_six.get("combined_avg_innovation", 0), 1)
        winner = "RAG" if delta_reward > 0 else "No-RAG" if delta_reward < 0 else "TIE"

        print(f"  {sk.upper():<8} R={no_rag_six.get('combined_avg_reliability',0):.1f} I={no_rag_six.get('combined_avg_innovation',0):.1f}    "
              f"R={rag_six.get('combined_avg_reliability',0):.1f} I={rag_six.get('combined_avg_innovation',0):.1f}    "
              f"{delta_reward:+.1f}        {delta_rel:+.1f}           {delta_inn:+.1f}           {winner}")

        summary[sk] = {
            "no_rag_avg_reward": no_rag_avg_r,
            "rag_avg_reward": rag_avg_r,
            "delta_reward": delta_reward,
            "delta_rel": delta_rel,
            "delta_inn": delta_inn,
            "winner": winner,
        }

    # Find best RAG strategy
    best_rag = max(summary.items(), key=lambda x: x[1]["rag_avg_reward"])
    print(f"\n  Best RAG strategy: {best_rag[0].upper()} (reward={best_rag[1]['rag_avg_reward']:.1f})")
    print(f"  Largest RAG gain: {max(summary.items(), key=lambda x: x[1]['delta_reward'])[0].upper()} (Δ={max(s['delta_reward'] for s in summary.values()):+.1f})")

    _save_results("exp42_rag_rl_cross", {"all_results": all_results, "summary": summary, "best_rag": best_rag[0]})
    return {"summary": summary, "best_rag": best_rag[0]}


# =========================================================
# Exp4.3: RAG Integration Depth
# =========================================================

def run_exp43_integration_depth(best_strategy: str = None, best_config: str = None):
    """
    Fixed: Best strategy from 4.2, best retrieval config from 4.1, 5 questions
    Vary: 4 RAG injection modes (pre_gen, post_gen, iterative, full_pipeline)
    Plus baseline: No-RAG
    """
    if best_strategy is None:
        best_strategy = EXP43_STRATEGY
    if best_config is None:
        best_config = "hybrid×all"

    parts = best_config.split("×")
    best_method, best_source = parts[0], parts[1] if len(parts) > 1 else "all"

    questions = _load_questions(EXP43_N_QUESTIONS)
    prior_protocols = build_prior_protocols_from_jsonl(BIO_TEST_PATH, n=EXP43_N_QUESTIONS)

    _print_header("Experiment 4.3: RAG Integration Depth")
    print(f"  Strategy: {best_strategy.upper()}")
    print(f"  Retrieval: {best_method} × {best_source}")
    print(f"  Questions: {EXP43_N_QUESTIONS}")
    print(f"  Modes: {EXP43_MODES} + baseline (no_rag)")

    modes = EXP43_MODES + ["no_rag"]
    mode_results = {m: [] for m in modes}

    for i, q in enumerate(questions):
        print(f"\n  Q{i+1}/{len(questions)}: {q[:100]}...")

        # Build shared store
        store = DocumentStore()
        if best_source in ("papers", "all"):
            papers = KnowledgeSource().pubmed.search_for_task(q)
            if papers:
                store.add_pubmed_papers(papers)
        if best_source in ("prior", "all"):
            if prior_protocols:
                store.add_prior_protocols(prior_protocols)

        retriever = get_retriever(best_method)

        for mode in modes:
            if mode == "no_rag":
                print(f"    [{mode}] baseline...")
                try:
                    import importlib
                    mod = importlib.import_module(f"{best_strategy}_agent")
                    cls = getattr(mod, f"{best_strategy.upper()}Agent")
                    base_agent = cls()
                    result = base_agent.run(q, verbose=False)
                    mode_results[mode].append(result)
                    print(f"      → reward={result.get('best_reward', 0):.1f}")
                except Exception as e:
                    print(f"      [!] Error: {e}")
                    mode_results[mode].append({"best_reward": 0, "error": str(e)})
            else:
                print(f"    [{mode}]...")
                try:
                    rag = create_rag_agent(best_strategy, retriever=best_method, mode=mode,
                                           prior_protocols=prior_protocols if best_source != "papers" else None)
                    rag.knowledge_source = KnowledgeSource()
                    rag.last_retrieved_docs = retriever.retrieve(q, store)
                    rag.last_retrieved_context = build_retrieved_context(rag.last_retrieved_docs)

                    if mode == "pre_gen":
                        result = rag._run_pre_gen(q)
                    elif mode == "post_gen":
                        result = rag._run_post_gen(q)
                    elif mode == "iterative":
                        result = rag._run_iterative(q)
                    elif mode == "full_pipeline":
                        result = rag._run_full_pipeline(q)
                    else:
                        result = rag.run(q)

                    mode_results[mode].append(result)
                    print(f"      → reward={result.get('best_reward', 0):.1f}")
                except Exception as e:
                    print(f"      [!] Error ({mode}): {e}")
                    mode_results[mode].append({"best_reward": 0, "error": str(e)})

    # Summary
    _print_header("4.3 Results: Integration Depth Ranking")
    print(f"\n  {'Mode':<16} {'Avg Reward':<12} {'Comb R':<9} {'Comb I':<9} {'vs No-RAG':<12} {'API Cost':<10}")
    print(f"  {'-'*16} {'-'*12} {'-'*9} {'-'*9} {'-'*12} {'-'*10}")

    base_avg = _avg([r.get("best_reward", 0) for r in mode_results.get("no_rag", [])])

    ranking = []
    for mode in modes:
        valid = [r for r in mode_results[mode] if "error" not in r]
        avg_r = _avg([r.get("best_reward", 0) for r in valid])
        six = {}
        if valid:
            for k in ["combined_avg_reliability", "combined_avg_innovation"]:
                six[k] = _avg([r.get("six_scores", {}).get(k, 0) for r in valid])

        delta = round(avg_r - base_avg, 1)
        api_cost = _estimate_api_cost(mode, best_strategy)
        print(f"  {mode:<16} {avg_r:<12.1f} {six.get('combined_avg_reliability',0):.1f}      {six.get('combined_avg_innovation',0):.1f}      {delta:+.1f}        {api_cost}")
        ranking.append((mode, avg_r, delta))

    ranking.sort(key=lambda x: x[1], reverse=True)
    best_mode = ranking[0][0]
    print(f"\n  Best integration mode: {best_mode} (reward={ranking[0][1]:.1f})")

    _save_results("exp43_integration_depth", {"mode_results": mode_results, "ranking": ranking, "best_mode": best_mode})
    return {"ranking": ranking, "best_mode": best_mode}


def _estimate_api_cost(mode: str, strategy: str) -> str:
    """Rough API call count estimate."""
    base = {"ppo": 8, "grpo": 10, "gspo": 12, "dapo": 15, "odp": 12}.get(strategy, 10)
    multiplier = {
        "no_rag": 1.0,
        "pre_gen": 1.1,      # +1 embedding call + 1 PubMed call
        "post_gen": 1.3,     # + critique + refine calls
        "iterative": 1.8,    # ×2 rounds
        "full_pipeline": 2.2, # all stages
    }
    return f"~{int(base * multiplier.get(mode, 1.0))} calls"


# =========================================================
# Run All
# =========================================================

def run_all():
    """Run experiments 4.1 → 4.2 → 4.3, passing best configs."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4: RAG-Enhanced RL Experiment Protocol Generation")
    print("  Running sub-experiments 4.1 → 4.2 → 4.3")
    print("=" * 70)

    # 4.1: Find best retrieval config
    exp41 = run_exp41_retrieval_ablation()
    best_config = exp41["best_config"]
    print(f"\n  >>> Best retrieval config from 4.1: {best_config}")

    # 4.2: RAG × RL cross with best config
    exp42 = run_exp42_rag_rl_cross(best_config=best_config)
    best_strategy = exp42["best_rag"]
    print(f"\n  >>> Best RAG strategy from 4.2: {best_strategy}")

    # 4.3: Integration depth with best strategy and config
    exp43 = run_exp43_integration_depth(best_strategy=best_strategy, best_config=best_config)
    best_mode = exp43["best_mode"]
    print(f"\n  >>> Best integration mode from 4.3: {best_mode}")

    # Final recommendation
    _print_header("FINAL RECOMMENDATION")
    print(f"  Best configuration: {best_config}")
    print(f"  Best strategy:      {best_strategy.upper()}")
    print(f"  Best RAG mode:      {best_mode}")
    print(f"  Recommended setup:  RAG-{best_strategy.upper()} + {best_mode} + {best_config}")

    _save_results("exp4_final_summary", {
        "best_config": best_config,
        "best_strategy": best_strategy,
        "best_mode": best_mode,
        "exp41": exp41,
        "exp42": exp42,
        "exp43": exp43,
    })


# =========================================================
# Save Utility
# =========================================================

def _save_results(name: str, data: Dict):
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, f"{name}_{int(time.time())}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[Saved] {output_path}")


# =========================================================
# Main Entry
# =========================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exp_arg = sys.argv[1] if len(sys.argv) > 1 else ""
        # Support --exp 4.1, --exp 4.2, etc.
        for i, arg in enumerate(sys.argv):
            if arg == "--exp" and i + 1 < len(sys.argv):
                exp_arg = sys.argv[i + 1]
                break

        if exp_arg in ("4.1", "41", "ablation"):
            run_exp41_retrieval_ablation()
        elif exp_arg in ("4.2", "42", "cross"):
            # Try to load best config from a previous 4.1 run
            best_config = None
            exp4_dir = os.path.dirname(os.path.abspath(__file__))
            for f in sorted(os.listdir(exp4_dir), reverse=True):
                if f.startswith("exp41_") and f.endswith(".json"):
                    try:
                        with open(os.path.join(exp4_dir, f)) as fp:
                            prev = json.load(fp)
                            best_config = prev.get("best")
                            print(f"  Loaded best config from {f}: {best_config}")
                            break
                    except Exception:
                        pass
            run_exp42_rag_rl_cross(best_config=best_config)
        elif exp_arg in ("4.3", "43", "depth"):
            run_exp43_integration_depth()
        elif exp_arg in ("all", "full"):
            run_all()
        else:
            print("Usage: python rag_runner.py --exp [4.1|4.2|4.3|all]")
            print("  4.1 — Retrieval Strategy Ablation")
            print("  4.2 — RAG × RL Cross Experiment")
            print("  4.3 — RAG Integration Depth")
            print("  all — Run all three sequentially (4.1→4.2→4.3)")
    else:
        # Interactive
        print("""
╔══════════════════════════════════════════════════════════════════╗
║    Experiment 4: RAG-Enhanced RL Protocol Generation             ║
║    External Knowledge (PubMed) × 5 RL Strategies                 ║
╚══════════════════════════════════════════════════════════════════╝

Commands:
  /exp41   — Retrieval Strategy Ablation (Dense/Sparse/Hybrid)
  /exp42   — RAG × RL Cross Experiment (5 strategies × RAG/NoRAG)
  /exp43   — RAG Integration Depth (4 injection modes)
  /all     — Run all experiments sequentially
  /quit    — Exit
""")
        while True:
            try:
                cmd = input("\n> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye!")
                break
            elif cmd in ("/exp41", "/41"):
                run_exp41_retrieval_ablation()
            elif cmd in ("/exp42", "/42"):
                run_exp42_rag_rl_cross()
            elif cmd in ("/exp43", "/43"):
                run_exp43_integration_depth()
            elif cmd in ("/all", "/run_all"):
                run_all()
            else:
                print("  Unknown command. Use /exp41, /exp42, /exp43, /all, or /quit.")

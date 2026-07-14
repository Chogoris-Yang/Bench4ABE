"""
Experiment_4/rag_components.py — RAG-enhanced RL base infrastructure.
Wraps Experiment_3 agents with retrieval-augmented generation.

Integration modes:
  pre_gen:      Retrieve → inject context → Actor generates
  post_gen:     Actor generates → retrieve → critique/refine
  iterative:    [Retrieve → generate → critique] × N rounds
  full_pipeline: pre_gen + post_gen + iterative

Pattern: Wrapper around Experiment_3 agents — no code duplication.
"""

import os
import sys
import time
import re
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod

# Import Experiment_3 components
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

from rl_components import (
    BaseRLAgent, ScoreTracker,
    call_llm_by_role, score_single_plan,
    GENERATION_PROMPT, OPTIMIZATION_PROMPT, DISTILLATION_PROMPT,
    format_six_scores_block,
)

from retrieval.retriever import (
    KnowledgeSource, DocumentStore,
    BaseRetriever, DenseRetriever, SparseRetriever, HybridRetriever,
    get_retriever,
)
from config_rag import (
    RETRIEVAL_TOP_K, RAG_CONTEXT_MAX_TOKENS,
    MODEL_ROLES,
)

# =========================================================
# RAG Prompt Templates
# =========================================================

RAG_GENERATION_PROMPT = """You are an expert AI research scientist with access to a KNOWLEDGE RETRIEVAL SYSTEM.

Below are RELEVANT RESEARCH PAPERS and PROTOCOLS retrieved for your task:

---
{retrieved_context}
---

Based on the research objective AND the retrieved knowledge above, generate a complete, rigorous, and innovative experimental plan.

Research Objective:
{question}

Additional Context:
{context}

INSTRUCTIONS:
1. INCORPORATE relevant methods and findings from the retrieved papers
2. CITE specific papers when you use their methodology (e.g., "As demonstrated by Smith et al. (2024)...")
3. IDENTIFY gaps in existing approaches that your plan could address
4. SYNTHESIZE insights from multiple retrieved sources
5. DO NOT fabricate citations — only reference papers shown above

Generate a comprehensive and well-structured experimental plan."""


RAG_OPTIMIZATION_PROMPT = """You are an expert research mentor optimizing an experimental plan with RETRIEVED KNOWLEDGE.

RETRIEVED KNOWLEDGE (external evidence):
---
{retrieved_context}
---

ORIGINAL PLAN:
----------------
{plan}
----------------

CURRENT SCORES: Reliability={reliability}, Innovation={innovation}
REWARD SIGNAL: {reward}
ADVANTAGE: {advantage}

{rl_specific_instructions}

OPTIMIZATION INSTRUCTIONS:
1. Use the retrieved knowledge to FIX weaknesses in the original plan
2. If retrieved papers suggest BETTER methods, adopt them
3. If original plan CONTRADICTS external evidence, correct it
4. ADD missing components that retrieved protocols include

Generate the improved plan directly."""


RAG_CRITIQUE_PROMPT = """You are an expert scientific reviewer with access to EXTERNAL KNOWLEDGE.

RETRIEVED KNOWLEDGE:
---
{retrieved_context}
---

PLAN TO CRITIQUE:
---
{plan}
---

Research Question: {question}

CRITIQUE TASK:
1. FACT-CHECK claims against retrieved papers — flag any contradictions
2. IDENTIFY missed opportunities — what methods from the literature should be included?
3. SUGGEST improvements grounded in the retrieved evidence
4. ASSESS novelty — does this plan go beyond existing work?

Output a structured critique. Be specific and cite the retrieved papers."""


RAG_TEACHER_PROMPT = """You are an EXPERT TEACHER generating a reference plan informed by EXTERNAL KNOWLEDGE.

RETRIEVED KNOWLEDGE:
---
{retrieved_context}
---

Research Question: {question}

Generate a gold-standard reference plan that:
1. SYNTHESIZES the best methods from retrieved papers
2. ADDS your own expert insights beyond the retrieved content
3. Provides a COMPLETE experimental protocol

This reference will guide the student's learning."""


# =========================================================
# RAG Context Builder
# =========================================================

def build_retrieved_context(docs: List[Dict], max_tokens: int = RAG_CONTEXT_MAX_TOKENS) -> str:
    """Format retrieved documents into a single context string (token-limited)."""
    if not docs:
        return "(No relevant external knowledge found.)"

    parts = []
    total_chars = 0
    char_limit = max_tokens * 3  # Rough: 1 token ≈ 3 chars

    for i, doc in enumerate(docs):
        source = doc.get("source", "unknown")
        if source == "pubmed":
            entry = (
                f"[Paper {i+1}] {doc.get('title', 'Untitled')}\n"
                f"  Journal: {doc.get('journal', 'N/A')} ({doc.get('year', 'N/A')})\n"
                f"  PMID: {doc.get('pmid', 'N/A')}\n"
                f"  Abstract: {doc.get('abstract', doc.get('text', ''))}\n"
            )
        else:
            entry = (
                f"[Protocol {i+1}] {doc.get('task', 'Prior protocol')}\n"
                f"  {doc.get('text', '')}\n"
            )

        if total_chars + len(entry) > char_limit:
            # Truncate last entry
            remaining = char_limit - total_chars
            if remaining > 200:
                parts.append(entry[:remaining] + "...")
            break

        parts.append(entry)
        total_chars += len(entry)

    return "\n".join(parts)


def build_prior_protocols_from_jsonl(jsonl_path: str, n: int = 10) -> List[Dict]:
    """
    Load prior experimental protocols from a JSONL dataset.
    Each protocol = {task: user question, text: any existing plan/method description}.
    """
    try:
        from rl_components import load_jsonl, extract_clean_question, extract_user_content
        all_entries = load_jsonl(jsonl_path, sample_size=0)
        protocols = []
        for entry in all_entries[:n]:
            user_content = extract_user_content(entry)
            protocols.append({
                "task": extract_clean_question(user_content),
                "text": user_content[:2000],  # Use the user request as protocol context
            })
        return protocols
    except Exception as e:
        print(f"  [WARN] Could not load prior protocols from {jsonl_path}: {e}")
        return []


# =========================================================
# RAG Wrapper — injects retrieval into any BaseRLAgent
# =========================================================

class RAGWrapper:
    """
    Wraps any BaseRLAgent from Experiment_3 with RAG capability.

    The wrapper intercepts generation calls and injects retrieved context
    without modifying the underlying agent's RL logic.

    Usage:
        from ppo_agent import PPOAgent
        agent = RAGWrapper(PPOAgent(), retriever=DenseRetriever(), mode="pre_gen")
        result = agent.run(question, knowledge_source=ks)
    """

    def __init__(
        self,
        base_agent: BaseRLAgent,
        retriever: BaseRetriever,
        mode: str = "pre_gen",
        knowledge_source: KnowledgeSource = None,
        prior_protocols: List[Dict] = None,
    ):
        self.agent = base_agent
        self.retriever = retriever
        self.mode = mode  # pre_gen, post_gen, iterative, full_pipeline
        self.knowledge_source = knowledge_source or KnowledgeSource()
        self.prior_protocols = prior_protocols or []
        self.top_k = RETRIEVAL_TOP_K

        # Store last retrieval for analysis
        self.last_retrieved_docs: List[Dict] = []
        self.last_retrieved_context: str = ""

    @property
    def strategy_name(self) -> str:
        return f"RAG-{self.agent.strategy_name}"

    def run(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the RAG-enhanced agent.

        1. Build knowledge store for the question
        2. Retrieve relevant documents
        3. Inject into the agent's generation process based on mode
        4. Run the agent with RAG-augmented generation
        """
        print(f"\n{'*'*70}")
        print(f"RAG-{self.agent.strategy_name} (mode={self.mode})")
        print(f"  Retriever: {self.retriever.name()}  |  Top-K: {self.top_k}")
        print(f"{'*'*70}")

        # Step 1: Build knowledge store
        print(f"\n  [RAG] Building knowledge store...")
        store = self.knowledge_source.build_store(
            question,
            prior_protocols=self.prior_protocols,
        )

        if store.size() == 0:
            print(f"  [RAG] WARNING: Empty knowledge store. Falling back to no-RAG.")
            return self.agent.run(question, verbose=verbose)

        # Step 2: Retrieve
        print(f"\n  [RAG] Retrieving (method={self.retriever.name()})...")
        self.last_retrieved_docs = self.retriever.retrieve(question, store, top_k=self.top_k)
        self.last_retrieved_context = build_retrieved_context(self.last_retrieved_docs)

        # Step 3: Inject RAG into agent based on mode
        if self.mode == "pre_gen":
            return self._run_pre_gen(question, verbose)
        elif self.mode == "post_gen":
            return self._run_post_gen(question, verbose)
        elif self.mode == "iterative":
            return self._run_iterative(question, verbose)
        elif self.mode == "full_pipeline":
            return self._run_full_pipeline(question, verbose)
        else:
            raise ValueError(f"Unknown RAG mode: {self.mode}")

    # ── Pre-Gen: Retrieve context before generation ──

    def _run_pre_gen(self, question: str, verbose: bool = True) -> Dict:
        """Inject retrieved context before Actor generates."""
        print(f"\n  [RAG pre_gen] Injecting {len(self.last_retrieved_docs)} docs into generation context")

        # Monkey-patch the agent's _generate_plan to include RAG context
        original_generate = self.agent._generate_plan

        def rag_generate(question: str, context: str = "", temperature: float = 0.3) -> str:
            return call_llm_by_role(
                "actor",
                "You are an expert with access to retrieved research knowledge.",
                RAG_GENERATION_PROMPT.format(
                    retrieved_context=self.last_retrieved_context,
                    question=question,
                    context=context,
                ),
                temperature=temperature,
            )

        self.agent._generate_plan = rag_generate

        try:
            result = self.agent.run(question, verbose=verbose)
            result["rag_mode"] = "pre_gen"
            result["rag_retriever"] = self.retriever.name()
            result["rag_docs_count"] = len(self.last_retrieved_docs)
            result["strategy"] = f"RAG-{self.agent.strategy_name}"
            return result
        finally:
            self.agent._generate_plan = original_generate

    # ── Post-Gen: Generate first, then retrieve to critique ──

    def _run_post_gen(self, question: str, verbose: bool = True) -> Dict:
        """Generate plan first, then use retrieved knowledge to critique and refine."""
        # Step 1: Generate without RAG
        print(f"\n  [RAG post_gen] Phase 1: Generate initial plan (no RAG)")
        result = self.agent.run(question, verbose=verbose)
        initial_plan = result.get("best_plan", "")
        initial_scores = result.get("best_scores", {})

        # Step 2: Retrieve relevant knowledge based on the generated plan
        print(f"\n  [RAG post_gen] Phase 2: Retrieve knowledge based on generated plan")
        # Also search for plan content
        additional_docs = self.retriever.retrieve(
            question + " " + initial_plan[:500],
            self.knowledge_source.build_store(question, prior_protocols=self.prior_protocols),
            top_k=self.top_k,
        )
        all_docs = self.last_retrieved_docs + additional_docs
        all_context = build_retrieved_context(all_docs)
        self.last_retrieved_context = all_context

        # Step 3: Critique using retrieved knowledge
        print(f"\n  [RAG post_gen] Phase 3: Critique with external knowledge")
        critique = call_llm_by_role(
            "actor",
            "You are an expert reviewer with access to published literature.",
            RAG_CRITIQUE_PROMPT.format(
                retrieved_context=all_context,
                plan=initial_plan[:5000],
                question=question,
            ),
            temperature=0.2,
        )
        print(f"    Critique: {len(critique)} chars")

        # Step 4: Refine based on critique
        print(f"\n  [RAG post_gen] Phase 4: Refine plan based on critique")
        refined_plan = call_llm_by_role(
            "actor",
            "You are refining an experimental plan based on expert critique and literature.",
            RAG_OPTIMIZATION_PROMPT.format(
                retrieved_context=all_context,
                plan=initial_plan[:5000],
                reliability=initial_scores.get("reliability", 0),
                innovation=initial_scores.get("innovation", 0),
                reward=result.get("best_reward", 0),
                advantage=0.0,
                rl_specific_instructions=f"CRITIQUE:\n{critique[:2000]}\n\nIncorporate these suggestions into an improved plan.",
            ),
            temperature=0.3,
        )

        # Score refined plan
        refined_scores = score_single_plan(refined_plan, verbose=verbose)
        refined_reward = refined_scores["reward"]
        initial_reward = result.get("best_reward", 0)

        print(f"    Post-gen improvement: {initial_reward:.1f} → {refined_reward:.1f} (Δ={refined_reward-initial_reward:+.1f})")

        if refined_reward > initial_reward:
            result["best_plan"] = refined_plan
            result["best_scores"] = refined_scores["combined"]
            result["best_reward"] = refined_reward

        result["rag_mode"] = "post_gen"
        result["rag_retriever"] = self.retriever.name()
        result["rag_docs_count"] = len(all_docs)
        result["post_gen_improvement"] = refined_reward - initial_reward
        result["rag_critique"] = critique
        result["strategy"] = f"RAG-{self.agent.strategy_name}"
        return result

    # ── Iterative: Alternate retrieve ↔ generate ↔ critique ──

    def _run_iterative(self, question: str, verbose: bool = True, rounds: int = 2) -> Dict:
        """Iterative RAG: generate → retrieve → critique → refine (N rounds)."""
        print(f"\n  [RAG iterative] {rounds} rounds of generate↔retrieve↔critique")

        # Start with pre-gen
        print(f"\n  --- Round 1 (pre_gen) ---")
        result = self._run_pre_gen(question, verbose=verbose)
        current_plan = result.get("best_plan", "")
        current_scores = result.get("best_scores", {})
        current_reward = result.get("best_reward", 0)

        all_critiques = []
        for r in range(1, rounds):
            print(f"\n  --- Round {r+1}/{rounds} (critique → refine) ---")

            # Retrieve fresh knowledge based on current plan
            store = self.knowledge_source.build_store(question, prior_protocols=self.prior_protocols)
            docs = self.retriever.retrieve(
                question + " " + current_plan[:300],
                store,
                top_k=self.top_k,
            )
            context = build_retrieved_context(docs)

            # Critique
            critique = call_llm_by_role(
                "actor",
                "Expert reviewer with literature access.",
                RAG_CRITIQUE_PROMPT.format(
                    retrieved_context=context,
                    plan=current_plan[:5000],
                    question=question,
                ),
                temperature=0.2,
            )
            all_critiques.append(critique)

            # Refine
            refined = call_llm_by_role(
                "actor",
                "Refining plan based on literature-backed critique.",
                RAG_OPTIMIZATION_PROMPT.format(
                    retrieved_context=context,
                    plan=current_plan[:5000],
                    reliability=current_scores.get("reliability", 0),
                    innovation=current_scores.get("innovation", 0),
                    reward=current_reward,
                    advantage=0.0,
                    rl_specific_instructions=f"CRITIQUE (Round {r+1}):\n{critique[:2000]}",
                ),
                temperature=0.3,
            )

            new_scores = score_single_plan(refined, verbose=verbose)
            new_reward = new_scores["reward"]
            print(f"    Round {r+1}: {current_reward:.1f} → {new_reward:.1f} (Δ={new_reward-current_reward:+.1f})")

            if new_reward > current_reward:
                current_plan = refined
                current_scores = new_scores["combined"]
                current_reward = new_reward

        result["best_plan"] = current_plan
        result["best_scores"] = current_scores
        result["best_reward"] = current_reward
        result["rag_mode"] = "iterative"
        result["rag_iterative_rounds"] = rounds
        result["rag_critiques"] = all_critiques
        result["strategy"] = f"RAG-{self.agent.strategy_name}"
        return result

    # ── Full Pipeline: pre_gen + post_gen + iterative + Critic RAG ──

    def _run_full_pipeline(self, question: str, verbose: bool = True) -> Dict:
        """Full RAG pipeline: all injection points active."""
        print(f"\n  [RAG full_pipeline] All injection points active")

        # Step 1: Pre-gen with RAG
        print(f"\n  === Stage 1: Pre-gen RAG ===")
        result = self._run_pre_gen(question, verbose=verbose)

        # Step 2: Post-gen critique
        print(f"\n  === Stage 2: Post-gen critique ===")
        plan = result.get("best_plan", "")
        scores = result.get("best_scores", {})

        store = self.knowledge_source.build_store(question, prior_protocols=self.prior_protocols)
        docs = self.retriever.retrieve(question + " " + plan[:300], store, top_k=self.top_k)
        context = build_retrieved_context(docs)

        critique = call_llm_by_role(
            "actor",
            "Expert reviewer.",
            RAG_CRITIQUE_PROMPT.format(retrieved_context=context, plan=plan[:5000], question=question),
            temperature=0.2,
        )

        # Step 3: Refine with RAG
        print(f"\n  === Stage 3: RAG-informed refinement ===")
        refined = call_llm_by_role(
            "actor",
            "Refining plan comprehensively.",
            RAG_OPTIMIZATION_PROMPT.format(
                retrieved_context=context,
                plan=plan[:5000],
                reliability=scores.get("reliability", 0),
                innovation=scores.get("innovation", 0),
                reward=result.get("best_reward", 0),
                advantage=0.0,
                rl_specific_instructions=f"POST-GEN CRITIQUE:\n{critique[:2000]}\n\nApply all improvements.",
            ),
            temperature=0.25,
        )

        new_scores = score_single_plan(refined, verbose=verbose)
        new_reward = new_scores["reward"]
        old_reward = result.get("best_reward", 0)

        if new_reward > old_reward:
            result["best_plan"] = refined
            result["best_scores"] = new_scores["combined"]
            result["best_reward"] = new_reward

        print(f"    Full pipeline: {old_reward:.1f} → {new_reward:.1f} (Δ={new_reward-old_reward:+.1f})")

        result["rag_mode"] = "full_pipeline"
        result["rag_retriever"] = self.retriever.name()
        result["rag_docs_count"] = len(docs)
        result["full_pipeline_improvement"] = new_reward - old_reward
        result["strategy"] = f"RAG-{self.agent.strategy_name}"
        return result


# =========================================================
# RAG Agent Factory
# =========================================================

def create_rag_agent(
    strategy_key: str,
    retriever: str = "hybrid",
    mode: str = "pre_gen",
    prior_protocols: List[Dict] = None,
) -> RAGWrapper:
    """
    Factory: create a RAG-wrapped agent for any Experiment_3 strategy.

    Args:
        strategy_key: "ppo" | "grpo" | "gspo" | "dapo" | "odp"
        retriever: "dense" | "sparse" | "hybrid"
        mode: "pre_gen" | "post_gen" | "iterative" | "full_pipeline"
        prior_protocols: List of {task, text} dicts

    Returns:
        RAGWrapper ready to call .run(question)
    """
    # Dynamic import of Experiment_3 agents
    strategy_map = {
        "ppo":  ("ppo_agent",  "PPOAgent"),
        "grpo": ("grpo_agent", "GRPOAgent"),
        "gspo": ("gspo_agent", "GSPOAgent"),
        "dapo": ("dapo_agent", "DAPOAgent"),
        "odp":  ("odp_agent",  "ODPAgent"),
    }

    if strategy_key not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy_key}. Options: {list(strategy_map.keys())}")

    module_name, class_name = strategy_map[strategy_key]
    import importlib
    module = importlib.import_module(module_name)
    agent_class = getattr(module, class_name)
    base_agent = agent_class()

    retriever_obj = get_retriever(retriever)
    ks = KnowledgeSource()

    return RAGWrapper(
        base_agent=base_agent,
        retriever=retriever_obj,
        mode=mode,
        knowledge_source=ks,
        prior_protocols=prior_protocols,
    )

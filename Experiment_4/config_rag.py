"""
Experiment_4/config_rag.py — RAG configuration for Experiment_4.

Sub-experiments:
  4.1 — Retrieval Strategy Ablation (Dense/Sparse/Hybrid × Papers/Prior/All)
  4.2 — RAG × RL Cross Experiment (5 strategies × with/without RAG)
  4.3 — RAG Integration Depth (Pre-Gen / Post-Gen / Iterative / Full Pipeline)
  (4.4 skipped — knowledge source ablation)

Model architecture:
  ACTOR    → deepseek-v4-pro   (plan generation)
  CRITIC   → kimi-k2.6         (value estimation, PPO)
  SECTION  → qwen3.7max        (token-level scoring, DAPO)
  AUX_TEACHER → GLM5.1         (auxiliary reference, ODP)
  EMBED    → text-embedding-3-small  (OpenAI embeddings for dense retrieval)
  JUDGE_DS → deepseek-chat     (scoring)
  JUDGE_GPT → gpt-5.4          (scoring + ODP teacher)
"""

import os
import sys

# =========================================================
# Import Experiment_3 config (extend, not replace)
# =========================================================
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXP3_DIR = None
for _candidate in ("experiment_3", "Experiment_3", "EXPERIMENT_3"):
    _path = os.path.join(_PARENT_DIR, _candidate)
    if os.path.isdir(_path):
        _EXP3_DIR = _path
        break

if _EXP3_DIR is not None and _EXP3_DIR not in sys.path:
    sys.path.insert(0, _EXP3_DIR)

# Dynamic import of Experiment_3 config
_exp3_config_loaded = False
if _EXP3_DIR is not None:
    try:
        _exp3_config = __import__("config")
        for _attr in dir(_exp3_config):
            if not _attr.startswith("_"):
                globals()[_attr] = getattr(_exp3_config, _attr)
        _exp3_config_loaded = True
    except ImportError:
        pass

# Fallback: define critical settings if not loaded from Exp3
if not _exp3_config_loaded:
    DEEPSEEK_API_KEY = "enter-your-api-key"
    AUTODL_API_KEY = "enter-your-api-key"
    AUTODL_BASE_URL = "https://www.autodl.art/api/v1"
    ACTOR_MODEL = "deepseek-v4-pro"
    REWARD_WEIGHTS = {"reliability": 0.6, "innovation": 0.4}

# =========================================================
# RAG-specific Model
# =========================================================
EMBED_MODEL = "text-embedding-3-small"
EMBED_API_KEY = AUTODL_API_KEY if hasattr(__import__('sys').modules.get('config', None), 'AUTODL_API_KEY') else "enter-your-api-key"
EMBED_BASE_URL = AUTODL_BASE_URL if hasattr(__import__('sys').modules.get('config', None), 'AUTODL_BASE_URL') else "https://www.autodl.art/api/v1"

# =========================================================
# PubMed Configuration
# =========================================================
PUBMED_MAX_RESULTS = 20          # Max PMIDs to fetch per query
PUBMED_TOP_K = 5                 # Top-K abstracts to return
PUBMED_EMAIL = "research@example.com"  # NCBI requires email

# =========================================================
# Retrieval Configuration
# =========================================================
RETRIEVAL_TOP_K = 5              # docs per retrieval call
DENSE_EMBED_MODEL = EMBED_MODEL
SPARSE_K1 = 1.5                  # BM25 k1 parameter
SPARSE_B = 0.75                  # BM25 b parameter
HYBRID_DENSE_CANDIDATES = 20     # Dense top-N before sparse rerank

# =========================================================
# RAG Integration Parameters
# =========================================================
RAG_CONTEXT_MAX_TOKENS = 3000    # Max tokens for retrieved context
RAG_INJECTION_MODES = {
    "pre_gen":     "Inject retrieved docs before Actor generates plan",
    "post_gen":    "Generate first, then retrieve to critique/refine",
    "iterative":   "Alternate generate↔retrieve↔critique for N rounds",
    "full_pipeline":"pre_gen + post_gen + iterative + Critic also uses RAG",
}

# =========================================================
# Sub-experiment Parameters
# =========================================================

# 4.1: Retrieval Strategy Ablation
EXP41_STRATEGY = "grpo"          # Fixed RL strategy
EXP41_N_QUESTIONS = 3            # Questions to test per combination
EXP41_RETRIEVAL_METHODS = ["dense", "sparse", "hybrid"]
EXP41_KNOWLEDGE_SOURCES = ["papers", "prior", "all"]

# 4.2: RAG × RL Cross Experiment
EXP42_N_QUESTIONS = 10           # All bio_test questions
EXP42_STRATEGIES = ["ppo", "grpo", "gspo", "dapo", "odp"]
EXP42_MODES = ["no_rag", "rag"]

# 4.3: RAG Integration Depth
EXP43_STRATEGY = "grpo"          # Best strategy from 4.2
EXP43_N_QUESTIONS = 5
EXP43_MODES = ["pre_gen", "post_gen", "iterative", "full_pipeline"]

# =========================================================
# Knowledge Source Paths
# =========================================================
EXPERIMENT_4_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_3_DIR = os.path.join(os.path.dirname(EXPERIMENT_4_DIR), "Experiment_3")
BIO_TEST_PATH = os.path.join(EXPERIMENT_3_DIR, "bio_test.jsonl")

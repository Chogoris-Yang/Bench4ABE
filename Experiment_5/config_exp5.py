"""
Experiment_5/config_exp5.py — Configuration for the 4-stage pipeline.
Completely self-contained — does NOT import or modify sys.path from Experiment_3.

Pipeline:
  Stage 1 — Information Retrieval (PubMed / PDF folder / Hybrid)
  Stage 2 — Plan Self-Iteration (generate + refine + issue tracking)
  Stage 3 — Prompt Iteration (prompt optimization + issue tracking)
  Stage 4 — Code Generation (experiment code output)

Models:
  Generator: deepseek-v4-pro  (via api.deepseek.com)
  Judges:    MiniMax-M2.5      (via autodl)
             GPT-5.4           (via autodl)
"""

import os

# =========================================================
# API Credentials (no dependency on Experiment_3 config)
# =========================================================
DEEPSEEK_API_KEY = "enter-your-api-key"
AUTODL_API_KEY   = "enter-your-api-key"

# =========================================================
# API Endpoints — separate base URLs per provider
# =========================================================
DEEPSEEK_BASE_URL = "https://api.deepseek.com"       # Generator
AUTODL_BASE_URL   = "https://www.autodl.art/api/v1"  # Judges (both MiniMax + GPT)

# =========================================================
# Model Assignment
# =========================================================
GENERATOR_MODEL = "deepseek-v4-pro"   # Plan/code generation → DeepSeek endpoint
JUDGE_MODEL_1   = "minimax-m2.5"      # Judge 1 → AutoDL endpoint
JUDGE_MODEL_2   = "gpt-5.4"           # Judge 2 → AutoDL endpoint

# =========================================================
# Proxy
# =========================================================
os.environ["HTTP_PROXY"]  = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

# =========================================================
# Paths (self-contained — no sys.path manipulation)
# =========================================================
EXPERIMENT_5_DIR   = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR         = os.path.join(EXPERIMENT_5_DIR, "memory")
ITERATION_ISSUES_PATH = os.path.join(MEMORY_DIR, "iteration_issues.md")
PROMPT_ISSUES_PATH    = os.path.join(MEMORY_DIR, "prompt_issues.md")
PDF_PAPERS_DIR     = os.path.join(EXPERIMENT_5_DIR, "pdf_papers")
GENERATED_CODE_DIR = os.path.join(EXPERIMENT_5_DIR, "generated_code")

# bio_test.jsonl — find via parent directory (no sys.path import)
_PARENT_DIR = os.path.dirname(EXPERIMENT_5_DIR)
_BIO_TEST_CANDIDATES = [
    os.path.join(_PARENT_DIR, "Experiment_3", "bio_test.jsonl"),
    os.path.join(_PARENT_DIR, "experiment_3", "bio_test.jsonl"),
    os.path.join(EXPERIMENT_5_DIR, "bio_test.jsonl"),  # fallback: local copy
]
BIO_TEST_PATH = ""
for _p in _BIO_TEST_CANDIDATES:
    if os.path.exists(_p):
        BIO_TEST_PATH = _p
        break

# =========================================================
# Stage 1: Retrieval Configuration
# =========================================================
RETRIEVAL_MODE = "hybrid"            # "pubmed" | "pdf" | "hybrid"
RETRIEVAL_TOP_K = 5
PUBMED_EMAIL = "research@example.com"

# =========================================================
# Pipeline — always 3 full rounds, no early stopping
# =========================================================
TOTAL_ROUNDS = 3

# =========================================================
# Scoring Weights
# =========================================================
STAGE_WEIGHTS = {
    "retrieval":       0.15,
    "plan_iteration":  0.35,
    "prompt_iteration": 0.20,
    "code_generation": 0.30,
}

REWARD_WEIGHTS = {
    "reliability": 0.6,
    "innovation":  0.4,
}

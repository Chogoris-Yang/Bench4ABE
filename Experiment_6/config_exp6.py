"""
Experiment_6/config_exp6.py — RL-enhanced pipeline with GLM optimizer.
Extends Experiment_5 pipeline with selectable RL strategies.
"""

import os, sys

# =========================================================
# API Credentials (same as Exp5)
# =========================================================
DEEPSEEK_API_KEY = "enter-your-api-key"
AUTODL_API_KEY   = "enter-your-api-key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
AUTODL_BASE_URL   = "https://www.autodl.art/api/v1"

# Models
GENERATOR_MODEL = "deepseek-v4-pro"   # Plan/code generation
RL_MODEL         = "GLM-5.1"           # RL policy optimizer
JUDGE_MODEL_1    = "minimax-m2.5"
JUDGE_MODEL_2    = "gpt-5.4"

os.environ["HTTP_PROXY"]  = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

# =========================================================
# RL Strategy
# =========================================================
RL_STRATEGY = "ppo"   # "ppo" | "grpo" | "direct"
PPO_CLIP_EPSILON = 0.2
GRPO_GROUP_SIZE = 3
TOTAL_ROUNDS = 3

# =========================================================
# Paths — find Experiment_5 for reuse
# =========================================================
_EXP6_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_EXP6_DIR)

# Add Experiment_5 to path for importing its modules
_EXP5_DIR = None
for _c in ("Experiment_5", "experiment_5", "EXPERIMENT_5"):
    _p = os.path.join(_PARENT_DIR, _c)
    if os.path.isdir(_p): _EXP5_DIR = _p; break
if _EXP5_DIR and _EXP5_DIR not in sys.path:
    sys.path.insert(0, _EXP5_DIR)

# bio_test.jsonl
for _c in ("Experiment_3", "experiment_3"):
    _p = os.path.join(_PARENT_DIR, _c, "bio_test.jsonl")
    if os.path.exists(_p): BIO_TEST_PATH = _p; break

GENERATED_CODE_DIR = os.path.join(_EXP6_DIR, "generated_code")
MEMORY_DIR = os.path.join(_EXP6_DIR, "memory")
ITERATION_ISSUES_PATH = os.path.join(MEMORY_DIR, "iteration_issues.md")
PROMPT_ISSUES_PATH    = os.path.join(MEMORY_DIR, "prompt_issues.md")
PDF_PAPERS_DIR = os.path.join(_EXP6_DIR, "pdf_papers")

RETRIEVAL_MODE = "hybrid"
RETRIEVAL_TOP_K = 5
PUBMED_EMAIL = "research@example.com"

STAGE_WEIGHTS = {"retrieval":0.15, "plan_iteration":0.35, "prompt_iteration":0.20, "code_generation":0.30}
REWARD_WEIGHTS = {"reliability":0.6, "innovation":0.4}

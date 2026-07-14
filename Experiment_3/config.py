"""
Experiment_3/config.py — Multi-model configuration for all 5 RL strategies.

Model Architecture (6 models, each with a dedicated RL role):
┌──────────────┬─────────────────────┬──────────────────────────────────┐
│ Role         │ Model               │ Used by                          │
├──────────────┼─────────────────────┼──────────────────────────────────┤
│ ACTOR        │ deepseek-v4-pro     │ All strategies (plan generation) │
│ CRITIC       │ kimi-k2.6           │ PPO (value function estimation)  │
│ SECTION      │ qwen3.7max          │ DAPO (token-level scoring)       │
│ AUX_TEACHER  │ GLM5.1              │ ODP (auxiliary reference plans)  │
│ JUDGE_DS     │ deepseek-chat       │ All strategies (Reliability+Inno)│
│ JUDGE_GPT    │ gpt-5.4             │ All strategies (Reliability+Inno)│
└──────────────┴─────────────────────┴──────────────────────────────────┘

Why multi-model?
  - PPO requires a separate Critic (Value) model distinct from the Actor policy
  - DAPO's token-level section scoring benefits from a model not used for generation
  - ODP's teacher-student paradigm needs teacher ≠ student models
  - Using different models prevents "self-judging" bias in the RL loop
"""

import os

# =========================================================
# Proxy Settings
# =========================================================
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

# =========================================================
# API Endpoints
# =========================================================
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
AUTODL_BASE_URL = "https://www.autodl.art/api/v1"

# =========================================================
# API Keys
# =========================================================
DEEPSEEK_API_KEY = "enter-your-api-key"
AUTODL_API_KEY = "enter-your-api-key"

# =========================================================
# Model Registry — Role → (model_name, api_key, base_url)
# =========================================================

# ── ACTOR: Primary plan generator (all strategies) ──
ACTOR_MODEL = "deepseek-v4-pro"
ACTOR_API_KEY = DEEPSEEK_API_KEY
ACTOR_BASE_URL = DEEPSEEK_BASE_URL

# ── CRITIC: Value function estimator (PPO) ──
# PPO needs a separate Critic model to estimate baseline V(s).
# Using the same model as Actor creates bias (the policy evaluates itself).
CRITIC_MODEL = "kimi-k2.6"
CRITIC_API_KEY = AUTODL_API_KEY
CRITIC_BASE_URL = AUTODL_BASE_URL

# ── SECTION JUDGE: Token/section-level scoring (DAPO) ──
# DAPO evaluates individual plan sections for fine-grained advantage.
# A separate model prevents the holistic judge from leaking section info.
SECTION_MODEL = "qwen3.7max"
SECTION_API_KEY = AUTODL_API_KEY
SECTION_BASE_URL = AUTODL_BASE_URL

# ── AUXILIARY TEACHER: Additional reference (ODP) ──
# ODP uses a teacher-student paradigm. GLM5.1 provides a second
# teacher perspective beyond GPT-5.4 for richer distillation.
AUX_TEACHER_MODEL = "GLM5.1"
AUX_TEACHER_API_KEY = AUTODL_API_KEY
AUX_TEACHER_BASE_URL = AUTODL_BASE_URL

# ── JUDGE 1: DeepSeek-based scorer ──
JUDGE_DS_MODEL = "deepseek-chat"
JUDGE_DS_API_KEY = DEEPSEEK_API_KEY
JUDGE_DS_BASE_URL = DEEPSEEK_BASE_URL

# ── JUDGE 2: GPT-based scorer ──
JUDGE_GPT_MODEL = "gpt-5.4"
JUDGE_GPT_API_KEY = AUTODL_API_KEY
JUDGE_GPT_BASE_URL = AUTODL_BASE_URL

# =========================================================
# Legacy aliases (backward compatibility)
# =========================================================
GENERATOR_MODEL = ACTOR_MODEL
JUDGE_MODEL_DS = JUDGE_DS_MODEL
JUDGE_MODEL_GPT = JUDGE_GPT_MODEL
DEEPSEEK_API_KEY = DEEPSEEK_API_KEY
GPT_API_KEY = AUTODL_API_KEY

# =========================================================
# Dataset Path
# =========================================================
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gen_lcot", "data", "train.jsonl"
)

# =========================================================
# RL Hyperparameters
# =========================================================

# PPO
PPO_CLIP_EPSILON = 0.2           # Clipping range
PPO_GAMMA = 0.95                 # Discount factor
PPO_LAMBDA = 0.95                # GAE lambda
PPO_KL_PENALTY = 0.02            # KL divergence penalty
PPO_N_EPOCHS = 3                 # Policy update epochs per batch
PPO_N_VARIANTS = 4               # Number of variants to generate per iteration

# GRPO
GRPO_GROUP_SIZE = 5              # K samples per group
GRPO_N_GENERATIONS = 3           # Number of group-generation rounds
GRPO_TEMPERATURE = 0.7           # Sampling temperature for diversity

# GSPO
GSPO_GROUP_SIZE = 4              # Sequential group size
GSPO_N_ROUNDS = 4                # Sequential sampling rounds
GSPO_ELITISM_RATIO = 0.25        # Top fraction to keep
GSPO_MUTATION_STRENGTH = 0.3     # Temperature for mutation

# DAPO
DAPO_CLIP_LOW = 0.2              # ε_low (asymmetric clipping)
DAPO_CLIP_HIGH = 0.3             # ε_high (asymmetric clipping)
DAPO_GROUP_SIZE = 5
DAPO_MIN_PLAN_LENGTH = 500       # Dynamic filter: minimum plan chars
DAPO_MAX_PLAN_LENGTH = 8000      # Overlong penalty threshold
DAPO_OVERLONG_PENALTY = 0.1      # Penalty coefficient for overlong plans

# ODP
ODP_TEACHER_TEMP = 0.1           # Teacher temperature (low = conservative)
ODP_STUDENT_TEMP = 0.7           # Student temperature (high = exploratory)
ODP_DISTILL_STEPS = 3            # Online distillation rounds
ODP_DISTILL_WEIGHT = 0.6         # α: teacher vs reward balance

# =========================================================
# General RL Settings
# =========================================================
MAX_ITERATIONS = 5               # Max RL iterations per task
REWARD_WEIGHTS = {               # How to combine Reliability & Innovation into scalar reward
    "reliability": 0.6,
    "innovation": 0.4,
}
PASS_THRESHOLD = 70.0            # Combined score threshold for "good enough"

# =========================================================
# Model Role Summary (for display/reference)
# =========================================================
MODEL_ROLES = {
    "ACTOR":        {"model": ACTOR_MODEL,     "role": "Plan generation (all strategies)"},
    "CRITIC":       {"model": CRITIC_MODEL,    "role": "Value function V(s) estimation (PPO)"},
    "SECTION":      {"model": SECTION_MODEL,   "role": "Token-level section scoring (DAPO)"},
    "AUX_TEACHER":  {"model": AUX_TEACHER_MODEL,"role": "Auxiliary teacher reference (ODP)"},
    "JUDGE_DS":     {"model": JUDGE_DS_MODEL,  "role": "Reliability + Innovation scoring"},
    "JUDGE_GPT":    {"model": JUDGE_GPT_MODEL, "role": "Reliability + Innovation scoring"},
}

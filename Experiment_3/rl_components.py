"""
Experiment_3/rl_components.py — Shared RL infrastructure.
Multi-client architecture: 6 models mapped to specific RL roles.
Provides: LLM dispatch, scoring with 6-metric tracking, data loading, base classes.
"""

import json
import os
import re
import time
import random
import math
import uuid
from typing import TypedDict, List, Dict, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod

from openai import OpenAI
import httpx

# Optional numpy — fall back to pure Python statistics
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    import statistics as _stats

from config import (
    # Actor
    ACTOR_MODEL, ACTOR_API_KEY, ACTOR_BASE_URL,
    # Critic
    CRITIC_MODEL, CRITIC_API_KEY, CRITIC_BASE_URL,
    # Section judge
    SECTION_MODEL, SECTION_API_KEY, SECTION_BASE_URL,
    # Aux teacher
    AUX_TEACHER_MODEL, AUX_TEACHER_API_KEY, AUX_TEACHER_BASE_URL,
    # Judge DS
    JUDGE_DS_MODEL, JUDGE_DS_API_KEY, JUDGE_DS_BASE_URL,
    # Judge GPT
    JUDGE_GPT_MODEL, JUDGE_GPT_API_KEY, JUDGE_GPT_BASE_URL,
    # Settings
    DATASET_PATH, REWARD_WEIGHTS, MAX_ITERATIONS, MODEL_ROLES,
    # Legacy
    DEEPSEEK_API_KEY,
    GENERATOR_MODEL, JUDGE_MODEL_DS, JUDGE_MODEL_GPT,
    AUTODL_API_KEY, AUTODL_BASE_URL, DEEPSEEK_BASE_URL,
)

# =========================================================
# Multi-Client Architecture (one client per RL role)
# =========================================================

_client_cache: Dict[str, OpenAI] = {}

def _get_client(api_key: str, base_url: str) -> OpenAI:
    """Get or create an OpenAI client for a given endpoint."""
    cache_key = f"{api_key}@{base_url}"
    if cache_key not in _client_cache:
        _client_cache[cache_key] = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(300.0)
        )
    return _client_cache[cache_key]


# ── Role-based client factories ──
actor_client = _get_client(ACTOR_API_KEY, ACTOR_BASE_URL)
critic_client = _get_client(CRITIC_API_KEY, CRITIC_BASE_URL)
section_client = _get_client(SECTION_API_KEY, SECTION_BASE_URL)
aux_teacher_client = _get_client(AUX_TEACHER_API_KEY, AUX_TEACHER_BASE_URL)
judge_ds_client = _get_client(JUDGE_DS_API_KEY, JUDGE_DS_BASE_URL)
judge_gpt_client = _get_client(JUDGE_GPT_API_KEY, JUDGE_GPT_BASE_URL)

# ── Legacy clients (backward compatibility) ──
deepseek_client = actor_client
gpt_client = judge_gpt_client

# ── Role → (client, model_name) dispatch table ──
ROLE_DISPATCH = {
    "actor":        (actor_client,        ACTOR_MODEL),
    "critic":       (critic_client,       CRITIC_MODEL),
    "section":      (section_client,      SECTION_MODEL),
    "aux_teacher":  (aux_teacher_client,  AUX_TEACHER_MODEL),
    "judge_ds":     (judge_ds_client,     JUDGE_DS_MODEL),
    "judge_gpt":    (judge_gpt_client,    JUDGE_GPT_MODEL),
}


def call_llm_by_role(
    role: str,
    system_prompt: str,
    user_input: str,
    temperature: float = 0.1,
    max_retries: int = 5,
    max_tokens: int = 8192,
) -> str:
    """
    Call an LLM identified by its RL role (not by raw model name).

    Roles: 'actor', 'critic', 'section', 'aux_teacher', 'judge_ds', 'judge_gpt'

    This is the PREFERRED way to call LLMs in all strategies.
    """
    if role not in ROLE_DISPATCH:
        raise ValueError(f"Unknown role '{role}'. Valid roles: {list(ROLE_DISPATCH.keys())}")

    client, model = ROLE_DISPATCH[role]
    return _call_llm_internal(client, model, role, system_prompt, user_input,
                              temperature, max_retries, max_tokens)


def _call_llm_internal(
    client: OpenAI,
    model: str,
    role_tag: str,
    system_prompt: str,
    user_input: str,
    temperature: float = 0.1,
    max_retries: int = 5,
    max_tokens: int = 8192,
) -> str:
    """Internal LLM caller with retry logic."""
    sleep_time = 1
    for attempt in range(1, max_retries + 1):
        try:
            print(f"    [{role_tag}] {model} attempt {attempt}...")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=temperature,
                timeout=180,
                max_tokens=max_tokens,
            )
            print(f"    [{role_tag}] {model} OK (attempt {attempt})")
            return response.choices[0].message.content
        except Exception as e:
            print(f"    [{role_tag}] error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(sleep_time)
                sleep_time = min(sleep_time * 2, 60)
    return "[ERROR] All API retries failed"


# Legacy wrapper — still available for backward compatibility
def call_llm(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_input: str,
    temperature: float = 0.1,
    max_retries: int = 5,
    max_tokens: int = 8192,
) -> str:
    """Legacy LLM caller. Prefer call_llm_by_role() in new code."""
    return _call_llm_internal(client, model, "legacy", system_prompt, user_input,
                              temperature, max_retries, max_tokens)


# =========================================================
# Score Tracker — 6 mandatory output metrics
# =========================================================

class ScoreTracker:
    """
    Tracks per-judge and combined scores across all iterations of a strategy run.

    The 6 required output scores:
      1. DS Average Reliability    (DeepSeek-chat judge)
      2. DS Average Innovation     (DeepSeek-chat judge)
      3. GPT Average Reliability   (GPT-5.4 judge)
      4. GPT Average Innovation    (GPT-5.4 judge)
      5. Combined Average Reliability  (DS+GPT averaged)
      6. Combined Average Innovation   (DS+GPT averaged)
    """

    def __init__(self):
        self.ds_reliabilities: List[float] = []
        self.ds_innovations: List[float] = []
        self.gpt_reliabilities: List[float] = []
        self.gpt_innovations: List[float] = []
        self.combined_reliabilities: List[float] = []
        self.combined_innovations: List[float] = []
        self.rewards: List[float] = []
        self.entries: List[Dict] = []  # Full record per scoring event

    def record(self, scores: Dict, label: str = "") -> None:
        """
        Record one scoring event.

        Args:
            scores: Dict from score_single_plan() with keys {ds, gpt, combined, reward}
            label: Optional label (e.g. "baseline", "epoch_1", "gen_2_plan_A")
        """
        ds = scores.get("ds", {})
        gpt = scores.get("gpt", {})
        combined = scores.get("combined", {})
        reward = scores.get("reward", 0)

        ds_rel = ds.get("reliability", 0)
        ds_inn = ds.get("innovation", 0)
        gpt_rel = gpt.get("reliability", 0)
        gpt_inn = gpt.get("innovation", 0)
        comb_rel = combined.get("reliability", 0)
        comb_inn = combined.get("innovation", 0)

        self.ds_reliabilities.append(ds_rel)
        self.ds_innovations.append(ds_inn)
        self.gpt_reliabilities.append(gpt_rel)
        self.gpt_innovations.append(gpt_inn)
        self.combined_reliabilities.append(comb_rel)
        self.combined_innovations.append(comb_inn)
        self.rewards.append(reward)

        self.entries.append({
            "label": label,
            "ds_reliability": ds_rel,
            "ds_innovation": ds_inn,
            "gpt_reliability": gpt_rel,
            "gpt_innovation": gpt_inn,
            "combined_reliability": comb_rel,
            "combined_innovation": comb_inn,
            "reward": reward,
        })

    def get_six_scores(self) -> Dict[str, float]:
        """
        Compute the 6 required average scores.

        Returns:
            {
                "ds_avg_reliability": float,
                "ds_avg_innovation": float,
                "gpt_avg_reliability": float,
                "gpt_avg_innovation": float,
                "combined_avg_reliability": float,
                "combined_avg_innovation": float,
            }
        """
        return {
            "ds_avg_reliability":      _avg(self.ds_reliabilities),
            "ds_avg_innovation":       _avg(self.ds_innovations),
            "gpt_avg_reliability":     _avg(self.gpt_reliabilities),
            "gpt_avg_innovation":      _avg(self.gpt_innovations),
            "combined_avg_reliability": _avg(self.combined_reliabilities),
            "combined_avg_innovation":  _avg(self.combined_innovations),
        }

    def get_best_reward(self) -> float:
        return max(self.rewards) if self.rewards else 0.0

    def get_best_entry(self) -> Optional[Dict]:
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e["reward"])

    def count(self) -> int:
        return len(self.entries)

    def summary_string(self) -> str:
        """One-line summary of the 6 scores."""
        s = self.get_six_scores()
        return (
            f"DS(R={s['ds_avg_reliability']:.1f} I={s['ds_avg_innovation']:.1f}) "
            f"GPT(R={s['gpt_avg_reliability']:.1f} I={s['gpt_avg_innovation']:.1f}) "
            f"→ Combined(R={s['combined_avg_reliability']:.1f} I={s['combined_avg_innovation']:.1f})"
        )


def format_six_scores_block(six: Dict[str, float], title: str = "SIX SCORE SUMMARY") -> str:
    """Format the 6 scores as a clearly labeled block for final reports."""
    return "\n".join([
        "",
        "-" * 50,
        f"  {title}",
        "-" * 50,
        f"  [Judge 1 — DeepSeek-chat]",
        f"    Avg Method Reliability:  {six['ds_avg_reliability']:.1f} / 100",
        f"    Avg Method Innovation:   {six['ds_avg_innovation']:.1f} / 100",
        "",
        f"  [Judge 2 — GPT-5.4]",
        f"    Avg Method Reliability:  {six['gpt_avg_reliability']:.1f} / 100",
        f"    Avg Method Innovation:   {six['gpt_avg_innovation']:.1f} / 100",
        "",
        f"  [Combined (DS + GPT average)]",
        f"    Avg Combined Reliability: {six['combined_avg_reliability']:.1f} / 100",
        f"    Avg Combined Innovation:  {six['combined_avg_innovation']:.1f} / 100",
        "-" * 50,
    ])


# =========================================================
# Data Loading
# =========================================================

_dataset_cache: Optional[List[Dict]] = None


def load_jsonl(path: str, sample_size: int = 0) -> List[Dict]:
    """
    Robust JSONL loader — fast line-by-line first, falls back to multi-line parser.
    Each JSON object must be on its own line with proper escaping (\\n for newlines).

    Args:
        path: Path to .jsonl file
        sample_size: If > 0, randomly sample this many entries (0 = all)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = []
    bad_lines = 0

    # ── Fast path: line-by-line (works for properly formatted JSONL) ──
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "messages" in obj and isinstance(obj["messages"], list) and len(obj["messages"]) >= 1:
                    data.append(obj)
                else:
                    bad_lines += 1
            except json.JSONDecodeError:
                bad_lines += 1
                if bad_lines <= 3:
                    print(f"  [WARN] Line {line_num}: JSON parse error — line may contain unescaped newlines")

    # ── Fallback: if ALL lines failed, try multi-line parsing ──
    if len(data) == 0 and bad_lines > 0:
        print(f"  [WARN] All {bad_lines} lines failed line-by-line. Trying multi-line JSON parser...")
        data = _load_jsonl_multiline(path)
        bad_lines = 0 if data else bad_lines
    elif bad_lines > 0:
        print(f"  [INFO] Loaded {len(data)} valid entries, skipped {bad_lines} malformed lines from {os.path.basename(path)}")

    if sample_size and sample_size < len(data):
        random.seed(42)
        data = random.sample(data, sample_size)

    return data


def _load_jsonl_multiline(path: str) -> List[Dict]:
    """
    Multi-line JSONL fallback: parses by tracking brace depth and string boundaries.
    Handles JSON objects whose string values contain literal (unescaped) newlines.
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    accumulated = ""
    depth = 0
    in_string = False
    escape_next = False

    for ch in raw_text:
        if escape_next:
            escape_next = False
            accumulated += ch
            continue
        if ch == '\\':
            escape_next = True
            accumulated += ch
            continue
        if ch == '"':
            in_string = not in_string
            accumulated += ch
            continue
        if not in_string:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    accumulated += ch
                    # Replace literal control chars inside strings before parsing
                    sanitized = _sanitize_json_strings(accumulated)
                    try:
                        obj = json.loads(sanitized)
                        if "messages" in obj and isinstance(obj["messages"], list):
                            data.append(obj)
                    except json.JSONDecodeError:
                        pass
                    accumulated = ""
                    continue
            elif depth == 0 and ch not in (' ', '\t', '\n', '\r'):
                continue
        accumulated += ch

    print(f"  [INFO] Multi-line parser recovered {len(data)} JSON objects")
    return data


def _sanitize_json_strings(text: str) -> str:
    """
    Escape literal control characters (newlines, tabs) inside JSON string values.
    Only modifies unescaped control chars that are inside quoted strings.
    This is a best-effort recovery for broken JSONL files.
    """
    result = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            escape_next = False
            result.append(ch)
            continue
        if ch == '\\':
            escape_next = True
            result.append(ch)
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string:
            if ch == '\n':
                result.append('\\n')
            elif ch == '\r':
                result.append('\\r')
            elif ch == '\t':
                result.append('\\t')
            else:
                result.append(ch)
        else:
            result.append(ch)
    return ''.join(result)


def load_dataset(path: str = DATASET_PATH, sample_size: int = 50) -> List[Dict]:
    """
    Load and randomly sample the dataset.
    Uses robust load_jsonl internally.
    """
    global _dataset_cache
    if _dataset_cache is not None and sample_size == 50:
        return _dataset_cache

    data = load_jsonl(path, sample_size=0)

    if sample_size and sample_size < len(data):
        random.seed(42)
        data = random.sample(data, sample_size)

    if sample_size == 50:
        _dataset_cache = data
    return data


def extract_user_content(sample: Dict) -> str:
    """Extract user question from a conversation sample."""
    for msg in sample.get("messages", []):
        if msg["role"] == "user":
            return msg["content"]
    return ""


def extract_clean_question(raw_text: str) -> str:
    """Extract clean research question from raw user content."""
    match = re.search(r"<user_request>\s*(.*?)(?:<think>|$)", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()[:500]
    return raw_text[:500]


# =========================================================
# Scoring System (Dual-Judge with ScoreTracker support)
# =========================================================

SCORING_PROMPT = """You are an expert scientific reviewer.

Please evaluate the given research methodology from the following two aspects:

1. Method Reliability (0-100)
2. Method Innovation (0-100)

Scoring Criteria:
Method Reliability: technical correctness, experimental rigor, feasibility, logical consistency, reproducibility
Method Innovation: originality, novelty, creativity, research contribution, uniqueness of methodology

Your output format MUST strictly follow:
Reliability: xx
Innovation: xx

Only output the scores. Do not provide explanations."""


def parse_score(text: Optional[str]) -> Dict[str, int]:
    """Parse Reliability and Innovation scores from judge output."""
    if text is None:
        return {"reliability": 0, "innovation": 0}
    try:
        rel_match = re.search(r"Reliability\s*:\s*(\d+)", text, re.IGNORECASE)
        inn_match = re.search(r"Innovation\s*:\s*(\d+)", text, re.IGNORECASE)
        return {
            "reliability": int(rel_match.group(1)) if rel_match else 0,
            "innovation": int(inn_match.group(1)) if inn_match else 0,
        }
    except Exception:
        return {"reliability": 0, "innovation": 0}


def score_single_plan(
    plan_text: str,
    verbose: bool = True,
    tracker: Optional[ScoreTracker] = None,
    label: str = "",
) -> Dict:
    """
    Dual-judge scoring of a single plan.
    Uses JUDGE_DS (DeepSeek-chat) + JUDGE_GPT (GPT-5.4) via role dispatch.

    Returns: {ds, gpt, combined, reward}
    Optionally records into a ScoreTracker for the 6-score summary.
    """
    if verbose:
        print(f"    [judge_ds] scoring...")
    ds_text = call_llm_by_role("judge_ds", SCORING_PROMPT, plan_text, temperature=0.0)
    ds = parse_score(ds_text)

    if verbose:
        print(f"    [judge_gpt] scoring...")
    gpt_text = call_llm_by_role("judge_gpt", SCORING_PROMPT, plan_text, temperature=0.0)
    gpt = parse_score(gpt_text)

    combined_rel = round((ds["reliability"] + gpt["reliability"]) / 2, 1)
    combined_inn = round((ds["innovation"] + gpt["innovation"]) / 2, 1)

    reward = round(
        combined_rel * REWARD_WEIGHTS["reliability"] +
        combined_inn * REWARD_WEIGHTS["innovation"],
        1
    )

    result = {
        "ds": ds,
        "gpt": gpt,
        "combined": {"reliability": combined_rel, "innovation": combined_inn},
        "reward": reward,
    }

    # Track scores
    if tracker is not None:
        tracker.record(result, label=label)

    if verbose:
        print(f"    Scores: DS(R={ds['reliability']} I={ds['innovation']}) "
              f"GPT(R={gpt['reliability']} I={gpt['innovation']}) "
              f"→ Combined(R={combined_rel} I={combined_inn}) reward={reward}")

    return result


def score_plans_parallel(
    plans: List[Dict],
    max_workers: int = 3,
    tracker: Optional[ScoreTracker] = None,
) -> List[Dict]:
    """
    Score multiple plans in parallel using ThreadPoolExecutor.
    Each plan dict should have a 'plan' key with the plan text.
    """
    def _score(p: Dict) -> Dict:
        result = dict(p)
        label = f"plan_{p.get('plan_id', '?')}"
        result["scores"] = score_single_plan(p["plan"], verbose=True, tracker=tracker, label=label)
        return result

    if len(plans) <= 1:
        return [_score(p) for p in plans]

    results = []
    with ThreadPoolExecutor(max_workers=min(len(plans), max_workers)) as pool:
        futures = {pool.submit(_score, p): i for i, p in enumerate(plans)}
        for f in as_completed(futures):
            results.append(f.result())

    # Sort back by plan_id
    try:
        results.sort(key=lambda x: str(x.get("plan_id", "")))
    except Exception:
        pass
    return results


# =========================================================
# RL Utility Functions
# =========================================================

def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    if _HAS_NUMPY:
        return float(np.mean(values))
    return _stats.mean(values)


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    if _HAS_NUMPY:
        return float(np.std(values))
    return _stats.stdev(values)


def compute_advantage_group(rewards: List[float], normalize: bool = True) -> List[float]:
    """Group-relative advantage (GRPO-style). A_i = (r_i - μ) / σ"""
    if len(rewards) <= 1:
        return [0.0] * len(rewards)
    mean_r = _mean(rewards)
    std_r = _std(rewards)
    if normalize and std_r > 0:
        return [(r - mean_r) / std_r for r in rewards]
    elif std_r > 0:
        return [r - mean_r for r in rewards]
    else:
        return [0.0] * len(rewards)


def compute_discounted_returns(rewards: List[float], gamma: float = 0.95) -> float:
    total = 0.0
    for i, r in enumerate(rewards):
        total += r * (gamma ** i)
    return total


def clip_ratio(ratio: float, epsilon: float) -> float:
    return max(1.0 - epsilon, min(ratio, 1.0 + epsilon))


def compute_combined_score(reliability: float, innovation: float) -> float:
    return reliability * REWARD_WEIGHTS["reliability"] + innovation * REWARD_WEIGHTS["innovation"]


def pass_threshold(scores: Dict) -> bool:
    c = scores.get("combined", {})
    return c.get("reliability", 0) >= 70 and c.get("innovation", 0) >= 60


def _avg(lst: List[float]) -> float:
    return round(sum(lst) / len(lst), 1) if lst else 0.0


# =========================================================
# Prompt Templates
# =========================================================

GENERATION_PROMPT = """You are an expert AI research scientist. Based on the following research objective,
generate a complete, rigorous, and innovative experimental plan.

Requirements:
1. Clearly state the Research Objective
2. Design the Proposed Methodology with technical depth
3. Provide detailed Experimental Design including variables, controls, and procedures
4. Specify Datasets and Evaluation Metrics
5. Analyze Expected Outcomes and Potential Limitations
6. Maintain rigorous scientific reasoning
7. Encourage methodological innovation
8. Ensure the plan is practical and reproducible

Research Objective:
{question}

Additional Context:
{context}

Generate a comprehensive and well-structured experimental plan."""

OPTIMIZATION_PROMPT = """You are an expert research mentor. Improve the following experimental plan
based on the provided feedback and reward signal.

ORIGINAL PLAN:
----------------
{plan}
----------------

CURRENT SCORES: Reliability={reliability}, Innovation={innovation}
REWARD SIGNAL: {reward}
ADVANTAGE: {advantage}

{rl_specific_instructions}

Generate the improved plan directly. Make it comprehensive and academically professional.
Focus specifically on the improvement areas indicated by the advantage signal."""

GROUP_GENERATION_PROMPT = """You are an expert AI research scientist. Generate {k} DISTINCT experimental plans
for the following research question. Each plan should take a GENUINELY DIFFERENT approach.

Research Question:
{question}

Requirements:
- Each plan must have a unique core idea and methodology
- Explore diverse angles: theoretical, empirical, computational, cross-disciplinary
- Vary the evaluation strategies and dataset choices

Output exactly {k} plans as a JSON array:
[
  {{"plan_id": "A", "core_idea": "...", "plan": "complete plan text..."}},
  ...
]

Make the plans TRULY diverse — not just minor variations of each other."""

DISTILLATION_PROMPT = """You are an expert AI research scientist acting as a TEACHER model.
Your student has generated the following experimental plan. Provide structured feedback
that the student can use to improve.

STUDENT'S PLAN:
---------------
{student_plan}
---------------

YOUR OWN REFERENCE PLAN:
---------------
{teacher_plan}
---------------

REWARD SIGNAL: {reward}
DISTILLATION WEIGHT: {alpha}

Provide:
1. What the student did well (preserve these strengths)
2. What the student should improve (specific, actionable)
3. Key insights from the teacher's approach that the student should absorb
4. A distilled improvement direction

Output your feedback in a structured format that guides the student's next iteration."""


# =========================================================
# Base RL Agent Class (updated with multi-client + ScoreTracker)
# =========================================================

class BaseRLAgent(ABC):
    """
    Abstract base class for all RL strategy agents.

    Key features:
    - Multi-client architecture via call_llm_by_role()
    - ScoreTracker for the mandatory 6-score output
    - Model role display on init
    """

    strategy_name: str = "base"

    def __init__(self):
        self.tracker = ScoreTracker()
        self.iteration_history: List[Dict] = []
        self.best_plan: str = ""
        self.best_scores: Dict = {}
        self.best_reward: float = 0.0
        self._printed_model_info = False

    def _print_model_roles(self):
        """Print which models are used by this strategy."""
        if self._printed_model_info:
            return
        self._printed_model_info = True
        print(f"\n  [{self.strategy_name}] Model Assignment:")
        for role_name, info in MODEL_ROLES.items():
            print(f"    {role_name:<14} → {info['model']:<20} ({info['role']})")

    @abstractmethod
    def run(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        """Run the RL strategy on a single research question."""
        pass

    def reset(self):
        """Reset agent state for a new task."""
        self.tracker = ScoreTracker()
        self.iteration_history = []
        self.best_plan = ""
        self.best_scores = {}
        self.best_reward = 0.0

    def _update_best(self, plan: str, scores: Dict, reward: float):
        """Track the best plan found so far."""
        if reward > self.best_reward or not self.best_plan:
            self.best_plan = plan
            self.best_scores = scores
            self.best_reward = reward

    def _generate_plan(self, question: str, context: str = "", temperature: float = 0.3) -> str:
        """Generate a plan using the ACTOR model."""
        return call_llm_by_role(
            "actor",
            "You are an expert AI research scientist generating experimental plans.",
            GENERATION_PROMPT.format(question=question, context=context),
            temperature=temperature,
        )

    def _generate_plan_critic(self, question: str, context: str = "", temperature: float = 0.1) -> str:
        """Generate a plan using the CRITIC model (for value estimation)."""
        return call_llm_by_role(
            "critic",
            "You are a value function estimator. Generate baseline assessment.",
            GENERATION_PROMPT.format(question=question, context=context),
            temperature=temperature,
        )

    def _generate_plan_section(self, prompt: str, system: str = "", temperature: float = 0.0) -> str:
        """Generate using the SECTION model (for token-level analysis)."""
        return call_llm_by_role("section", system or "You are a fine-grained plan section analyzer.", prompt, temperature=temperature)

    def _generate_plan_aux_teacher(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        """Generate using the AUX_TEACHER model (for ODP auxiliary reference)."""
        return call_llm_by_role("aux_teacher", system or "You are an auxiliary teacher providing reference plans.", prompt, temperature=temperature)

    def _score_plan(self, plan_text: str, label: str = "", verbose: bool = True) -> Dict:
        """Score a plan and automatically track scores."""
        return score_single_plan(plan_text, verbose=verbose, tracker=self.tracker, label=label)

    def _build_six_score_section(self) -> str:
        """Build the standardized 6-score output section for final reports."""
        six = self.tracker.get_six_scores()
        return format_six_scores_block(six, title=f"6-SCORE SUMMARY ({self.strategy_name})")

    def _build_final_report_header(self, topic: str, extra_config: List[str] = None) -> List[str]:
        """Build common header lines for final reports."""
        lines = [
            "=" * 70,
            f"   {self.strategy_name} — Experiment Protocol Report",
            "=" * 70,
            "",
            f"Research Topic: {topic[:300]}",
            "",
            "-" * 70,
            "Model Configuration (Multi-Client Architecture)",
            "-" * 70,
        ]
        for role_name, info in MODEL_ROLES.items():
            lines.append(f"  {role_name:<14} → {info['model']}")

        if extra_config:
            lines += ["", "-" * 70, "Strategy Configuration", "-" * 70]
            lines += extra_config

        return lines


# =========================================================
# Batch Evaluation Utilities
# =========================================================

def compute_batch_metrics(all_results: List[Dict], strategy_name: str) -> Dict:
    """
    Compute cross-sample average metrics for any RL strategy.
    Includes the 6-score averages across all samples.
    """
    all_rewards = []
    ds_rels, ds_inns = [], []
    gpt_rels, gpt_inns = [], []
    comb_rels, comb_inns = [], []

    for r in all_results:
        # Use the six_scores field if present
        six = r.get("six_scores", {})
        if six:
            ds_rels.append(six.get("ds_avg_reliability", 0))
            ds_inns.append(six.get("ds_avg_innovation", 0))
            gpt_rels.append(six.get("gpt_avg_reliability", 0))
            gpt_inns.append(six.get("gpt_avg_innovation", 0))
            comb_rels.append(six.get("combined_avg_reliability", 0))
            comb_inns.append(six.get("combined_avg_innovation", 0))

        all_rewards.append(r.get("best_reward", 0))

    return {
        "strategy": strategy_name,
        "sample_count": len(all_results),
        # 6-score cross-sample averages
        "cross_ds_avg_reliability":      _avg(ds_rels),
        "cross_ds_avg_innovation":       _avg(ds_inns),
        "cross_gpt_avg_reliability":     _avg(gpt_rels),
        "cross_gpt_avg_innovation":      _avg(gpt_inns),
        "cross_combined_avg_reliability": _avg(comb_rels),
        "cross_combined_avg_innovation":  _avg(comb_inns),
        # Legacy
        "avg_reward": _avg(all_rewards),
        "avg_reliability": _avg(comb_rels),
        "avg_innovation": _avg(comb_inns),
        "best_reward": max(all_rewards) if all_rewards else 0,
        "min_reward": min(all_rewards) if all_rewards else 0,
        "max_reward": max(all_rewards) if all_rewards else 0,
    }

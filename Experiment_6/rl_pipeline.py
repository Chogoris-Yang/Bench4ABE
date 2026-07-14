"""
Experiment_6/rl_pipeline.py — RL-enhanced 4-stage pipeline with GLM optimizer.

GLM analyzes judge scores/critiques from each round and generates strategic
refinement directives. The RL reward = score improvement between rounds.

Strategies:
  ppo    — GLM proposes changes; clipped advantage constrains update magnitude
  grpo   — GLM generates K alternative strategies; best wins via group-relative A
  direct — GLM directly outputs refinement strategy (no RL constraints, baseline)
"""

import os, sys, json, time, re, math
from typing import Dict, List
from openai import OpenAI
import httpx

from config_exp6 import (
    DEEPSEEK_API_KEY, AUTODL_API_KEY, DEEPSEEK_BASE_URL, AUTODL_BASE_URL,
    GENERATOR_MODEL, RL_MODEL, JUDGE_MODEL_1, JUDGE_MODEL_2,
    RL_STRATEGY, PPO_CLIP_EPSILON, GRPO_GROUP_SIZE, TOTAL_ROUNDS,
    MEMORY_DIR, ITERATION_ISSUES_PATH, PROMPT_ISSUES_PATH,
    PDF_PAPERS_DIR, GENERATED_CODE_DIR, RETRIEVAL_MODE, RETRIEVAL_TOP_K,
)

# ═══════════════════════════════════════
# Reuse Experiment_5 components
# ═══════════════════════════════════════
from pipeline_components import (
    call_llm, generator_client, judge_client,
    dual_judge_score, MemoryManager,
    load_bio_test_questions, extract_question,
    ScoreAccumulator, _avg, load_pdfs_from_directory,
)
from stages.stage1_retrieval import run_stage1, BM25Retriever, PubMedClient, build_pubmed_query

# GLM client via AutoDL
glm_client = OpenAI(api_key=AUTODL_API_KEY, base_url=AUTODL_BASE_URL, timeout=httpx.Timeout(300.0))

iter_mem = MemoryManager(ITERATION_ISSUES_PATH)
prompt_mem = MemoryManager(PROMPT_ISSUES_PATH)


# ═══════════════════════════════════════
# RL Strategy: GLM analyzes critiques, outputs refinement plan
# ═══════════════════════════════════════

GLM_POLICY_PROMPT = """You are an RL policy optimizer (GLM-5.1). Analyze the judge feedback from
the current round and determine the OPTIMAL REFINEMENT STRATEGY for the next round.

Research Question: {question}

ROUND {round} RESULTS:
- Stage2 Plan: Reliability={s2_rel}, Innovation={s2_inn}
- Stage3 Prompt: Reliability={s3_rel}, Innovation={s3_inn}
- Stage4 Code: Reliability={s4_rel}, Innovation={s4_inn}
- Round Reward: {reward}
- Reward Delta from Previous: {delta}

CRITIQUES:
Stage2: {s2_critique}
Stage3: {s3_critique}
Stage4: {s4_critique}

{rl_instructions}

Output a structured REFINEMENT STRATEGY:
1. PRIORITY: Which stage needs the most improvement? (Stage2/Stage3/Stage4)
2. FOCUS_AREAS: 2-3 specific aspects to fix (be concrete)
3. STRATEGY: How should the next round's generation differ?
4. AGGRESSIVENESS: (0-100) How radically should we change the approach?

Output as:
Priority: <stage>
Focus_Areas: <list>
Strategy: <text>
Aggressiveness: <number>"""

PPO_RL_INSTR = """PPO MODE: You operate under a clipped surrogate objective (ε={epsilon}).
- If the delta is positive, propose CONSERVATIVE refinements (stay close to current policy)
- If the delta is negative, propose more aggressive changes
- Your refinement ratio must stay within [1-ε, 1+ε] of the current approach"""

GRPO_RL_INSTR = """GRPO MODE: Generate {k} DISTINCT alternative refinement strategies.
Each strategy should take a DIFFERENT approach to improving the plan.
Output for EACH strategy: Priority, Focus_Areas, Strategy, Aggressiveness.
Format as JSON array of strategy objects."""

DIRECT_RL_INSTR = """DIRECT MODE: Output the single best refinement strategy.
No RL constraints — optimize freely based on the critiques."""


def glm_policy_decision(question, round_num, s2_sc, s3_sc, s4_sc, reward, delta,
                        s2_critique="", s3_critique="", s4_critique="",
                        strategy="ppo"):
    """GLM analyzes round results → outputs refinement directives."""
    rl_inst = {
        "ppo": PPO_RL_INSTR.format(epsilon=PPO_CLIP_EPSILON),
        "grpo": GRPO_RL_INSTR.format(k=GRPO_GROUP_SIZE),
        "direct": DIRECT_RL_INSTR,
    }.get(strategy, DIRECT_RL_INSTR)

    prompt = GLM_POLICY_PROMPT.format(
        question=question, round=round_num,
        s2_rel=s2_sc["combined"]["reliability"], s2_inn=s2_sc["combined"]["innovation"],
        s3_rel=s3_sc["combined"]["reliability"], s3_inn=s3_sc["combined"]["innovation"],
        s4_rel=s4_sc["combined"]["reliability"], s4_inn=s4_sc["combined"]["innovation"],
        reward=reward, delta=f"{delta:+.1f}",
        s2_critique=s2_critique[:1000], s3_critique=s3_critique[:1000], s4_critique=s4_critique[:1000],
        rl_instructions=rl_inst,
    )

    if strategy == "grpo":
        # Generate K alternative strategies
        raw = call_llm(glm_client, RL_MODEL,
            f"RL policy optimizer — generate {GRPO_GROUP_SIZE} diverse refinement strategies.",
            prompt, temperature=0.5, max_tokens=4096)
        strategies = _parse_json_strategies(raw)
        if len(strategies) >= 2:
            # Score each strategy heuristically (by aggressiveness appropriateness)
            # Positive delta → prefer lower aggressiveness; negative → higher
            best = min(strategies, key=lambda s: abs(s.get("aggressiveness", 50) - _target_aggressiveness(delta)))
            return best
        return {"priority": "Stage2", "focus_areas": ["methodology"], "strategy": raw[:500], "aggressiveness": 50}
    else:
        raw = call_llm(glm_client, RL_MODEL,
            "RL policy optimizer — determine optimal refinement strategy.",
            prompt, temperature=0.3, max_tokens=2048)
        return _parse_strategy(raw)


def _target_aggressiveness(delta: float) -> float:
    """Negative delta → more aggressive change needed."""
    if delta > 2: return 20
    if delta > 0: return 40
    if delta > -2: return 60
    return 80


def _parse_strategy(text: str) -> Dict:
    result = {"priority": "Stage2", "focus_areas": [], "strategy": text[:500], "aggressiveness": 50}
    for line in text.split("\n"):
        line = line.strip()
        if line.lower().startswith("priority:"):
            result["priority"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("focus_areas:"):
            areas = line.split(":", 1)[1].strip()
            result["focus_areas"] = [a.strip() for a in areas.split(",") if a.strip()]
        elif line.lower().startswith("strategy:"):
            result["strategy"] = line.split(":", 1)[1].strip()[:500]
        elif line.lower().startswith("aggressiveness:"):
            try:
                result["aggressiveness"] = int(re.search(r"\d+", line).group())
            except Exception:
                pass
    return result


def _parse_json_strategies(raw: str) -> List[Dict]:
    try:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            arr = json.loads(m.group(0))
            return [{**_parse_strategy(str(s)), **s} if isinstance(s, dict) else _parse_strategy(str(s)) for s in arr]
    except Exception:
        pass
    return [_parse_strategy(raw)]


# ═══════════════════════════════════════
# Plan/Code generation — reusing Exp5 patterns
# ═══════════════════════════════════════

GEN_INITIAL = """You are an expert AI research scientist. Generate a complete experimental plan.
Research Question: {question}
Retrieved Knowledge: {context}
Include: Objective, Methodology, Experimental Design, Datasets & Metrics, Limitations."""

GEN_REFINE = """Refine this plan based on judge feedback AND an RL optimizer's strategic directives.

Research Question: {question}
Retrieved Knowledge: {context}
PREVIOUS PLAN: {prev_plan}

JUDGE SCORES: Reliability={comb_rel}, Innovation={comb_inn}
RL OPTIMIZER STRATEGY:
- Priority Stage: {priority}
- Focus Areas: {focus_areas}
- Strategy Directive: {rl_strategy}
- Aggressiveness: {aggressiveness}/100 (higher = more radical change)

CRITIQUES FROM PAST ROUNDS:
{past_issues}

Improve the plan addressing ALL focus areas. Match the specified aggressiveness level."""

CODE_GEN = """Expert bioinformatics engineer. Generate Python code from this plan.
PLAN: {plan}
Question: {question}
RL Directive: {rl_strategy}
Generate 4 files: main.py, model.py, train.py, utils.py
Output: ===FILE:main.py=== (code) ===FILE:model.py=== (code) ===FILE:train.py=== (code) ===FILE:utils.py=== (code)"""

PROMPT_REFINE = """Improve this generation prompt based on RL optimizer feedback.
Question: {question}
Current Prompt: {current_prompt}
RL Optimization Strategy: {rl_strategy}
Past Prompt Issues: {past_prompt}
Output the complete improved prompt."""


# ═══════════════════════════════════════
# Single Question Runner
# ═══════════════════════════════════════

def run_rl_pipeline(question: str, q_idx: int, strategy: str = None) -> Dict:
    if strategy is None:
        strategy = RL_STRATEGY

    print(f"\n{'#'*70}\n  EXP6 Q{q_idx+1}: {question[:120]}\n  Strategy={strategy} | RL={RL_MODEL}\n{'#'*70}")

    # Stage 1: Retrieval (once)
    s1 = run_stage1(question, mode=RETRIEVAL_MODE)
    context = s1["context"]
    s1_sc = dual_judge_score(
        f"Research: {question}\nRetrieved:\n{context[:5000]}",
        label="retrieval", verbose=True)

    rounds_data = []
    current_plan, current_prompt = "", GEN_INITIAL
    prev_reward = s1_sc["reward"]

    for r in range(1, TOTAL_ROUNDS + 1):
        print(f"\n{'='*50}\n  ROUND {r}/{TOTAL_ROUNDS} [RL={strategy}]\n{'='*50}")
        rd = {"round": r}

        # ── Stage 2: Plan ──
        if r == 1:
            plan = call_llm(generator_client, GENERATOR_MODEL,
                "Expert scientist.", GEN_INITIAL.format(question=question, context=context[:5000]),
                temperature=0.3, max_tokens=8192)
        else:
            prev = rounds_data[-1]
            pol = prev.get("rl_policy", {})
            plan = call_llm(generator_client, GENERATOR_MODEL,
                "Expert refining plan via RL guidance.",
                GEN_REFINE.format(
                    question=question, context=context[:4000],
                    prev_plan=current_plan[:5000],
                    comb_rel=prev["s2_scores"]["combined"]["reliability"],
                    comb_inn=prev["s2_scores"]["combined"]["innovation"],
                    priority=pol.get("priority","Stage2"),
                    focus_areas=", ".join(pol.get("focus_areas",[])),
                    rl_strategy=pol.get("strategy",""),
                    aggressiveness=pol.get("aggressiveness",50),
                    past_issues=iter_mem.get_summary()[:2000],
                ), temperature=0.3 + pol.get("aggressiveness",50)/200, max_tokens=8192)
        current_plan = plan
        rd["plan"] = plan
        s2_sc = dual_judge_score(plan, label=f"plan_r{r}", verbose=True)
        rd["s2_scores"] = s2_sc

        # Critique for plan
        s2c = call_llm(generator_client, GENERATOR_MODEL, "Expert reviewer.",
            CRITIQUE_TMPL("plan", question, s2_sc, plan[:4000]),
            temperature=0.3, max_tokens=1536)
        rd["s2_critique"] = s2c
        iter_mem.append(question, f"[R{r}] Plan R={s2_sc['combined']['reliability']} I={s2_sc['combined']['innovation']}", s2c[:800])

        # ── Stage 3: Prompt ──
        past_p = prompt_mem.get_summary()
        current_prompt = call_llm(generator_client, GENERATOR_MODEL, "Prompt engineer.",
            PROMPT_REFINE.format(question=question, current_prompt=current_prompt[:3000],
                rl_strategy=pol.get("strategy","") if r>1 else "", past_prompt=past_p[:1500]),
            temperature=0.25, max_tokens=2048)
        rd["optimized_prompt"] = current_prompt
        s3_sc = dual_judge_score(f"Research: {question}\nPrompt:\n{current_prompt[:5000]}",
                                  label=f"prompt_r{r}", verbose=True)
        rd["s3_scores"] = s3_sc
        s3c = call_llm(generator_client, GENERATOR_MODEL, "Prompt critic.",
            CRITIQUE_TMPL("prompt", question, s3_sc, current_prompt[:3000]),
            temperature=0.3, max_tokens=1024)
        rd["s3_critique"] = s3c
        prompt_mem.append(question, f"[R{r}] Prompt R={s3_sc['combined']['reliability']}", s3c[:800])

        # ── Stage 4: Code ──
        code_raw = call_llm(generator_client, GENERATOR_MODEL, "Code engineer.",
            CODE_GEN.format(plan=plan[:8000], question=question,
                rl_strategy=pol.get("strategy","") if r>1 else ""),
            temperature=0.2, max_tokens=8192)
        code_files = _parse_code(code_raw)
        code_dir = _save_code(q_idx, r, code_files)
        rd["code_dir"] = code_dir
        rd["code_files"] = list(code_files.keys())
        code_txt = "\n\n".join(f"=== {fn} ===\n{fc[:2000]}" for fn, fc in list(code_files.items())[:2])
        s4_sc = dual_judge_score(f"Research: {question}\nCode:\n{code_txt[:6000]}",
                                  label=f"code_r{r}", verbose=True)
        rd["s4_scores"] = s4_sc
        s4c = call_llm(generator_client, GENERATOR_MODEL, "Code reviewer.",
            CRITIQUE_TMPL("code", question, s4_sc, code_txt[:3000]),
            temperature=0.3, max_tokens=1024)
        rd["s4_critique"] = s4c
        iter_mem.append(question, f"[R{r}] Code R={s4_sc['combined']['reliability']}", s4c[:800])

        # ── RL Policy Decision (GLM) ──
        reward = round(s2_sc["reward"]*0.5 + s3_sc["reward"]*0.2 + s4_sc["reward"]*0.3, 1)
        delta = reward - prev_reward
        prev_reward = reward
        rd["reward"] = reward
        rd["reward_delta"] = delta

        rl_policy = glm_policy_decision(
            question, r, s2_sc, s3_sc, s4_sc, reward, delta,
            s2_critique=s2c, s3_critique=s3c, s4_critique=s4c,
            strategy=strategy)
        rd["rl_policy"] = rl_policy
        print(f"  [GLM] Priority={rl_policy['priority']} Aggressiveness={rl_policy.get('aggressiveness','?')}")

        rounds_data.append(rd)

    # Build result
    return {
        "question_index": q_idx, "question": question,
        "rl_strategy": strategy, "rl_model": RL_MODEL,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage1": {"mode": s1["mode"], "doc_count": s1["doc_count"], "scores": s1_sc,
                   "retrieved_docs": [{"source": d.get("source","?"),
                       "title": d.get("title",d.get("filename",""))[:120]} for d in s1.get("retrieved_docs",[])]},
        "rounds": rounds_data,
        "summary": {
            "total_rounds": TOTAL_ROUNDS,
            "s1_reward": s1_sc["reward"],
            "s2_final": rounds_data[-1]["s2_scores"]["combined"],
            "s3_final": rounds_data[-1]["s3_scores"]["combined"],
            "s4_final": rounds_data[-1]["s4_scores"]["combined"],
            "final_reward": rounds_data[-1]["reward"],
            "rewards": [rd["reward"] for rd in rounds_data],
            "deltas": [rd["reward_delta"] for rd in rounds_data],
        },
    }


def CRITIQUE_TMPL(stage: str, question: str, scores: Dict, content: str) -> str:
    c = scores["combined"]
    return f"""Critique this {stage} (R={c['reliability']} I={c['innovation']}).
Question: {question}
Content: {content[:4000]}
Output: 1.Specific weaknesses 2.Concrete improvements 3.What to preserve."""


def _parse_code(raw: str) -> Dict[str, str]:
    files = {}
    for m in re.finditer(r'===FILE:(.+?)===\s*\n(.*?)(?=\n===FILE:|\Z)', raw, re.DOTALL):
        fn, code = m.group(1).strip(), m.group(2).strip()
        if code: files[fn] = code
    if not files:
        clean = re.sub(r'^```(?:python)?\s*\n?', '', raw, flags=re.MULTILINE)
        clean = re.sub(r'\n?```\s*$', '', clean, flags=re.MULTILINE)
        files["main.py"] = clean.strip()
    return files


def _save_code(q_idx: int, r: int, files: Dict[str, str]) -> str:
    d = os.path.join(GENERATED_CODE_DIR, f"q{q_idx+1:02d}_r{r}")
    os.makedirs(d, exist_ok=True)
    for fn, code in files.items():
        with open(os.path.join(d, fn), "w", encoding="utf-8") as f:
            f.write(f"# Exp6 Q{q_idx+1} R{r} — {fn}\n\n{code}")
    return d

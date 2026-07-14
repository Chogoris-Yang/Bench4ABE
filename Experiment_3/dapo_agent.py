"""
Experiment_3/dapo_agent.py
DAPO (Decoupled Alignment from Preferences Optimization) for experiment protocol generation.

Multi-model architecture:
  ACTOR    → deepseek-v4-pro (policy π — plan generation)
  SECTION  → qwen3.7max      (token/section-level fine-grained scoring)
  JUDGE_DS → deepseek-chat   (holistic Reliability + Innovation scoring)
  JUDGE_GPT → gpt-5.4        (holistic Reliability + Innovation scoring)

4 Key DAPO innovations:
  1. ASYMMETRIC CLIPPING: ε_high(0.3) > ε_low(0.2) — encourages upward exploration
  2. DYNAMIC SAMPLING: Filter low-quality samples before advantage computation
  3. TOKEN-LEVEL GRADIENT: Section-level scoring via qwen3.7max (not just holistic)
  4. OVERLONG REWARD SHAPING: Penalize plans exceeding length limit

Architecture: sample → dynamic_filter → section_score → asymmetric_advantages → [update → loop] → finalize
"""

import json
import time
import re
from typing import TypedDict, List, Dict, Any, Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from rl_components import (
    BaseRLAgent, ScoreTracker,
    call_llm_by_role,
    score_single_plan, score_plans_parallel,
    compute_advantage_group, _mean, _std,
    GROUP_GENERATION_PROMPT,
)
from config import (
    DAPO_CLIP_LOW, DAPO_CLIP_HIGH, DAPO_GROUP_SIZE,
    DAPO_MIN_PLAN_LENGTH, DAPO_MAX_PLAN_LENGTH, DAPO_OVERLONG_PENALTY,
    MAX_ITERATIONS, REWARD_WEIGHTS, MODEL_ROLES,
)

SECTION_SCORING_PROMPT = """Evaluate a specific SECTION of an experimental plan.

Evaluate on: 1. Technical Quality (0-100)  2. Completeness (0-100)

Output: Quality: xx\nCompleteness: xx"""

SECTION_EXTRACTION_PROMPT = """Extract sections from the experimental plan below. Mark missing sections as "NOT_FOUND".

PLAN: {plan}

Extract as JSON:
{{"objective":"...", "methodology":"...", "experimental_design":"...", "datasets_metrics":"...", "outcomes_limitations":"..."}}"""

DAPO_OPTIMIZE_PROMPT = """Optimize using ASYMMETRIC DAPO feedback.

DAPO Analysis:
  Overall reward: {reward}
  Asymmetric advantage: {advantage}σ  (ε_low={clip_low}, ε_high={clip_high})
  Section scores: {section_scores}
  Filter status: {filter_status}
  Length penalty: {length_penalty}

ASYMMETRIC RULE:
- Positive advantage → BOLD exploration allowed (ε_high={clip_high} ceiling)
- Negative advantage → CONSERVATIVE fixes only (ε_low={clip_low} floor)

Weak sections to redesign: {weak_sections}

ORIGINAL PLAN:
----------------
{plan}
----------------

Apply asymmetric update. For high-scoring sections: preserve (low mutation). For low-scoring sections: redesign (higher mutation)."""


def extract_sections(plan_text: str) -> Dict[str, str]:
    raw = call_llm_by_role(
        "section",
        "You are extracting structured sections from scientific plans.",
        SECTION_EXTRACTION_PROMPT.format(plan=plan_text[:6000]),
        temperature=0.0, max_tokens=4096,
    )
    try:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except json.JSONDecodeError:
        pass
    sections = {}
    patterns = {
        "objective": r"(?:Research\s+)?Objective[s]?\s*[:\-]\s*(.+?)(?:\n\n|\n(?=[A-Z]))",
        "methodology": r"(?:Proposed\s+)?Methodolog(?:y|ies)\s*[:\-]\s*(.+?)(?:\n\n|\n(?=[A-Z]))",
        "experimental_design": r"Experimental\s+Design\s*[:\-]\s*(.+?)(?:\n\n|\n(?=[A-Z]))",
        "datasets_metrics": r"(?:Datasets?|Evaluation)\s*(?:&\s*)?(?:Metrics?)?\s*[:\-]\s*(.+?)(?:\n\n|\n(?=[A-Z]))",
    }
    for key, pat in patterns.items():
        match = re.search(pat, plan_text, re.IGNORECASE | re.DOTALL)
        sections[key] = match.group(1).strip()[:2000] if match else "NOT_FOUND"
    sections.setdefault("outcomes_limitations", "NOT_FOUND")
    return sections


def score_section(section_name: str, section_text: str) -> Dict:
    if section_text == "NOT_FOUND" or len(section_text) < 50:
        return {"quality": 0, "completeness": 0}
    raw = call_llm_by_role(
        "section",
        f"Evaluating the '{section_name}' section.",
        SECTION_SCORING_PROMPT + f"\n\nSection: {section_name}\n\n{section_text[:2000]}",
        temperature=0.0,
    )
    from rl_components import parse_score
    scores = parse_score(raw)
    return {"quality": scores.get("reliability", 0), "completeness": scores.get("innovation", 0)}


def dynamic_filter(plans: List[Dict]) -> tuple:
    filtered, stats = [], {"removed": 0, "kept": 0, "reasons": []}
    for p in plans:
        plan_text = p.get("plan", "")
        reason = None
        if len(plan_text) < DAPO_MIN_PLAN_LENGTH:
            reason = f"too_short ({len(plan_text)} < {DAPO_MIN_PLAN_LENGTH})"
        elif plan_text.startswith("[ERROR]"):
            reason = "api_error"
        if reason:
            stats["removed"] += 1
            stats["reasons"].append(f"Plan {p.get('plan_id','?')}: {reason}")
        else:
            stats["kept"] += 1
            filtered.append(p)
    return filtered, stats


def compute_length_penalty(plan_text: str) -> float:
    length = len(plan_text)
    if length <= DAPO_MAX_PLAN_LENGTH:
        return 0.0
    return min(DAPO_OVERLONG_PENALTY * ((length - DAPO_MAX_PLAN_LENGTH) / 1000), 0.3)


class DAPOState(TypedDict):
    research_topic: str
    generation: int
    group_plans: List[Dict]
    all_section_scores: List[Dict]
    filter_stats: Dict
    asymmetric_advantages: List[float]
    generation_history: List[Dict]
    best_plan: str
    best_scores: Dict
    best_reward: float
    final_report: str


class DAPOAgent(BaseRLAgent):
    """DAPO — uses SECTION model (qwen3.7max) for token-level scoring."""

    strategy_name = "DAPO"

    def __init__(self):
        super().__init__()
        self.group_size = DAPO_GROUP_SIZE
        self.clip_low = DAPO_CLIP_LOW
        self.clip_high = DAPO_CLIP_HIGH
        self.n_generations = 3

    def build_graph(self) -> StateGraph:
        workflow = StateGraph(DAPOState)

        def sample_group(state: DAPOState) -> Dict:
            generation = state.get("generation", 0) + 1
            topic = state["research_topic"]

            if generation == 1:
                self._print_model_roles()

            print(f"\n{'='*60}")
            print(f"[DAPO] Gen {generation}/{self.n_generations} — Sampling (K={self.group_size})")
            print(f"  Actor: {MODEL_ROLES['ACTOR']['model']} | Section: {MODEL_ROLES['SECTION']['model']}")
            print(f"{'='*60}")

            raw = call_llm_by_role(
                "actor",
                "You are generating diverse plans for DAPO optimization.",
                GROUP_GENERATION_PROMPT.format(question=topic, k=self.group_size),
                temperature=0.6 + (generation - 1) * 0.1,
            )

            group_plans = []
            try:
                json_match = re.search(r"\[.*\]", raw, re.DOTALL)
                if json_match:
                    group_plans = json.loads(json_match.group(0))
            except (json.JSONDecodeError, Exception):
                parts = raw.split("\n\n")
                group_plans = [
                    {"plan_id": chr(65 + i), "core_idea": f"Approach {i+1}", "plan": p.strip()}
                    for i, p in enumerate(parts[:self.group_size]) if len(p.strip()) > 200
                ]

            for i, p in enumerate(group_plans):
                if "plan_id" not in p: p["plan_id"] = chr(65 + i)
                if "plan" not in p: p["plan"] = str(p)

            print(f"  → {len(group_plans)} plans generated")
            return {"generation": generation, "group_plans": group_plans[:self.group_size]}

        def filter_and_score(state: DAPOState) -> Dict:
            generation = state.get("generation", 1)
            group_plans = state.get("group_plans", [])

            print(f"\n{'='*60}")
            print(f"[DAPO] Dynamic Filter + Holistic Score + Section Score (Gen {generation})")
            print(f"{'='*60}")

            filtered, filter_stats = dynamic_filter(group_plans)
            print(f"  [Filter] {filter_stats['kept']} kept, {filter_stats['removed']} removed")
            for r in filter_stats.get("reasons", []):
                print(f"    REMOVED: {r}")
            if not filtered:
                filtered = group_plans

            # Holistic dual-judge scoring
            print(f"  [Holistic Scoring] {len(filtered)} plans...")
            scored = score_plans_parallel(filtered, max_workers=min(len(filtered), 3), tracker=self.tracker)

            # Length penalty
            for s in scored:
                lp = compute_length_penalty(s["plan"])
                s["length_penalty"] = lp
                s["length"] = len(s["plan"])
                reward = s.get("scores", {}).get("reward", 0)
                if lp > 0:
                    s["scores"]["reward_raw"] = reward
                    s["scores"]["reward"] = round(reward - lp * 100, 1)
                    print(f"    Plan {s['plan_id']}: len={s['length']}, penalty={lp:.3f}, reward adjusted")

            # Section-level scoring via SECTION model (qwen3.7max)
            print(f"\n  [Section Scoring] Using {MODEL_ROLES['SECTION']['model']} for token-level analysis...")
            all_section_scores = []
            for s in scored:
                sections = extract_sections(s["plan"])
                section_evals = {}
                for sec_name, sec_text in sections.items():
                    if sec_text != "NOT_FOUND" and len(sec_text) > 50:
                        section_evals[sec_name] = score_section(sec_name, sec_text)

                qualities = [e.get("quality", 0) for e in section_evals.values()]
                completenesses = [e.get("completeness", 0) for e in section_evals.values()]

                ss_data = {
                    "plan_id": s["plan_id"],
                    "sections": section_evals,
                    "avg_quality": sum(qualities) / len(qualities) if qualities else 0,
                    "avg_completeness": sum(completenesses) / len(completenesses) if completenesses else 0,
                    "weak_sections": [n for n, e in section_evals.items()
                                      if e.get("quality", 0) < 50 or e.get("completeness", 0) < 50],
                }
                all_section_scores.append(ss_data)
                print(f"    Plan {s['plan_id']}: sections={list(section_evals.keys())}, "
                      f"avg_Q={ss_data['avg_quality']:.1f}, weak={ss_data['weak_sections']}")

            # Asymmetric advantage
            rewards = [s.get("scores", {}).get("reward", 0) for s in scored]
            mean_r = _mean(rewards)
            std_r = _std(rewards) if len(rewards) > 1 else 1.0

            asymmetric_advantages = []
            for s in scored:
                r = s.get("scores", {}).get("reward", 0)
                raw_adv = (r - mean_r) / std_r if std_r > 0 else 0
                if raw_adv > 0:
                    clipped_adv = raw_adv * (1 + self.clip_high)  # Upward boost
                else:
                    clipped_adv = raw_adv * (1 + self.clip_low)   # Downward dampen
                asymmetric_advantages.append(clipped_adv)
                s["advantage"] = clipped_adv
                s["raw_advantage"] = raw_adv
                print(f"    Plan {s['plan_id']}: raw_A={raw_adv:+.2f}σ → asym_A={clipped_adv:+.2f}σ")

            # Track best
            best_in_gen = max(scored, key=lambda s: s.get("scores", {}).get("reward", 0))
            best_sc = best_in_gen.get("scores", {})
            best_reward = best_sc.get("reward", 0)
            prev_best = state.get("best_reward", 0)

            if best_reward > prev_best:
                gbp, gbs, gbr = best_in_gen["plan"], best_sc["combined"], best_reward
                print(f"  → New global best! ({prev_best:.1f} → {best_reward:.1f})")
            else:
                gbp = state.get("best_plan", best_in_gen.get("plan", ""))
                gbs = state.get("best_scores", best_sc.get("combined", {}))
                gbr = prev_best

            gen_history = list(state.get("generation_history", []))
            gen_history.append({
                "generation": generation, "filtered_count": len(scored),
                "removed_count": filter_stats["removed"], "best_reward": best_reward, "mean_reward": mean_r,
            })

            return {
                "group_plans": scored, "all_section_scores": all_section_scores,
                "filter_stats": filter_stats, "asymmetric_advantages": asymmetric_advantages,
                "generation_history": gen_history,
                "best_plan": gbp, "best_scores": gbs, "best_reward": gbr,
            }

        def dapo_update(state: DAPOState) -> Dict:
            generation = state.get("generation", 1)
            topic = state["research_topic"]
            group_plans = state.get("group_plans", [])
            all_section_scores = state.get("all_section_scores", [])

            print(f"\n{'='*60}")
            print(f"[DAPO] Asymmetric Update (Gen {generation}) — ε_low={self.clip_low}, ε_high={self.clip_high}")
            print(f"{'='*60}")

            section_lookup = {ss["plan_id"]: ss for ss in all_section_scores}
            optimized = []

            for gp in group_plans:
                advantage = gp.get("advantage", 0)
                sc = gp.get("scores", {})
                reward = sc.get("reward", 0)
                pid = gp["plan_id"]
                section_info = section_lookup.get(pid, {})
                weak = section_info.get("weak_sections", [])
                temp = 0.3 + advantage * 0.2 if advantage > 0.5 else max(0.1, 0.3 + advantage * 0.3)

                optimized_text = call_llm_by_role(
                    "actor",
                    f"DAPO asymmetric optimization for Plan {pid}.",
                    DAPO_OPTIMIZE_PROMPT.format(
                        plan=gp["plan"][:5000], reward=reward,
                        advantage=f"{advantage:+.2f}σ",
                        clip_low=self.clip_low, clip_high=self.clip_high,
                        section_scores=json.dumps({k: v for k, v in section_info.items()
                                                   if k not in ("plan_id", "weak_sections")},
                                                  ensure_ascii=False, indent=2),
                        filter_status="PASSED" if len(gp["plan"]) >= DAPO_MIN_PLAN_LENGTH else "FILTERED",
                        length_penalty=f"{gp.get('length_penalty', 0):.3f}",
                        weak_sections=", ".join(weak) if weak else "none",
                    ),
                    temperature=temp,
                )
                optimized.append({"plan_id": f"{pid}+", "original_id": pid, "plan": optimized_text,
                                  "dapo_advantage": advantage, "optimize_temp": temp})
                print(f"    DAPO update {pid}: A={advantage:+.2f}σ, T={temp:.2f}, weak=[{', '.join(weak) if weak else 'none'}]")

            return {"group_plans": optimized, "generation": generation + 1}

        def finalize(state: DAPOState) -> Dict:
            print(f"\n{'='*60}")
            print(f"[DAPO] Final Report")
            print(f"{'='*60}")

            topic = state["research_topic"]
            best_plan = state.get("best_plan", "")
            best_scores = state.get("best_scores", {})
            best_reward = state.get("best_reward", 0)
            gen_history = state.get("generation_history", [])

            display = best_plan[:5000] + "\n\n...(truncated)" if len(best_plan) > 5000 else best_plan

            header = self._build_final_report_header(topic, extra_config=[
                f"  Group size: {self.group_size}, Generations: {self.n_generations}",
                f"  Asymmetric clipping: ε_low={self.clip_low}, ε_high={self.clip_high}",
                f"  Dynamic filter: min_len={DAPO_MIN_PLAN_LENGTH}, overlong_penalty >{DAPO_MAX_PLAN_LENGTH}",
                f"  Actor: {MODEL_ROLES['ACTOR']['model']}",
                f"  Section Judge: {MODEL_ROLES['SECTION']['model']} (token-level scoring)",
            ])

            lines = header + [
                "",
                "-" * 70,
                "DAPO Innovations Applied",
                "-" * 70,
                "  1. Asymmetric Clipping: ε_high > ε_low encourages upward exploration",
                "  2. Dynamic Sampling: auto-filter low-quality samples",
                f"  3. Token-Level Gradient: section scoring via {MODEL_ROLES['SECTION']['model']}",
                "  4. Overlong Reward Shaping: length penalty for verbose plans",
                "",
                "-" * 70,
                "Generation History",
                "-" * 70,
            ]

            for gh in gen_history:
                lines.append(
                    f"  Gen {gh['generation']}: {gh['filtered_count']} valid ({gh['removed_count']} filtered), "
                    f"best_r={gh['best_reward']:.1f}, μ_r={gh['mean_reward']:.1f}"
                )

            if len(gen_history) >= 2:
                fb = gen_history[0]["best_reward"]
                lb = gen_history[-1]["best_reward"]
                lines.append(f"\n  DAPO improvement: {fb:.1f} → {lb:.1f} (Δ={lb-fb:+.1f})")

            lines.append(f"\n  Total scoring events: {self.tracker.count()}")
            lines.append(self._build_six_score_section())

            lines += [
                "",
                "-" * 70,
                "Best Plan (after DAPO optimization)",
                "-" * 70,
                "",
                display,
                "",
                "=" * 70,
            ]

            return {"final_report": "\n".join(lines)}

        def route_after_score(state: DAPOState) -> Literal["update", "finalize"]:
            if state.get("generation", 1) < self.n_generations:
                return "update"
            return "finalize"

        workflow.add_node("sample", sample_group)
        workflow.add_node("filter_score", filter_and_score)
        workflow.add_node("update", dapo_update)
        workflow.add_node("finalize", finalize)

        workflow.set_entry_point("sample")
        workflow.add_edge("sample", "filter_score")
        workflow.add_conditional_edges("filter_score", route_after_score, {"update": "update", "finalize": "finalize"})
        workflow.add_edge("update", "sample")
        workflow.add_edge("finalize", END)

        return workflow

    def run(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        self.reset()
        workflow = self.build_graph()
        app = workflow.compile(checkpointer=MemorySaver())

        initial: DAPOState = {
            "research_topic": question, "generation": 0, "group_plans": [],
            "all_section_scores": [], "filter_stats": {}, "asymmetric_advantages": [],
            "generation_history": [],
            "best_plan": "", "best_scores": {}, "best_reward": 0.0, "final_report": "",
        }

        config = {"configurable": {"thread_id": f"dapo-{int(time.time())}"}}

        print(f"\n{'*'*70}")
        print(f"DAPO: {question[:200]}")
        print(f"  K={self.group_size}, ε_low={self.clip_low}, ε_high={self.clip_high}")
        print(f"  Actor={MODEL_ROLES['ACTOR']['model']} | Section={MODEL_ROLES['SECTION']['model']}")
        print(f"{'*'*70}")

        final_state = None
        for event in app.stream(initial, config, stream_mode="values"):
            final_state = event

        if final_state is None:
            return {"final_report": "[ERROR]", "best_plan": "", "best_scores": {}, "best_reward": 0,
                    "six_scores": {}, "strategy": "DAPO"}

        return {
            "final_report": final_state.get("final_report", ""),
            "best_plan": final_state.get("best_plan", ""),
            "best_scores": final_state.get("best_scores", {}),
            "best_reward": final_state.get("best_reward", 0),
            "generation_history": final_state.get("generation_history", []),
            "generations_completed": final_state.get("generation", 0),
            "six_scores": self.tracker.get_six_scores(),
            "score_entries": self.tracker.entries,
            "strategy": "DAPO",
        }

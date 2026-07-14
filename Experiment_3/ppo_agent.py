"""
Experiment_3/ppo_agent.py
PPO (Proximal Policy Optimization) for experiment protocol generation.

Multi-model architecture:
  ACTOR   → deepseek-v4-pro (policy π — plan generation)
  CRITIC  → kimi-k2.6       (value function V(s) — baseline estimation)
  JUDGE_DS → deepseek-chat  (Reliability + Innovation scoring)
  JUDGE_GPT → gpt-5.4       (Reliability + Innovation scoring)

Math:
  L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
  where A_t = reward - V(s)  (Critic provides V(s))

Why separate Critic?
  Using the Actor to estimate its own value creates self-evaluation bias.
  kimi-k2.6 provides an independent baseline for unbiased advantage computation.

Architecture (LangGraph):
  actor_baseline → critic_estimate → generate_variants → [clip & loop] → finalize
"""

import json
import time
import math
from typing import TypedDict, List, Dict, Any, Optional, Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from rl_components import (
    BaseRLAgent, ScoreTracker,
    call_llm_by_role,
    score_single_plan, score_plans_parallel,
    compute_advantage_group, clip_ratio,
    GENERATION_PROMPT, OPTIMIZATION_PROMPT,
)
from config import (
    PPO_CLIP_EPSILON, PPO_GAMMA, PPO_N_EPOCHS, PPO_N_VARIANTS,
    MAX_ITERATIONS, REWARD_WEIGHTS, MODEL_ROLES,
)


# =========================================================
# PPO-specific Prompts
# =========================================================

CRITIC_PROMPT = """You are a VALUE FUNCTION ESTIMATOR (Critic model).
Your job is to estimate the EXPECTED quality of experimental plans for a research question
BEFORE seeing any actual plan. This is V(s) — the state value.

Research Question:
{question}

Based on the research question alone, analyze:
1. Domain complexity: How hard is this problem?
2. Methodology maturity: Are there well-established approaches?
3. Innovation potential: How much room is there for novel methods?

Then output your VALUE ESTIMATE:
Expected_Reliability: xx   (0-100 — what reliability score do you expect?)
Expected_Innovation: xx    (0-100 — what innovation score do you expect?)

Output format:
Expected_Reliability: xx
Expected_Innovation: xx

Only output these two numbers."""


PPO_INSTRUCTIONS = """PPO-SPECIFIC OPTIMIZATION:
The advantage signal (A={advantage:.1f}) = current_reward - critic_baseline({critic_baseline:.1f}).
This measures how much BETTER or WORSE your plan is compared to expectations.

- Positive advantage (A > 0): Plan EXCEEDS expectations. Make INCREMENTAL refinements.
- Negative advantage (A < 0): Plan BELOW expectations. Rethink approach more substantially.

Clipping constraint (ε={epsilon}):
  Probability ratio must stay in [{lower}, {upper}].
  Changes should be MEASURED, not radical rewrites."""


# =========================================================
# LangGraph State
# =========================================================

class PPOState(TypedDict):
    research_topic: str
    baseline_plan: str
    baseline_scores: Dict
    critic_value: Dict                # V(s) from CRITIC model
    current_plan: str
    current_scores: Dict
    variant_plans: List[Dict]
    epoch: int
    plan_history: List[Dict]
    best_plan: str
    best_scores: Dict
    best_reward: float
    final_report: str


# =========================================================
# PPO Graph Builder
# =========================================================

class PPOAgent(BaseRLAgent):
    """PPO — uses ACTOR (DeepSeek) for policy, CRITIC (kimi) for V(s)."""

    strategy_name = "PPO"

    def __init__(self):
        super().__init__()
        self.epsilon = PPO_CLIP_EPSILON

    def build_graph(self) -> StateGraph:
        workflow = StateGraph(PPOState)

        # ===== Node 1: Actor baseline (π_old) + Critic V(s) =====
        def init_baseline(state: PPOState) -> Dict:
            topic = state["research_topic"]
            self._print_model_roles()

            print(f"\n{'='*60}")
            print(f"[PPO] Phase 1: Actor Baseline (π_old) + Critic V(s)")
            print(f"  Actor:  {MODEL_ROLES['ACTOR']['model']}")
            print(f"  Critic: {MODEL_ROLES['CRITIC']['model']}")
            print(f"{'='*60}")

            # Actor generates baseline plan (conservative)
            print("  [ACTOR] Generating baseline plan...")
            baseline = call_llm_by_role(
                "actor",
                "You are a conservative, rigorous scientific planner.",
                GENERATION_PROMPT.format(
                    question=topic,
                    context="Use well-established, proven methodologies. Prioritize reliability."
                ),
                temperature=0.1,
            )
            print(f"  → Baseline plan: {len(baseline)} chars")

            # Score baseline (both judges)
            print("  [Scoring baseline]")
            baseline_scores = self._score_plan(baseline, label="baseline")

            # Critic estimates V(s) independently
            print(f"  [CRITIC] {MODEL_ROLES['CRITIC']['model']} estimating V(s)...")
            critic_raw = call_llm_by_role(
                "critic",
                "You are a value function V(s) estimator.",
                CRITIC_PROMPT.format(question=topic),
                temperature=0.0,
            )

            critic_val = parse_score_local(critic_raw)
            critic_reward = (
                critic_val.get("reliability", 50) * REWARD_WEIGHTS["reliability"] +
                critic_val.get("innovation", 50) * REWARD_WEIGHTS["innovation"]
            )
            print(f"    Critic V(s): Expected R={critic_val.get('reliability','?')} "
                  f"I={critic_val.get('innovation','?')} → V={critic_reward:.1f}")

            baseline_reward = baseline_scores["reward"]
            advantage = baseline_reward - critic_reward
            print(f"    A = reward({baseline_reward:.1f}) - V({critic_reward:.1f}) = {advantage:+.1f}")

            return {
                "baseline_plan": baseline,
                "baseline_scores": baseline_scores,
                "critic_value": {"reliability": critic_val.get("reliability", 50),
                                 "innovation": critic_val.get("innovation", 50),
                                 "reward": critic_reward},
                "current_plan": baseline,
                "current_scores": baseline_scores,
                "epoch": 0,
                "best_plan": baseline,
                "best_scores": baseline_scores["combined"],
                "best_reward": baseline_reward,
                "plan_history": [{
                    "epoch": 0,
                    "plan_type": "baseline",
                    "scores": baseline_scores,
                    "reward": baseline_reward,
                    "critic_v": critic_reward,
                    "advantage": advantage,
                }],
            }

        # ===== Node 2: Generate Variants + PPO Clip =====
        def generate_variants(state: PPOState) -> Dict:
            epoch = state.get("epoch", 0) + 1
            topic = state["research_topic"]
            current = state.get("current_plan", state.get("baseline_plan", ""))
            current_scores = state.get("current_scores", {})
            current_reward = current_scores.get("reward", 0)
            critic_value = state.get("critic_value", {}).get("reward", 50)
            advantage = current_reward - critic_value

            print(f"\n{'='*60}")
            print(f"[PPO] Epoch {epoch}/{PPO_N_EPOCHS} — Variants + PPO Clipping")
            print(f"  reward={current_reward:.1f}, V(s)={critic_value:.1f}, A={advantage:+.1f}")
            print(f"{'='*60}")

            combined_scores = current_scores.get("combined", {})

            # Generate N variants with ACTOR model
            variants = []
            for i in range(PPO_N_VARIANTS):
                temp = 0.2 + (i * 0.15)
                variant_text = call_llm_by_role(
                    "actor",
                    f"Exploratory variant generator #{i+1}.",
                    OPTIMIZATION_PROMPT.format(
                        plan=current[:5000],
                        reliability=combined_scores.get("reliability", 0),
                        innovation=combined_scores.get("innovation", 0),
                        reward=current_reward,
                        advantage=advantage,
                        rl_specific_instructions=PPO_INSTRUCTIONS.format(
                            advantage=advantage,
                            critic_baseline=critic_value,
                            epsilon=self.epsilon,
                            lower=1.0 - self.epsilon,
                            upper=1.0 + self.epsilon,
                        ),
                    ),
                    temperature=temp,
                )
                variants.append({"variant_id": i, "plan": variant_text})

            print(f"  → Generated {len(variants)} variants")

            # Score variants (automatically tracked via ScoreTracker)
            print("  [Scoring variants]")
            scored_variants = score_plans_parallel(variants, max_workers=min(PPO_N_VARIANTS, 3),
                                                   tracker=self.tracker)

            # PPO clipping computation
            for sv in scored_variants:
                r = sv.get("scores", {}).get("reward", 0)
                adv = r - critic_value
                raw_ratio = r / current_reward if current_reward > 0 else 1.0
                clipped_r = clip_ratio(raw_ratio, self.epsilon)
                ppo_obj = min(raw_ratio * adv, clipped_r * adv)

                sv["advantage"] = adv
                sv["raw_ratio"] = raw_ratio
                sv["clipped_ratio"] = clipped_r
                sv["ppo_objective"] = ppo_obj

                print(f"    Var {sv['variant_id']}: reward={r:.1f} A={adv:+.1f} "
                      f"ratio={raw_ratio:.3f} clipped={clipped_r:.3f} obj={ppo_obj:.1f}")

            # Select best by PPO objective
            best_variant = max(scored_variants, key=lambda v: v.get("ppo_objective", -999))
            best_v_reward = best_variant.get("scores", {}).get("reward", 0)

            if best_v_reward > current_reward:
                next_plan = best_variant["plan"]
                next_scores = best_variant["scores"]
                print(f"  → PPO ACCEPT: variant {best_variant['variant_id']} "
                      f"({current_reward:.1f} → {best_v_reward:.1f})")
            else:
                next_plan = current
                next_scores = current_scores
                print(f"  → PPO CLIP: no improvement, staying with current policy")

            plan_history = list(state.get("plan_history", []))
            plan_history.append({
                "epoch": epoch,
                "plan_type": "ppo_update",
                "num_variants": PPO_N_VARIANTS,
                "best_variant_reward": best_v_reward,
                "accepted": best_v_reward > current_reward,
                "advantage": advantage,
                "critic_v": critic_value,
            })

            # Track global best
            prev_best = state.get("best_reward", 0)
            if best_v_reward > prev_best:
                return {
                    "current_plan": next_plan,
                    "current_scores": next_scores,
                    "variant_plans": scored_variants,
                    "epoch": epoch,
                    "plan_history": plan_history,
                    "best_plan": best_variant["plan"],
                    "best_scores": best_variant["scores"]["combined"],
                    "best_reward": best_v_reward,
                }

            return {
                "current_plan": next_plan,
                "current_scores": next_scores,
                "variant_plans": scored_variants,
                "epoch": epoch,
                "plan_history": plan_history,
            }

        # ===== Node 3: Final Report with 6-score output =====
        def finalize(state: PPOState) -> Dict:
            print(f"\n{'='*60}")
            print(f"[PPO] Final Report")
            print(f"{'='*60}")

            topic = state["research_topic"]
            best_plan = state.get("best_plan", "")
            best_scores = state.get("best_scores", {})
            best_reward = state.get("best_reward", 0)
            baseline_scores = state.get("baseline_scores", {})
            plan_history = state.get("plan_history", [])
            critic_value = state.get("critic_value", {})

            improvement = best_reward - baseline_scores.get("reward", 0)
            display = best_plan[:5000] + "\n\n...(truncated)" if len(best_plan) > 5000 else best_plan

            header = self._build_final_report_header(topic, extra_config=[
                f"  Clip ε: {self.epsilon}",
                f"  Epochs: {PPO_N_EPOCHS}, Variants/epoch: {PPO_N_VARIANTS}",
                f"  Discount γ: {PPO_GAMMA}",
                f"  Actor:  {MODEL_ROLES['ACTOR']['model']}",
                f"  Critic: {MODEL_ROLES['CRITIC']['model']} (independent V(s) estimator)",
            ])

            lines = header + [
                "",
                "-" * 70,
                "PPO Training History",
                "-" * 70,
            ]

            for h in plan_history:
                if h.get("plan_type") == "baseline":
                    lines.append(
                        f"  Epoch 0 (π_old): reward={h['reward']:.1f}, "
                        f"V(s)={h.get('critic_v','?'):.1f}, A={h.get('advantage','?'):+.1f}"
                    )
                else:
                    lines.append(
                        f"  Epoch {h['epoch']}: {h['num_variants']} variants, "
                        f"best_reward={h['best_variant_reward']:.1f}, "
                        f"A={h.get('advantage',0):+.1f}, accepted={h.get('accepted',False)}"
                    )

            lines += [
                "",
                "-" * 70,
                "PPO Learning Outcome",
                "-" * 70,
                f"  Baseline reward (π_old): {baseline_scores.get('reward', 0):.1f}",
                f"  Best reward (π_new):    {best_reward:.1f}",
                f"  Δ Improvement:           {improvement:+.1f}",
                f"  Total scoring events:    {self.tracker.count()}",
            ]

            # === 6-SCORE OUTPUT (mandatory) ===
            lines.append(self._build_six_score_section())

            lines += [
                "",
                "-" * 70,
                "Best Experimental Plan",
                "-" * 70,
                "",
                display,
                "",
                "=" * 70,
            ]

            return {"final_report": "\n".join(lines)}

        # ===== Routing =====
        def route_after_variants(state: PPOState) -> Literal["variants", "finalize"]:
            epoch = state.get("epoch", 0)
            best_reward = state.get("best_reward", 0)
            if epoch >= PPO_N_EPOCHS:
                print(f"  → Max epochs reached, finalizing")
                return "finalize"
            if best_reward >= 88:
                print(f"  → Excellent reward ({best_reward}), finalizing")
                return "finalize"
            print(f"  → Continue PPO training (epoch {epoch}/{PPO_N_EPOCHS})")
            return "variants"

        # ===== Assemble Graph =====
        workflow.add_node("init", init_baseline)
        workflow.add_node("variants", generate_variants)
        workflow.add_node("finalize", finalize)

        workflow.set_entry_point("init")
        workflow.add_edge("init", "variants")
        workflow.add_conditional_edges(
            "variants", route_after_variants,
            {"variants": "variants", "finalize": "finalize"},
        )
        workflow.add_edge("finalize", END)

        return workflow

    def run(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        self.reset()
        workflow = self.build_graph()
        app = workflow.compile(checkpointer=MemorySaver())

        initial: PPOState = {
            "research_topic": question,
            "baseline_plan": "", "baseline_scores": {},
            "critic_value": {},
            "current_plan": "", "current_scores": {},
            "variant_plans": [],
            "epoch": 0, "plan_history": [],
            "best_plan": "", "best_scores": {}, "best_reward": 0.0,
            "final_report": "",
        }

        config = {"configurable": {"thread_id": f"ppo-{int(time.time())}"}}

        print(f"\n{'*'*70}")
        print(f"PPO: {question[:200]}")
        print(f"  ε={self.epsilon}, epochs={PPO_N_EPOCHS}, variants={PPO_N_VARIANTS}")
        print(f"  Actor={MODEL_ROLES['ACTOR']['model']} | Critic={MODEL_ROLES['CRITIC']['model']}")
        print(f"{'*'*70}")

        final_state = None
        for event in app.stream(initial, config, stream_mode="values"):
            final_state = event

        if final_state is None:
            return {"final_report": "[ERROR]", "best_plan": "", "best_scores": {}, "best_reward": 0,
                    "six_scores": {}, "strategy": "PPO"}

        return {
            "final_report": final_state.get("final_report", ""),
            "best_plan": final_state.get("best_plan", ""),
            "best_scores": final_state.get("best_scores", {}),
            "best_reward": final_state.get("best_reward", 0),
            "baseline_scores": final_state.get("baseline_scores", {}),
            "plan_history": final_state.get("plan_history", []),
            "epochs_completed": final_state.get("epoch", 0),
            "six_scores": self.tracker.get_six_scores(),
            "score_entries": self.tracker.entries,
            "strategy": "PPO",
        }


# Local helper (avoids circular import for parse_score in state graph)
def parse_score_local(text: Optional[str]) -> Dict[str, int]:
    from rl_components import parse_score
    return parse_score(text)

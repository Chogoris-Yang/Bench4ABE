"""
Experiment_3/grpo_agent.py
GRPO (Group Relative Policy Optimization) for experiment protocol generation.

Multi-model architecture:
  ACTOR    → deepseek-v4-pro (policy π — plan generation)
  (NO CRITIC needed — group-relative advantage replaces V(s))
  JUDGE_DS → deepseek-chat   (Reliability + Innovation scoring)
  JUDGE_GPT → gpt-5.4        (Reliability + Innovation scoring)

Math:
  A_i = (r_i - mean(r_group)) / std(r_group)    # Group-relative advantage
  No value model required!

Key innovation vs PPO: Eliminates the Critic model entirely.
Advantage is computed RELATIVE to the group — no external baseline needed.

Architecture (LangGraph):
  sample_group → score_all → compute_advantages → [optimize → loop] → finalize
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
    GENERATION_PROMPT, OPTIMIZATION_PROMPT, GROUP_GENERATION_PROMPT,
)
from config import (
    GRPO_GROUP_SIZE, GRPO_N_GENERATIONS, GRPO_TEMPERATURE,
    MAX_ITERATIONS, REWARD_WEIGHTS, MODEL_ROLES,
)


# =========================================================
# GRPO-specific Prompts
# =========================================================

GRPO_OPTIMIZE_PROMPT = """You are refining an experimental plan based on GROUP-RELATIVE feedback
(NO Critic/Value model — advantage is purely group-relative).

Your plan's position in the group:
  Your reward: {reward}
  Group mean: {mean_reward}
  Group std: {std_reward}
  Your advantage: {advantage}σ  (standard deviations above/below group mean)

{rl_specific_instructions}

ORIGINAL PLAN:
----------------
{plan}
----------------

Scores: Reliability={reliability}, Innovation={innovation}

Generate an improved plan. The advantage signal tells you how aggressively to change."""


GRPO_INSTRUCTIONS_HIGH = """ADVANTAGE={advantage}σ (>+1σ): TOP PERFORMER.
Make CONSERVATIVE, targeted refinements. Keep your successful core methodology.
Polish weak sections, add rigor, improve clarity. No major structural changes."""

GRPO_INSTRUCTIONS_MEDIUM = """ADVANTAGE={advantage}σ (near mean): AVERAGE.
Learn from what the best plans do differently. Add novelty from diverse angles.
Identify 2-3 specific improvements that would move you above the group mean."""

GRPO_INSTRUCTIONS_LOW = """ADVANTAGE={advantage}σ (<-1σ): BELOW MEAN.
Rethink your core approach. Study the high-advantage plans' strategies.
Consider a DIFFERENT methodology or experimental design paradigm."""

GRPO_COMPARE_PROMPT = """Analyze a group of experimental plans to identify success factors.

Research Question: {question}

Group Plans (sorted by reward, highest first):
{group_summaries}

Analyze:
1. What COMMON PATTERNS do top-ranked plans share?
2. What distinguishes the BEST plan from the rest?
3. What did the LOWEST-ranked plan do wrong?
4. What CROSS-CUTTING IMPROVEMENTS would lift all plans?"""


# =========================================================
# LangGraph State
# =========================================================

class GRPOState(TypedDict):
    research_topic: str
    generation: int
    group_plans: List[Dict]
    all_generations: List[Dict]
    group_insights: str
    best_plan: str
    best_scores: Dict
    best_reward: float
    final_report: str


# =========================================================
# GRPO Graph Builder
# =========================================================

class GRPOAgent(BaseRLAgent):
    """GRPO — group-relative advantage, NO Critic model needed."""

    strategy_name = "GRPO"

    def __init__(self):
        super().__init__()
        self.group_size = GRPO_GROUP_SIZE
        self.n_generations = GRPO_N_GENERATIONS

    def build_graph(self) -> StateGraph:
        workflow = StateGraph(GRPOState)

        def sample_group(state: GRPOState) -> Dict:
            generation = state.get("generation", 0) + 1
            topic = state["research_topic"]
            insights = state.get("group_insights", "")

            if generation == 1:
                self._print_model_roles()
                print(f"\n  [GRPO] No Critic model — advantage is group-relative only")

            print(f"\n{'='*60}")
            print(f"[GRPO] Generation {generation}/{self.n_generations} — Sampling Group (K={self.group_size})")
            print(f"  Actor: {MODEL_ROLES['ACTOR']['model']}")
            print(f"{'='*60}")

            context = ""
            if insights:
                context = f"INSIGHTS FROM PREVIOUS GENERATION:\n{insights[:2000]}"

            raw = call_llm_by_role(
                "actor",
                "You are a group-based generator. Create diverse experimental plans.",
                GROUP_GENERATION_PROMPT.format(question=topic, k=self.group_size)
                + (f"\n\n{context}" if context else ""),
                temperature=GRPO_TEMPERATURE if generation > 1 else 0.6,
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

            group_plans = group_plans[:self.group_size]
            print(f"  → Generated {len(group_plans)} plans")
            for gp in group_plans:
                print(f"     Plan {gp['plan_id']}: {gp.get('core_idea', gp['plan'][:60])}")

            return {"generation": generation, "group_plans": group_plans}

        def score_group(state: GRPOState) -> Dict:
            group_plans = list(state.get("group_plans", []))
            generation = state.get("generation", 1)

            print(f"\n{'='*60}")
            print(f"[GRPO] Scoring Group (Gen {generation}) — {len(group_plans)} plans")
            print(f"{'='*60}")

            scored = score_plans_parallel(group_plans, max_workers=min(self.group_size, 3),
                                          tracker=self.tracker)

            rewards = [s.get("scores", {}).get("reward", 0) for s in scored]
            advantages = compute_advantage_group(rewards, normalize=True)
            mean_r = _mean(rewards)
            std_r = _std(rewards) if len(rewards) > 1 else 1.0

            for i, s in enumerate(scored):
                s["advantage"] = advantages[i] if i < len(advantages) else 0.0
                sc = s.get("scores", {})
                c = sc.get("combined", {})
                print(f"    Plan {s['plan_id']}: R={c.get('reliability',0)} I={c.get('innovation',0)} "
                      f"reward={sc.get('reward',0):.1f} A={advantages[i]:+.2f}σ")

            print(f"  Group stats: μ_r={mean_r:.1f}, σ_r={std_r:.1f}")

            sorted_plans = sorted(scored, key=lambda s: s.get("scores", {}).get("reward", 0), reverse=True)
            summaries = []
            for rank, sp in enumerate(sorted_plans):
                sc = sp.get("scores", {})
                c = sc.get("combined", {})
                summaries.append(
                    f"Rank {rank+1}: Plan {sp['plan_id']} — R={c.get('reliability',0)} I={c.get('innovation',0)} "
                    f"reward={sc.get('reward',0):.1f} A={sp.get('advantage',0):+.2f}σ"
                )

            print("  [Generating group insights via ACTOR]")
            insights = call_llm_by_role(
                "actor",
                "You are analyzing group experimental data.",
                GRPO_COMPARE_PROMPT.format(
                    question=state["research_topic"],
                    group_summaries="\n".join(summaries),
                ),
                temperature=0.2,
            )

            best_in_group = sorted_plans[0]
            best_sc = best_in_group.get("scores", {})
            best_reward = best_sc.get("reward", 0)

            prev_best = state.get("best_reward", 0)
            if best_reward > prev_best:
                best_plan = best_in_group["plan"]
                best_scores = best_sc["combined"]
                print(f"  → New global best! ({prev_best:.1f} → {best_reward:.1f})")
            else:
                best_plan = state.get("best_plan", best_in_group.get("plan", ""))
                best_scores = state.get("best_scores", best_sc.get("combined", {}))
                best_reward = prev_best

            all_gens = list(state.get("all_generations", []))
            all_gens.append({
                "generation": generation,
                "group_size": len(scored),
                "mean_reward": mean_r,
                "std_reward": std_r,
                "best_reward": best_sc.get("reward", 0),
                "top_plan_id": best_in_group["plan_id"],
            })

            return {
                "group_plans": scored,
                "group_insights": insights,
                "all_generations": all_gens,
                "best_plan": best_plan,
                "best_scores": best_scores,
                "best_reward": best_reward,
            }

        def optimize_grpo(state: GRPOState) -> Dict:
            generation = state.get("generation", 1)
            topic = state["research_topic"]
            group_plans = state.get("group_plans", [])

            print(f"\n{'='*60}")
            print(f"[GRPO] Optimization — Group-relative advantage gradients")
            print(f"{'='*60}")

            optimized_plans = []
            for gp in group_plans:
                advantage = gp.get("advantage", 0)
                scores = gp.get("scores", {})
                reward = scores.get("reward", 0)
                sc = scores.get("combined", {})

                if advantage > 1.0:
                    instructions = GRPO_INSTRUCTIONS_HIGH.format(advantage=f"{advantage:+.2f}")
                elif advantage > -1.0:
                    instructions = GRPO_INSTRUCTIONS_MEDIUM.format(advantage=f"{advantage:+.2f}")
                else:
                    instructions = GRPO_INSTRUCTIONS_LOW.format(advantage=f"{advantage:+.2f}")

                temp = max(0.1, 0.4 - advantage * 0.15)

                optimized_text = call_llm_by_role(
                    "actor",
                    "You are optimizing plans via group-relative RL signal.",
                    GRPO_OPTIMIZE_PROMPT.format(
                        plan=gp["plan"][:5000],
                        reliability=sc.get("reliability", 0),
                        innovation=sc.get("innovation", 0),
                        reward=reward,
                        advantage=advantage,
                        mean_reward="group level",
                        std_reward="group level",
                        rl_specific_instructions=instructions,
                    ),
                    temperature=temp,
                )
                optimized_plans.append({
                    "plan_id": f"{gp['plan_id']}'",
                    "original_id": gp["plan_id"],
                    "plan": optimized_text,
                    "previous_advantage": advantage,
                })

            print(f"  → Optimized {len(optimized_plans)} plans")
            return {"group_plans": optimized_plans}

        def finalize(state: GRPOState) -> Dict:
            print(f"\n{'='*60}")
            print(f"[GRPO] Final Report")
            print(f"{'='*60}")

            topic = state["research_topic"]
            best_plan = state.get("best_plan", "")
            best_scores = state.get("best_scores", {})
            best_reward = state.get("best_reward", 0)
            all_gens = state.get("all_generations", [])

            display = best_plan[:5000] + "\n\n...(truncated)" if len(best_plan) > 5000 else best_plan

            header = self._build_final_report_header(topic, extra_config=[
                f"  Group size K: {self.group_size}",
                f"  Generations: {self.n_generations}",
                f"  Advantage: Group-relative (NO Critic/Value model)",
                f"  Actor: {MODEL_ROLES['ACTOR']['model']}",
                f"  Key innovation: Eliminates Value function — uses group statistics",
            ])

            lines = header + [
                "",
                "-" * 70,
                "Group Training History",
                "-" * 70,
            ]

            for g in all_gens:
                lines.append(
                    f"  Gen {g['generation']}: {g['group_size']} plans, "
                    f"μ_r={g['mean_reward']:.1f}, σ_r={g['std_reward']:.1f}, "
                    f"max_r={g['best_reward']:.1f}, top={g['top_plan_id']}"
                )

            if len(all_gens) >= 2:
                first_best = all_gens[0]["best_reward"]
                last_best = all_gens[-1]["best_reward"]
                lines.append(f"\n  Group improvement: {first_best:.1f} → {last_best:.1f} (Δ={last_best - first_best:+.1f})")

            lines += [
                "",
                f"  Total scoring events: {self.tracker.count()}",
            ]

            # === 6-SCORE OUTPUT ===
            lines.append(self._build_six_score_section())

            lines += [
                "",
                "-" * 70,
                "Best Plan (from GRPO group optimization)",
                "-" * 70,
                "",
                display,
                "",
                "=" * 70,
            ]

            return {"final_report": "\n".join(lines)}

        def route_after_score(state: GRPOState) -> Literal["optimize", "finalize"]:
            generation = state.get("generation", 1)
            if generation < self.n_generations:
                print(f"  → Continue optimization ({generation}/{self.n_generations})")
                return "optimize"
            print(f"  → Final generation complete")
            return "finalize"

        workflow.add_node("sample", sample_group)
        workflow.add_node("score", score_group)
        workflow.add_node("optimize", optimize_grpo)
        workflow.add_node("finalize", finalize)

        workflow.set_entry_point("sample")
        workflow.add_edge("sample", "score")
        workflow.add_conditional_edges("score", route_after_score,
                                       {"optimize": "optimize", "finalize": "finalize"})
        workflow.add_edge("optimize", "sample")
        workflow.add_edge("finalize", END)

        return workflow

    def run(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        self.reset()
        workflow = self.build_graph()
        app = workflow.compile(checkpointer=MemorySaver())

        initial: GRPOState = {
            "research_topic": question,
            "generation": 0, "group_plans": [],
            "all_generations": [], "group_insights": "",
            "best_plan": "", "best_scores": {}, "best_reward": 0.0,
            "final_report": "",
        }

        config = {"configurable": {"thread_id": f"grpo-{int(time.time())}"}}

        print(f"\n{'*'*70}")
        print(f"GRPO: {question[:200]}")
        print(f"  K={self.group_size}, generations={self.n_generations}")
        print(f"  Actor={MODEL_ROLES['ACTOR']['model']} | NO Critic (group-relative A)")
        print(f"{'*'*70}")

        final_state = None
        for event in app.stream(initial, config, stream_mode="values"):
            final_state = event

        if final_state is None:
            return {"final_report": "[ERROR]", "best_plan": "", "best_scores": {}, "best_reward": 0,
                    "six_scores": {}, "strategy": "GRPO"}

        return {
            "final_report": final_state.get("final_report", ""),
            "best_plan": final_state.get("best_plan", ""),
            "best_scores": final_state.get("best_scores", {}),
            "best_reward": final_state.get("best_reward", 0),
            "all_generations": final_state.get("all_generations", []),
            "generations_completed": final_state.get("generation", 0),
            "six_scores": self.tracker.get_six_scores(),
            "score_entries": self.tracker.entries,
            "strategy": "GRPO",
        }

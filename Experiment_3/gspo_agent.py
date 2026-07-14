"""
Experiment_3/gspo_agent.py
GSPO (Group Sampling Policy Optimization) — sequential elitism + directed mutation.

Multi-model architecture:
  ACTOR    → deepseek-v4-pro (policy π — plan generation + mutation)
  JUDGE_DS → deepseek-chat   (scoring)
  JUDGE_GPT → gpt-5.4        (scoring)

Math: Sequential group sampling with cumulative advantage:
  Round t: sample K → score → select elite (top E) → mutate → next round
  Policy improves by elitism + directed mutation (evolutionary strategy).

Architecture: init → score → [elite+mutate → score] × N → finalize
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
    score_plans_parallel,
    compute_advantage_group, _mean,
    GENERATION_PROMPT, GROUP_GENERATION_PROMPT,
)
from config import (
    GSPO_GROUP_SIZE, GSPO_N_ROUNDS, GSPO_ELITISM_RATIO, GSPO_MUTATION_STRENGTH,
    MAX_ITERATIONS, REWARD_WEIGHTS, MODEL_ROLES,
)

GSPO_MUTATION_PROMPT = """Apply DIRECTED MUTATION to an elite experimental plan (sequential policy optimization).

ELITE PLAN (Round {round}, Rank {rank}/{total_elite}):
---
{elite_plan}
---

Scores: Reliability={reliability}, Innovation={innovation}
Cumulative Advantage: {advantage}
Mutation: {mutation_direction}  (temperature={temperature})

Generate a MUTATED version. PRESERVE strengths, EXPLORE the variation direction, make COHERENT changes."""

MUTATION_DIRECTIONS = [
    "Increase methodological rigor — validation steps, formal definitions, reproducibility",
    "Boost innovation — cross-disciplinary technique or unconventional evaluation",
    "Improve practicality — implementation details, dataset specifics, feasibility",
    "Strengthen theoretical grounding — formal foundations, mathematical justification",
    "Enhance experimental design — ablation studies, control groups, statistical power",
    "Expand scope — generalize to broader problem classes",
    "Sharpen focus — narrow to most impactful sub-problem and go deeper",
]

ELITE_SELECTION_PROMPT = """Select elite plans for sequential group optimization.

Research Question: {question}  |  Round: {round}  |  Cumulative Best: {cumulative_best}

Scored Plans (sorted by reward):
{scored_summaries}

Elitism Ratio: {elite_ratio} ({num_elite} plans survive)

For each selected plan, specify mutation_direction and temperature (0.1=conservative, 0.5=exploratory).
Output as JSON array: [{{"plan_id":"...","mutation_direction":"...","temperature":0.X}}, ...]"""


class GSPOState(TypedDict):
    research_topic: str
    round: int
    group_plans: List[Dict]
    elite_plans: List[Dict]
    round_history: List[Dict]
    cumulative_best_reward: float
    best_plan: str
    best_scores: Dict
    best_reward: float
    final_report: str


class GSPOAgent(BaseRLAgent):
    """GSPO — sequential group sampling + elitism + directed mutation."""

    strategy_name = "GSPO"

    def __init__(self):
        super().__init__()
        self.group_size = GSPO_GROUP_SIZE
        self.n_rounds = GSPO_N_ROUNDS
        self.elite_ratio = GSPO_ELITISM_RATIO
        self.num_elite = max(1, int(self.group_size * self.elite_ratio))

    def build_graph(self) -> StateGraph:
        workflow = StateGraph(GSPOState)

        def init_group(state: GSPOState) -> Dict:
            topic = state["research_topic"]
            self._print_model_roles()
            print(f"\n  [GSPO] Sequential evolutionary optimization — no separate Critic needed")

            print(f"\n{'='*60}")
            print(f"[GSPO] Round 1/{self.n_rounds} — Initial Group (K={self.group_size})")
            print(f"  Actor: {MODEL_ROLES['ACTOR']['model']}")
            print(f"{'='*60}")

            raw = call_llm_by_role(
                "actor",
                "You are a sequential policy optimizer. Generate diverse initial plans.",
                GROUP_GENERATION_PROMPT.format(question=topic, k=self.group_size),
                temperature=0.6,
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
            print(f"  → {len(group_plans)} initial plans")
            return {"round": 1, "group_plans": group_plans}

        def score_group(state: GSPOState) -> Dict:
            round_num = state.get("round", 1)
            group_plans = state.get("group_plans", [])

            print(f"\n{'='*60}")
            print(f"[GSPO] Scoring Round {round_num} — {len(group_plans)} plans")
            print(f"{'='*60}")

            scored = score_plans_parallel(group_plans, max_workers=min(self.group_size, 3),
                                          tracker=self.tracker)
            scored.sort(key=lambda s: s.get("scores", {}).get("reward", 0), reverse=True)

            rewards = [s.get("scores", {}).get("reward", 0) for s in scored]
            advantages = compute_advantage_group(rewards, normalize=True)
            mean_r = _mean(rewards)

            for i, s in enumerate(scored):
                s["rank"] = i + 1
                s["advantage"] = advantages[i] if i < len(advantages) else 0.0
                sc = s.get("scores", {})
                c = sc.get("combined", {})
                print(f"    Rank {i+1}: Plan {s['plan_id']} R={c.get('reliability',0)} "
                      f"I={c.get('innovation',0)} reward={sc.get('reward',0):.1f} A={advantages[i]:+.2f}σ")

            best_in_round = scored[0]
            best_sc = best_in_round.get("scores", {})
            best_reward = best_sc.get("reward", 0)
            prev_best = state.get("best_reward", 0)
            cumulative_best = max(prev_best, best_reward)

            if best_reward > prev_best:
                global_best_plan = best_in_round["plan"]
                global_best_scores = best_sc["combined"]
                global_best_reward = best_reward
            else:
                global_best_plan = state.get("best_plan", best_in_round.get("plan", ""))
                global_best_scores = state.get("best_scores", best_sc.get("combined", {}))
                global_best_reward = prev_best

            round_history = list(state.get("round_history", []))
            round_history.append({
                "round": round_num, "group_size": len(scored),
                "mean_reward": mean_r, "best_reward": best_reward,
                "cumulative_best": cumulative_best, "top_plan_id": best_in_round["plan_id"],
            })

            return {
                "group_plans": scored, "round_history": round_history,
                "cumulative_best_reward": cumulative_best,
                "best_plan": global_best_plan, "best_scores": global_best_scores, "best_reward": global_best_reward,
            }

        def select_and_mutate(state: GSPOState) -> Dict:
            round_num = state.get("round", 1)
            group_plans = state.get("group_plans", [])
            topic = state["research_topic"]

            elite = group_plans[:self.num_elite]
            print(f"\n{'='*60}")
            print(f"[GSPO] Elite + Mutation (Round {round_num}→{round_num+1}) — {len(elite)}/{len(group_plans)} survive")
            print(f"{'='*60}")

            summaries = []
            for sp in group_plans:
                sc = sp.get("scores", {})
                c = sc.get("combined", {})
                summaries.append(
                    f"Plan {sp['plan_id']} (rank {sp.get('rank','?')}): "
                    f"R={c.get('reliability',0)} I={c.get('innovation',0)} reward={sc.get('reward',0):.1f}"
                )

            print("  [ACTOR] Generating mutation directions...")
            selection_raw = call_llm_by_role(
                "actor",
                "You are directing sequential policy optimization via elite mutation.",
                ELITE_SELECTION_PROMPT.format(
                    question=topic, round=round_num,
                    cumulative_best=state.get("cumulative_best_reward", 0),
                    scored_summaries="\n".join(summaries),
                    elite_ratio=self.elite_ratio, num_elite=self.num_elite,
                ),
                temperature=0.3,
            )

            mutation_guides = []
            try:
                json_match = re.search(r"\[.*\]", selection_raw, re.DOTALL)
                if json_match:
                    mutation_guides = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
            while len(mutation_guides) < self.num_elite:
                mutation_guides.append({
                    "plan_id": elite[len(mutation_guides) % len(elite)]["plan_id"] if elite else "?",
                    "mutation_direction": MUTATION_DIRECTIONS[len(mutation_guides) % len(MUTATION_DIRECTIONS)],
                    "temperature": 0.2 + len(mutation_guides) * 0.1,
                })

            new_group = []
            for i, elite_plan in enumerate(elite):
                guide = mutation_guides[i] if i < len(mutation_guides) else {}
                temp = min(0.5, guide.get("temperature", GSPO_MUTATION_STRENGTH))
                direction = guide.get("mutation_direction", MUTATION_DIRECTIONS[i % len(MUTATION_DIRECTIONS)])
                sc = elite_plan.get("scores", {}).get("combined", {})

                mutated = call_llm_by_role(
                    "actor",
                    f"Mutating elite plan {elite_plan['plan_id']}.",
                    GSPO_MUTATION_PROMPT.format(
                        round=round_num, rank=i + 1, total_elite=self.num_elite,
                        elite_plan=elite_plan["plan"][:5000],
                        reliability=sc.get("reliability", 0), innovation=sc.get("innovation", 0),
                        advantage=f"{elite_plan.get('advantage', 0):+.2f}σ",
                        temperature=temp, mutation_direction=direction,
                    ),
                    temperature=temp,
                )
                new_group.append({
                    "plan_id": f"{elite_plan['plan_id']}{round_num+1}",
                    "parent_id": elite_plan["plan_id"],
                    "core_idea": f"Mutation: {direction[:60]}",
                    "plan": mutated,
                })

            if len(new_group) < self.group_size:
                extra = call_llm_by_role(
                    "actor",
                    "Generate a novel experimental plan completely different from existing ones.",
                    GENERATION_PROMPT.format(
                        question=topic,
                        context=f"Existing: {', '.join(p['plan_id'] for p in new_group)}. Generate something different."
                    ),
                    temperature=0.7,
                )
                new_group.append({"plan_id": f"NEW-{round_num+1}", "core_idea": "Exploratory new direction", "plan": extra})

            print(f"  → New group: {len(new_group)} plans for round {round_num+1}")
            return {"round": round_num + 1, "group_plans": new_group, "elite_plans": elite}

        def finalize(state: GSPOState) -> Dict:
            print(f"\n{'='*60}")
            print(f"[GSPO] Final Report")
            print(f"{'='*60}")

            topic = state["research_topic"]
            best_plan = state.get("best_plan", "")
            best_scores = state.get("best_scores", {})
            best_reward = state.get("best_reward", 0)
            round_history = state.get("round_history", [])

            display = best_plan[:5000] + "\n\n...(truncated)" if len(best_plan) > 5000 else best_plan

            header = self._build_final_report_header(topic, extra_config=[
                f"  Group size K: {self.group_size}",
                f"  Rounds: {self.n_rounds}",
                f"  Elite ratio: {self.elite_ratio} ({self.num_elite} plans survive/round)",
                f"  Strategy: Sequential group sampling + elitism + directed mutation",
                f"  Actor: {MODEL_ROLES['ACTOR']['model']}",
            ])

            lines = header + [
                "",
                "-" * 70,
                "Round-by-Round Evolution",
                "-" * 70,
            ]

            for rh in round_history:
                lines.append(
                    f"  Round {rh['round']}: best_r={rh['best_reward']:.1f}, "
                    f"cum_best={rh['cumulative_best']:.1f}, μ_r={rh['mean_reward']:.1f}, top={rh['top_plan_id']}"
                )

            if len(round_history) >= 2:
                first_best = round_history[0]["best_reward"]
                last_best = round_history[-1]["best_reward"]
                improvement = last_best - first_best
                lines.append(f"\n  Sequential improvement: {first_best:.1f} → {last_best:.1f} (Δ={improvement:+.1f})")

            lines.append(f"\n  Total scoring events: {self.tracker.count()}")
            lines.append(self._build_six_score_section())

            lines += [
                "",
                "-" * 70,
                "Final Best Plan (after sequential evolutionary optimization)",
                "-" * 70,
                "",
                display,
                "",
                "=" * 70,
            ]

            return {"final_report": "\n".join(lines)}

        def route_after_score(state: GSPOState) -> Literal["mutate", "finalize"]:
            round_num = state.get("round", 1)
            if round_num >= self.n_rounds:
                print(f"  → All {self.n_rounds} rounds complete, finalizing")
                return "finalize"
            if state.get("best_reward", 0) >= 88:
                print(f"  → Exceptional reward, early stop")
                return "finalize"
            return "mutate"

        workflow.add_node("init", init_group)
        workflow.add_node("score", score_group)
        workflow.add_node("mutate", select_and_mutate)
        workflow.add_node("finalize", finalize)

        workflow.set_entry_point("init")
        workflow.add_edge("init", "score")
        workflow.add_conditional_edges("score", route_after_score, {"mutate": "mutate", "finalize": "finalize"})
        workflow.add_edge("mutate", "score")
        workflow.add_edge("finalize", END)

        return workflow

    def run(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        self.reset()
        workflow = self.build_graph()
        app = workflow.compile(checkpointer=MemorySaver())

        initial: GSPOState = {
            "research_topic": question, "round": 0,
            "group_plans": [], "elite_plans": [], "round_history": [],
            "cumulative_best_reward": 0.0,
            "best_plan": "", "best_scores": {}, "best_reward": 0.0,
            "final_report": "",
        }

        config = {"configurable": {"thread_id": f"gspo-{int(time.time())}"}}

        print(f"\n{'*'*70}")
        print(f"GSPO: {question[:200]}")
        print(f"  K={self.group_size}, rounds={self.n_rounds}, elite_ratio={self.elite_ratio}")
        print(f"  Actor={MODEL_ROLES['ACTOR']['model']}")
        print(f"{'*'*70}")

        final_state = None
        for event in app.stream(initial, config, stream_mode="values"):
            final_state = event

        if final_state is None:
            return {"final_report": "[ERROR]", "best_plan": "", "best_scores": {}, "best_reward": 0,
                    "six_scores": {}, "strategy": "GSPO"}

        return {
            "final_report": final_state.get("final_report", ""),
            "best_plan": final_state.get("best_plan", ""),
            "best_scores": final_state.get("best_scores", {}),
            "best_reward": final_state.get("best_reward", 0),
            "round_history": final_state.get("round_history", []),
            "rounds_completed": final_state.get("round", 0),
            "six_scores": self.tracker.get_six_scores(),
            "score_entries": self.tracker.entries,
            "strategy": "GSPO",
        }

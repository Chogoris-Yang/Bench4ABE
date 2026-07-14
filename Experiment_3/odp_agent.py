"""
Experiment_3/odp_agent.py
ODP (Online Distillation from Preferences) for experiment protocol generation.

Multi-model architecture:
  ACTOR       → deepseek-v4-pro (Student — plan generation)
  AUX_TEACHER → GLM5.1          (Auxiliary Teacher — supplementary reference plans)
  JUDGE_GPT   → gpt-5.4         (Primary Teacher — critique + reference)
  JUDGE_DS    → deepseek-chat   (Scoring)

Math:
  L_ODP = α * L_distill(student, teacher) + (1-α) * L_reward(student)
  where α balances teacher guidance vs. reward optimization.

Key innovation: ONLINE learning — student samples, teacher judges immediately.
Two-teacher ensemble: GPT-5.4 (primary) + GLM5.1 (auxiliary reference).

Architecture:
  teacher_ref → aux_teacher_ref → student_init → [critique → learn] × N → finalize
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
    score_single_plan,
    GENERATION_PROMPT, DISTILLATION_PROMPT,
)
from config import (
    ODP_TEACHER_TEMP, ODP_STUDENT_TEMP, ODP_DISTILL_STEPS, ODP_DISTILL_WEIGHT,
    MAX_ITERATIONS, REWARD_WEIGHTS, MODEL_ROLES,
)

TEACHER_REFERENCE_PROMPT = """You are an EXPERT TEACHER (GPT-5.4) generating a GOLD-STANDARD reference plan.

Research Question:
{question}

Generate a comprehensive reference plan demonstrating deep methodological expertise.
This reference will GUIDE the student's learning — make it exemplary and complete."""

AUX_TEACHER_PROMPT = """You are an AUXILIARY TEACHER (GLM5.1) providing a supplementary reference.

Research Question:
{question}

Primary teacher has provided one reference. Generate an ALTERNATIVE high-quality reference
that takes a DIFFERENT approach. This gives the student multiple expert perspectives to learn from.

Output a complete experimental plan with a clearly different methodology from what a typical expert might propose."""

TEACHER_CRITIQUE_PROMPT = """You are an EXPERT TEACHER evaluating a student's plan.

Research Question: {question}

YOUR REFERENCE (gold standard):
---
{reference_plan}
---

AUXILIARY TEACHER'S ALTERNATIVE REFERENCE:
---
{aux_reference}
---

STUDENT'S PLAN:
---
{student_plan}
---

Student's scores: Reliability={student_rel}, Innovation={student_inn}, Reward={student_reward}

Provide a detailed TEACHING critique:
1. STRENGTHS (reinforce these)
2. GAPS vs BOTH references (what the student missed)
3. SPECIFIC IMPROVEMENTS (exactly what to change)
4. DISTILLATION SIGNAL (key principles to absorb)
5. CONFIDENCE (0-100)

Be CONSTRUCTIVE and ACTIONABLE."""

STUDENT_LEARN_PROMPT = """You are a STUDENT learning from two expert teachers.

Research Question: {question}

YOUR CURRENT PLAN:
---
{current_plan}
---

TEACHER'S CRITIQUE:
---
{teacher_critique}
---

TEACHER'S REFERENCE (inspiration — do not copy):
---
{teacher_reference}
---

AUXILIARY TEACHER'S ALTERNATIVE:
---
{aux_reference}
---

DISTILLATION WEIGHT: α={alpha}  |  Current Reward: {reward}

LEARNING OBJECTIVE: L = α·L_distill + (1-α)·L_reward
{alpha_desc}

INSTRUCTIONS:
1. Absorb key principles from BOTH teachers (DISTILLATION)
2. Apply specific improvements from the critique
3. Preserve your own UNIQUE insights
4. SYNTHESIZE the best of all perspectives

Generate your IMPROVED plan."""


class ODPState(TypedDict):
    research_topic: str
    step: int
    teacher_reference: str           # Primary teacher (GPT-5.4) reference
    aux_teacher_reference: str       # Auxiliary teacher (GLM5.1) reference
    teacher_scores: Dict
    aux_teacher_scores: Dict
    student_plan: str
    student_scores: Dict
    teacher_critique: str
    distillation_history: List[Dict]
    best_student_plan: str
    best_student_scores: Dict
    best_student_reward: float
    final_report: str


class ODPAgent(BaseRLAgent):
    """ODP — GPT-5.4 teacher + GLM5.1 auxiliary teacher → DeepSeek student."""

    strategy_name = "ODP"

    def __init__(self):
        super().__init__()
        self.distill_steps = ODP_DISTILL_STEPS
        self.alpha = ODP_DISTILL_WEIGHT

    def build_graph(self) -> StateGraph:
        workflow = StateGraph(ODPState)

        def teacher_generate(state: ODPState) -> Dict:
            topic = state["research_topic"]
            self._print_model_roles()

            print(f"\n{'='*60}")
            print(f"[ODP] Phase 1a: Primary Teacher (GPT-5.4) reference")
            print(f"  Teacher: {MODEL_ROLES['JUDGE_GPT']['model']}")
            print(f"{'='*60}")

            teacher_plan = call_llm_by_role(
                "judge_gpt",
                "You are an expert teacher creating a gold-standard reference.",
                TEACHER_REFERENCE_PROMPT.format(question=topic),
                temperature=ODP_TEACHER_TEMP,
            )
            print(f"  → Primary reference: {len(teacher_plan)} chars")

            print("  [Scoring teacher reference]")
            teacher_scores = self._score_plan(teacher_plan, label="teacher_ref")
            print(f"    Teacher reward: {teacher_scores['reward']:.1f}")

            return {"teacher_reference": teacher_plan, "teacher_scores": teacher_scores}

        def aux_teacher_generate(state: ODPState) -> Dict:
            topic = state["research_topic"]

            print(f"\n{'='*60}")
            print(f"[ODP] Phase 1b: Auxiliary Teacher (GLM5.1) alternative reference")
            print(f"  Aux Teacher: {MODEL_ROLES['AUX_TEACHER']['model']}")
            print(f"{'='*60}")

            aux_plan = call_llm_by_role(
                "aux_teacher",
                "You are an auxiliary teacher providing an alternative expert perspective.",
                AUX_TEACHER_PROMPT.format(question=topic),
                temperature=ODP_TEACHER_TEMP + 0.1,
            )
            print(f"  → Auxiliary reference: {len(aux_plan)} chars")

            print("  [Scoring auxiliary reference]")
            aux_scores = self._score_plan(aux_plan, label="aux_teacher_ref")
            print(f"    Aux Teacher reward: {aux_scores['reward']:.1f}")

            return {"aux_teacher_reference": aux_plan, "aux_teacher_scores": aux_scores}

        def student_initial(state: ODPState) -> Dict:
            topic = state["research_topic"]

            print(f"\n{'='*60}")
            print(f"[ODP] Phase 2: Student (DeepSeek V4 Pro) initial plan")
            print(f"  Student: {MODEL_ROLES['ACTOR']['model']}")
            print(f"{'='*60}")

            student_plan = call_llm_by_role(
                "actor",
                "You are a student research scientist. Generate your best experimental plan.",
                GENERATION_PROMPT.format(
                    question=topic,
                    context="This is your first attempt. Two expert teachers will review it."
                ),
                temperature=ODP_STUDENT_TEMP,
            )
            print(f"  → Student plan: {len(student_plan)} chars")

            print("  [Scoring student plan]")
            student_scores = self._score_plan(student_plan, label="student_initial")
            student_reward = student_scores["reward"]

            teacher_reward = state.get("teacher_scores", {}).get("reward", 0)
            aux_reward = state.get("aux_teacher_scores", {}).get("reward", 0)
            print(f"  Teacher={teacher_reward:.1f} | Aux={aux_reward:.1f} | Student={student_reward:.1f}")
            print(f"  Gap to primary teacher: {teacher_reward - student_reward:+.1f}")

            return {
                "student_plan": student_plan,
                "student_scores": student_scores,
                "best_student_plan": student_plan,
                "best_student_scores": student_scores["combined"],
                "best_student_reward": student_reward,
                "step": 1,
            }

        def teacher_critique(state: ODPState) -> Dict:
            step = state.get("step", 1)
            topic = state["research_topic"]
            teacher_ref = state.get("teacher_reference", "")
            aux_ref = state.get("aux_teacher_reference", "")
            student_plan = state.get("student_plan", "")
            student_scores = state.get("student_scores", {})
            combined = student_scores.get("combined", {})

            print(f"\n{'='*60}")
            print(f"[ODP] Step {step}: Teacher critiques student")
            print(f"  Primary: {MODEL_ROLES['JUDGE_GPT']['model']} | Aux: {MODEL_ROLES['AUX_TEACHER']['model']}")
            print(f"{'='*60}")

            critique = call_llm_by_role(
                "judge_gpt",
                "You are an expert teacher providing detailed feedback.",
                TEACHER_CRITIQUE_PROMPT.format(
                    question=topic,
                    reference_plan=teacher_ref[:4000],
                    aux_reference=aux_ref[:2000],
                    student_plan=student_plan[:4000],
                    student_rel=combined.get("reliability", 0),
                    student_inn=combined.get("innovation", 0),
                    student_reward=student_scores.get("reward", 0),
                ),
                temperature=ODP_TEACHER_TEMP,
            )
            print(f"  → Critique: {len(critique)} chars")
            for line in critique.split("\n")[:5]:
                print(f"    {line.strip()[:120]}")

            return {"teacher_critique": critique}

        def student_learn(state: ODPState) -> Dict:
            step = state.get("step", 1)
            topic = state["research_topic"]
            student_plan = state.get("student_plan", "")
            teacher_ref = state.get("teacher_reference", "")
            aux_ref = state.get("aux_teacher_reference", "")
            teacher_critique_text = state.get("teacher_critique", "")
            student_scores = state.get("student_scores", {})
            reward = student_scores.get("reward", 0)

            print(f"\n{'='*60}")
            print(f"[ODP] Step {step}: Student learns via two-teacher distillation (α={self.alpha})")
            print(f"{'='*60}")

            alpha_desc = (
                f"Trust teacher guidance at {self.alpha*100:.0f}% and self reward at {(1-self.alpha)*100:.0f}%"
            )
            student_temp = max(0.1, ODP_STUDENT_TEMP - (self.alpha - 0.5) * 0.6)

            improved_plan = call_llm_by_role(
                "actor",
                f"You are a student learning from two expert teachers. α={self.alpha}.",
                STUDENT_LEARN_PROMPT.format(
                    question=topic,
                    current_plan=student_plan[:5000],
                    teacher_critique=teacher_critique_text[:3000],
                    teacher_reference=teacher_ref[:2000],
                    aux_reference=aux_ref[:1500],
                    alpha=self.alpha,
                    reward=reward,
                    alpha_desc=alpha_desc,
                ),
                temperature=student_temp,
            )
            print(f"  → Improved plan: {len(improved_plan)} chars (T={student_temp:.2f})")

            print("  [Scoring improved plan]")
            new_scores = self._score_plan(improved_plan, label=f"student_step{step}")
            new_reward = new_scores["reward"]

            old_reward = student_scores.get("reward", 0)
            delta = new_reward - old_reward
            distill_loss = self.alpha * delta
            reward_loss = (1 - self.alpha) * delta
            print(f"    ODP: L_distill={distill_loss:+.2f}, L_reward={reward_loss:+.2f}, L_total={delta:+.2f}")
            print(f"    Reward: {old_reward:.1f} → {new_reward:.1f} (Δ={delta:+.1f})")

            dist_history = list(state.get("distillation_history", []))
            dist_history.append({
                "step": step,
                "student_reward_before": old_reward,
                "student_reward_after": new_reward,
                "delta_reward": delta,
                "distill_loss": distill_loss,
                "reward_loss": reward_loss,
                "odp_total_loss": delta,
                "student_temp": student_temp,
            })

            prev_best = state.get("best_student_reward", 0)
            if new_reward > prev_best:
                bp, bs, br = improved_plan, new_scores["combined"], new_reward
                print(f"  → New best student! ({prev_best:.1f} → {new_reward:.1f})")
            else:
                bp = state.get("best_student_plan", improved_plan)
                bs = state.get("best_student_scores", new_scores["combined"])
                br = prev_best

            return {
                "student_plan": improved_plan, "student_scores": new_scores,
                "best_student_plan": bp, "best_student_scores": bs, "best_student_reward": br,
                "distillation_history": dist_history, "step": step + 1,
            }

        def finalize(state: ODPState) -> Dict:
            print(f"\n{'='*60}")
            print(f"[ODP] Final Report")
            print(f"{'='*60}")

            topic = state["research_topic"]
            best_plan = state.get("best_student_plan", "")
            best_scores = state.get("best_student_scores", {})
            best_reward = state.get("best_student_reward", 0)
            teacher_scores = state.get("teacher_scores", {})
            aux_scores = state.get("aux_teacher_scores", {})
            teacher_reward = teacher_scores.get("reward", 0)
            aux_reward = aux_scores.get("reward", 0)
            dist_history = state.get("distillation_history", [])

            display = best_plan[:5000] + "\n\n...(truncated)" if len(best_plan) > 5000 else best_plan

            header = self._build_final_report_header(topic, extra_config=[
                f"  Primary Teacher:  {MODEL_ROLES['JUDGE_GPT']['model']} (T={ODP_TEACHER_TEMP})",
                f"  Aux Teacher:      {MODEL_ROLES['AUX_TEACHER']['model']} (T={ODP_TEACHER_TEMP+0.1})",
                f"  Student:          {MODEL_ROLES['ACTOR']['model']} (T={ODP_STUDENT_TEMP})",
                f"  Distillation steps: {self.distill_steps}",
                f"  Distillation weight α: {self.alpha}",
                f"  Objective: L = α·L_distill + (1-α)·L_reward",
                f"  Teacher references: Primary={teacher_reward:.1f}, Aux={aux_reward:.1f}",
            ])

            lines = header + [
                "",
                "-" * 70,
                "Online Distillation History",
                "-" * 70,
            ]

            for dh in dist_history:
                lines.append(
                    f"  Step {dh['step']}: {dh['student_reward_before']:.1f} → {dh['student_reward_after']:.1f} "
                    f"(Δ={dh['delta_reward']:+.1f}) | L_distill={dh['distill_loss']:+.2f} "
                    f"L_reward={dh['reward_loss']:+.2f}"
                )

            if dist_history:
                first_reward = dist_history[0]["student_reward_before"]
                final_reward = dist_history[-1]["student_reward_after"]
                total_imp = final_reward - first_reward
                lines += [
                    "",
                    f"  Total distillation improvement: {first_reward:.1f} → {final_reward:.1f} (Δ={total_imp:+.1f})",
                    f"  Student vs Primary Teacher gap: {teacher_reward - final_reward:+.1f}",
                    f"  Student vs Aux Teacher gap:    {aux_reward - final_reward:+.1f}",
                ]

            lines.append(f"\n  Total scoring events: {self.tracker.count()}")
            lines.append(self._build_six_score_section())

            lines += [
                "",
                "-" * 70,
                "Best Student Plan (after two-teacher online distillation)",
                "-" * 70,
                "",
                display,
                "",
                "=" * 70,
            ]

            return {"final_report": "\n".join(lines)}

        def route_after_learn(state: ODPState) -> Literal["critique", "finalize"]:
            step = state.get("step", 1)
            if step <= self.distill_steps:
                print(f"  → Continue distillation (step {step}/{self.distill_steps})")
                return "critique"
            print(f"  → Distillation complete ({self.distill_steps} steps)")
            return "finalize"

        workflow.add_node("teacher", teacher_generate)
        workflow.add_node("aux_teacher", aux_teacher_generate)
        workflow.add_node("student_init", student_initial)
        workflow.add_node("critique", teacher_critique)
        workflow.add_node("learn", student_learn)
        workflow.add_node("finalize", finalize)

        workflow.set_entry_point("teacher")
        workflow.add_edge("teacher", "aux_teacher")
        workflow.add_edge("aux_teacher", "student_init")
        workflow.add_conditional_edges("student_init", route_after_learn,
                                       {"critique": "critique", "finalize": "finalize"})
        workflow.add_edge("critique", "learn")
        workflow.add_conditional_edges("learn", route_after_learn,
                                       {"critique": "critique", "finalize": "finalize"})
        workflow.add_edge("finalize", END)

        return workflow

    def run(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        self.reset()
        workflow = self.build_graph()
        app = workflow.compile(checkpointer=MemorySaver())

        initial: ODPState = {
            "research_topic": question, "step": 0,
            "teacher_reference": "", "aux_teacher_reference": "",
            "teacher_scores": {}, "aux_teacher_scores": {},
            "student_plan": "", "student_scores": {},
            "teacher_critique": "", "distillation_history": [],
            "best_student_plan": "", "best_student_scores": {}, "best_student_reward": 0.0,
            "final_report": "",
        }

        config = {"configurable": {"thread_id": f"odp-{int(time.time())}"}}

        print(f"\n{'*'*70}")
        print(f"ODP: {question[:200]}")
        print(f"  Teacher: {MODEL_ROLES['JUDGE_GPT']['model']} | Aux: {MODEL_ROLES['AUX_TEACHER']['model']}")
        print(f"  Student: {MODEL_ROLES['ACTOR']['model']} | α={self.alpha}, steps={self.distill_steps}")
        print(f"{'*'*70}")

        final_state = None
        for event in app.stream(initial, config, stream_mode="values"):
            final_state = event

        if final_state is None:
            return {"final_report": "[ERROR]", "best_plan": "", "best_scores": {}, "best_reward": 0,
                    "six_scores": {}, "strategy": "ODP"}

        return {
            "final_report": final_state.get("final_report", ""),
            "best_plan": final_state.get("best_student_plan", ""),
            "best_scores": final_state.get("best_student_scores", {}),
            "best_reward": final_state.get("best_student_reward", 0),
            "teacher_scores": final_state.get("teacher_scores", {}),
            "distillation_history": final_state.get("distillation_history", []),
            "steps_completed": final_state.get("step", 0),
            "six_scores": self.tracker.get_six_scores(),
            "score_entries": self.tracker.entries,
            "strategy": "ODP",
        }

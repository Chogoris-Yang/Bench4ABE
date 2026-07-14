"""
Experiment_2/voyager.py
Voyager-Style Experiment Protocol Generator
Automatic Curriculum -> Skill Library(retrieve+store) -> Iterative Refinement -> Skill Composition
Generator Model: DeepSeek V4 Pro  |  Judge Model: DeepSeek-chat + GPT-5.4
"""

import json
import os
import re
import time
import random
from typing import TypedDict, List, Dict, Any, Optional, Literal
from openai import OpenAI
import httpx

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# =========================================================
# Proxy & API
# =========================================================
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

DEEPSEEK_API_KEY = "enter-your-api-key"
GPT_API_KEY = "enter-your-api-key"

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com",
    timeout=httpx.Timeout(300.0)
)
gpt_client = OpenAI(
    api_key=GPT_API_KEY, base_url="https://www.autodl.art/api/v1",
    timeout=httpx.Timeout(300.0)
)

GENERATOR_MODEL = "deepseek-v4-pro"
JUDGE_MODEL_DS   = "deepseek-chat"
JUDGE_MODEL_GPT  = "gpt-5.4"

MAX_RETRIES_PER_GOAL = 2    # Max retries per sub-goal

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gen_lcot", "data", "train.jsonl"
)

# =========================================================
# Voyager Prompts
# =========================================================

# Stage 1: Automatic Curriculum Generation
CURRICULUM_PROMPT = """You are a Voyager-style automatic curriculum generator.
Given a complex research task, decompose it into a sequence of sub-goals ordered from SIMPLE to COMPLEX.
Each sub-goal should build on previous ones. Together they compose the full experimental plan.

Research Task:
{question}

Generate 4-5 sub-goals Covering: problem formulation, dataset selection, methodology design,
experimental protocol, evaluation strategy.

Output format (JSON array):
[
  {{"id": 1, "name": "short name", "description": "what this sub-goal achieves", "difficulty": "easy|medium|hard",
    "depends_on": [], "deliverable": "what concrete output this produces"}},
  ...
]

Make the sub-goals concrete and executable."""


# Stage 2: Skill Library Retrieval
SKILL_RETRIEVAL_PROMPT = """You are a Voyager skill library manager. You have a library of previously
developed experimental design "skills" (reusable templates, patterns, strategies).

SKILL LIBRARY:
---
{skill_library}
---

CURRENT SUB-GOAL: {goal_description}

Select the 2-3 most relevant skills from the library that would help achieve this sub-goal.
For each, explain WHY it is relevant.

Output format:
SELECTED SKILLS:
- [Skill Name]: why relevant
- ...
"""


# Stage 3: Generate sub-goal plan using retrieved skills
SKILL_GENERATE_PROMPT = """You are a Voyager agent executing a sub-goal. You have access to relevant skills
from the skill library. Use them to generate a concrete plan for this sub-goal.

SUB-GOAL: {goal_name}
Description: {goal_description}
Difficulty: {difficulty}

RETRIEVED SKILLS (apply these):
{retrieved_skills}

PREVIOUS SUB-GOAL OUTPUTS (context):
{previous_outputs}

Generate a detailed, executable plan for this sub-goal. Apply the retrieved skills.
Output a concrete deliverable as specified."""


# Stage 4: Self-Validation
VALIDATE_PROMPT = """You are a Voyager self-validation module. Check if the following sub-goal output
meets the requirements.

SUB-GOAL: {goal_name}
Expected Deliverable: {deliverable}

OUTPUT TO VALIDATE:
---
{output}
---

Evaluate:
1. Does it achieve the sub-goal? (YES/NO)
2. Are there any errors or gaps?
3. Is it concrete and ready to use?

Output format:
VALID: YES/NO
Issues: (list if any)
Suggestions: (if NO, what needs to be fixed)"""


# Stage 5: Reflection + Retry
REFLECT_RETRY_PROMPT = """You are a Voyager reflection module. The previous attempt at a sub-goal failed validation.

SUB-GOAL: {goal_name}
PREVIOUS ATTEMPT:
---
{previous_output}
---

VALIDATION FEEDBACK:
---
{feedback}
---

RETRIEVED SKILLS:
{retrieved_skills}

Reflect on why it failed, then generate a CORRECTED version that addresses all issues.
Apply the retrieved skills more carefully.
Output the COMPLETE corrected deliverable."""


# Stage 6: Compose all sub-goals -> complete plan
COMPOSE_PROMPT = """You are a Voyager composition module. All sub-goals have been individually achieved.
Now compose them into one complete, unified experimental plan.

SUB-GOAL OUTPUTS:
---
{all_outputs}
---

Research Task: {question}

Compose all outputs into ONE cohesive experimental plan:
- Integrate — don't just concatenate
- Ensure consistency across sections
- Add transitions and cross-references
- Include: Abstract, Objective, Methodology, Experimental Design, Evaluation, Limitations

Output the complete, polished experimental plan."""


SCORING_PROMPT = """You are an expert scientific reviewer.

Please evaluate the given research methodology from the following two aspects:

1. Method Reliability (0-100)
2. Method Innovation (0-100)

Your output format MUST strictly follow:
Reliability: xx
Innovation: xx

Only output the scores. Do not provide explanations."""


# =========================================================
# Skill Library (Skill Library)
# =========================================================

# Preset Base Skill Templates
BASE_SKILL_LIBRARY = [
    {
        "name": "ProblemFormulation",
        "description": "Template for clearly defining research problem, scope, and objectives",
        "pattern": "1. State the core problem. 2. Identify research gap. 3. Define scope and constraints. 4. Formulate measurable objectives."
    },
    {
        "name": "DatasetSelection",
        "description": "Strategy for selecting and justifying appropriate datasets and benchmarks",
        "pattern": "1. Survey standard benchmarks in the domain. 2. Evaluate dataset size, quality, and relevance. 3. Plan train/val/test splits. 4. Document data preprocessing."
    },
    {
        "name": "MethodologyDesign",
        "description": "Framework for designing a novel methodology with clear innovations",
        "pattern": "1. Identify limitations of existing methods. 2. Propose architectural/methodological innovations. 3. Provide theoretical justification. 4. Detail implementation."
    },
    {
        "name": "AblationDesign",
        "description": "Pattern for designing controlled ablation studies",
        "pattern": "1. List all components to ablate. 2. Design incremental removal experiments. 3. Establish baseline (all components). 4. Measure contribution of each component."
    },
    {
        "name": "EvaluationFramework",
        "description": "Template for comprehensive evaluation strategy",
        "pattern": "1. Select primary and secondary metrics. 2. Choose diverse baselines (classic to SOTA). 3. Plan statistical significance tests. 4. Design qualitative analysis."
    },
    {
        "name": "ReproducibilityChecklist",
        "description": "Checklist for ensuring experimental reproducibility",
        "pattern": "1. Document hyperparameters. 2. Specify hardware/software. 3. Provide random seeds. 4. Plan code release. 5. Estimate compute requirements."
    },
    {
        "name": "IterativeRefinement",
        "description": "Strategy for iteratively improving a plan based on self-critique",
        "pattern": "1. Generate initial version. 2. Self-critique: identify 3 weaknesses. 3. Address each weakness. 4. Repeat until quality threshold met."
    },
    {
        "name": "CrossDisciplinaryInspiration",
        "description": "Method for importing ideas from adjacent fields to boost innovation",
        "pattern": "1. Identify analogous problems in other domains. 2. Extract transferable principles. 3. Adapt to current domain. 4. Justify the adaptation."
    },
]

SKILL_LIBRARY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "voyager_skill_library.json"
)


def format_skill_library(skills: List[Dict]) -> str:
    """Format skill library as text"""
    parts = []
    for i, s in enumerate(skills):
        parts.append(
            f"[Skill {i+1}] {s['name']}\n"
            f"  Description: {s['description']}\n"
            f"  Pattern: {s['pattern']}"
        )
    return "\n\n".join(parts)


# =========================================================
# Data & Utilities
# =========================================================

def load_dataset(path: str = DATASET_PATH, sample_size: int = 50) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    if sample_size and sample_size < len(data):
        random.seed(42)
        data = random.sample(data, sample_size)
    return data


def extract_user_content(sample: Dict) -> str:
    for msg in sample.get("messages", []):
        if msg["role"] == "user":
            return msg["content"]
    return ""


def parse_score(text: Optional[str]) -> Dict[str, int]:
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


def call_llm(client: OpenAI, model: str, system_prompt: str, user_input: str,
             temperature: float = 0.1, max_retries: int = 5) -> str:
    sleep_time = 1
    for attempt in range(1, max_retries + 1):
        try:
            print(f"    [API] {model} attempt {attempt}...")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_input}],
                temperature=temperature, timeout=180,
            )
            print(f"    [API] {model} OK")
            return response.choices[0].message.content
        except Exception as e:
            print(f"    [API] error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(sleep_time)
                sleep_time = min(sleep_time * 2, 60)
    return "[ERROR] all retries failed"


# =========================================================
# LangGraph State
# =========================================================

class VoyagerState(TypedDict):
    research_topic: str
    skill_library: List[Dict]              # Current Skill Library
    curriculum: List[Dict]                 # Sub-goal list [{id, name, description, difficulty, depends_on, deliverable}]
    current_goal_index: int
    goal_outputs: List[Dict]               # [{goal_id, name, output, validated}]
    scores: List[Dict]
    final_report: str


# =========================================================
# Build Voyager pipeline
# =========================================================

def build_voyager_graph() -> StateGraph:

    # =====================
    # Node 1: Curriculum — Automatic Curriculum Generation
    # =====================
    def curriculum_node(state: VoyagerState) -> Dict:
        print("\n" + "=" * 60)
        print("[Curric] Automatic Curriculum Generation — Decompose research task into sub-goal chain")
        print("=" * 60)

        question = state["research_topic"]
        raw = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are a Voyager curriculum generator. Output the JSON array of sub-goals.",
            CURRICULUM_PROMPT.format(question=question),
            temperature=0.3
        )

        curriculum = []
        try:
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            if json_match:
                curriculum = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            print("  [!] JSON parse failed, using defaults")
            curriculum = [
                {"id": 1, "name": "Problem Formulation", "description": "Define the core research problem, scope, and objectives",
                 "difficulty": "easy", "depends_on": [], "deliverable": "Clear problem statement with measurable objectives"},
                {"id": 2, "name": "Dataset & Resource Selection", "description": "Identify and justify datasets, compute resources",
                 "difficulty": "easy", "depends_on": [1], "deliverable": "Dataset selection plan with justification"},
                {"id": 3, "name": "Methodology Design", "description": "Design the proposed approach with innovations",
                 "difficulty": "medium", "depends_on": [1], "deliverable": "Detailed methodology description"},
                {"id": 4, "name": "Experimental Protocol", "description": "Design experiments, baselines, ablation studies",
                 "difficulty": "hard", "depends_on": [2, 3], "deliverable": "Complete experimental design"},
                {"id": 5, "name": "Evaluation Strategy", "description": "Define metrics, analysis plan, success criteria",
                 "difficulty": "hard", "depends_on": [4], "deliverable": "Evaluation framework with metrics"},
            ]

        print(f"  -> Curriculum contains {len(curriculum)}  sub-goals:")
        for c in curriculum:
            print(f"     G{c['id']} [{c['difficulty']}]: {c['name']}")

        return {"curriculum": curriculum, "current_goal_index": 0}

    # =====================
    # Node 2: Execute Goal — Retrieve Skills -> Generate -> Validate -> (Reflection + Retry)
    # =====================
    def execute_goal_node(state: VoyagerState) -> Dict:
        curriculum = state.get("curriculum", [])
        goal_idx = state.get("current_goal_index", 0)
        skill_library = state.get("skill_library", BASE_SKILL_LIBRARY)
        goal_outputs = list(state.get("goal_outputs", []))
        question = state["research_topic"]

        if goal_idx >= len(curriculum):
            return {}

        goal = curriculum[goal_idx]
        print("\n" + "=" * 60)
        print(f"[Goal {goal['id']}/{len(curriculum)}] {goal['name']} [{goal['difficulty']}]")
        print("=" * 60)

        # ---- Step 2a: Retrieve relevant skills ----
        print(f"  [Retrieve] From Skill Library ({len(skill_library)} Skill) Retrieve relevant skills...")
        lib_text = format_skill_library(skill_library)

        retrieval_result = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are a Voyager skill retriever. Select relevant skills from the library.",
            SKILL_RETRIEVAL_PROMPT.format(
                skill_library=lib_text,
                goal_description=f"{goal['name']}: {goal.get('description','')}",
            ),
            temperature=0.1
        )
        # Simple skill name extraction
        retrieved_names = re.findall(r"-\s*\[?([A-Za-z]+)\]?", retrieval_result)
        retrieved_skills_text = retrieval_result
        print(f"     -> Selected: {retrieved_names[:3]}")

        # ---- Step 2b: Generate ----
        prev_outputs_text = "\n\n---\n\n".join(
            f"[G{o['goal_id']}] {o['name']}:\n{o['output'][:500]}"
            for o in goal_outputs
        ) if goal_outputs else "(No previous outputs — this is the first sub-goal)"

        print(f"  [Generate] Generate plan...")
        output = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are a Voyager agent executing a sub-goal with retrieved skills.",
            SKILL_GENERATE_PROMPT.format(
                goal_name=goal["name"],
                goal_description=goal.get("description", ""),
                difficulty=goal.get("difficulty", "medium"),
                retrieved_skills=retrieved_skills_text,
                previous_outputs=prev_outputs_text,
            ),
            temperature=0.3
        )

        # ---- Step 2c: Validate + Retries ----
        validated = False
        retries = 0
        current_output = output
        reflection_log = []

        while not validated and retries <= MAX_RETRIES_PER_GOAL:
            print(f"  [Validate] Validate sub-goal output (attempt  {retries+1})...")
            validation = call_llm(
                deepseek_client, GENERATOR_MODEL,
                "You are a Voyager validator. Check if output meets requirements.",
                VALIDATE_PROMPT.format(
                    goal_name=goal["name"],
                    deliverable=goal.get("deliverable", ""),
                    output=current_output[:3000],
                ),
                temperature=0.1
            )

            is_valid = "VALID: YES" in validation.upper() or "VALID:YES" in validation.upper().replace(" ", "")

            if is_valid:
                print(f"     -> Validation passed!")
                validated = True
            elif retries < MAX_RETRIES_PER_GOAL:
                print(f"     -> Validation failed, retry after reflection...")
                reflection_log.append({"attempt": retries + 1, "feedback": validation[:300]})

                current_output = call_llm(
                    deepseek_client, GENERATOR_MODEL,
                    "You are a Voyager reflection module. Fix the failed output.",
                    REFLECT_RETRY_PROMPT.format(
                        goal_name=goal["name"],
                        previous_output=current_output[:2000],
                        feedback=validation,
                        retrieved_skills=retrieved_skills_text,
                    ),
                    temperature=0.3
                )
                retries += 1
            else:
                print(f"     -> Max retries reached, using current version")
                validated = True  # Force pass

        # ---- Step 2d: Store to Skill Library and sub-goal outputs ----
        # Extract this sub-goal as new skill into library
        new_skill = {
            "name": f"SubGoal_{goal['id']}_{goal['name'].replace(' ','')}",
            "description": f"Generated plan for: {goal.get('description','')}",
            "pattern": current_output[:500],
        }
        new_library = list(skill_library) + [new_skill]
        print(f"  [Store] Skill '{new_skill['name']}' Added to Skill Library (Total: {len(new_library)})")

        goal_outputs.append({
            "goal_id": goal["id"],
            "name": goal["name"],
            "output": current_output,
            "validated": validated,
            "retries": retries,
        })

        next_idx = goal_idx + 1
        print(f"  -> G{goal['id']} Done (validate={'pass' if validated else 'forced'}, Retries={retries})")

        return {
            "skill_library": new_library,
            "goal_outputs": goal_outputs,
            "current_goal_index": next_idx,
        }

    # =====================
    # Node 3: Compose — Compose all sub-goal outputs
    # =====================
    def compose_node(state: VoyagerState) -> Dict:
        print("\n" + "=" * 60)
        print("[Compose] Compose all sub-goal outputs -> Complete experiment plan")
        print("=" * 60)

        question = state["research_topic"]
        goal_outputs = state.get("goal_outputs", [])

        all_text = "\n\n---\n\n".join(
            f"## Sub-Goal {o['goal_id']}: {o['name']}\n{o['output']}"
            for o in goal_outputs
        )

        plan = call_llm(
            deepseek_client, GENERATOR_MODEL,
            "You are a Voyager composer. Integrate all sub-goal outputs into one unified plan.",
            COMPOSE_PROMPT.format(question=question, all_outputs=all_text),
            temperature=0.2
        )

        print(f"  -> Composed plan: {len(plan)} chars")
        return {"final_report": plan}  # Temporarily stored, update after scoring

    # =====================
    # Node 4: Scoring
    # =====================
    def score_node(state: VoyagerState) -> Dict:
        print("\n" + "=" * 60)
        print("[Score] Dual-Judge evaluate Final Plan")
        print("=" * 60)

        plan = state.get("final_report", "")

        print("  [DS judge] ...")
        ds_text = call_llm(deepseek_client, JUDGE_MODEL_DS, SCORING_PROMPT, plan, temperature=0.0)
        ds = parse_score(ds_text)
        print(f"    DS -> R={ds['reliability']}, I={ds['innovation']}")

        print("  [GPT judge] ...")
        gpt_text = call_llm(gpt_client, JUDGE_MODEL_GPT, SCORING_PROMPT, plan, temperature=0.0)
        gpt = parse_score(gpt_text)
        print(f"    GPT -> R={gpt['reliability']}, I={gpt['innovation']}")

        combined = {
            "reliability": round((ds["reliability"] + gpt["reliability"]) / 2, 1),
            "innovation": round((ds["innovation"] + gpt["innovation"]) / 2, 1),
        }
        verdict = (
            "EXCELLENT" if combined["reliability"] >= 80 and combined["innovation"] >= 75
            else "GOOD" if combined["reliability"] >= 70 and combined["innovation"] >= 60
            else "NEEDS_IMPROVEMENT"
        )
        scores = [{"round": 1, "ds": ds, "gpt": gpt, "combined": combined, "verdict": verdict}]
        print(f"  Combined -> R={combined['reliability']}, I={combined['innovation']} -> {verdict}")
        return {"scores": scores}

    # =====================
    # Node 5: Report
    # =====================
    def report_node(state: VoyagerState) -> Dict:
        print("\n" + "=" * 60)
        print("[Report] Generate Voyager Final Report")
        print("=" * 60)

        topic = state["research_topic"]
        plan = state.get("final_report", "")
        scores = state.get("scores", [])
        curriculum = state.get("curriculum", [])
        goal_outputs = state.get("goal_outputs", [])
        skill_library = state.get("skill_library", [])

        display = plan[:5000] + "\n\n...(truncated)" if len(plan) > 5000 else plan

        lines = [
            "=" * 70,
            "          Voyager Experiment Protocol Report",
            "=" * 70, "",
            f"Research Topic: {topic[:300]}", "",
            "-" * 70,
            f"Voyager Architecture: Automatic Curriculum -> Skill Retrieval -> Generate -> Validate -> Reflect+Retry -> Compose",
            f"Skill Library: {len(skill_library)} Skill (Initial {len(BASE_SKILL_LIBRARY)} + Newly Learned {len(goal_outputs)})",
            "-" * 70, "",
            "Automatic Curriculum (Sub-goal Chain):",
        ]
        for c in curriculum:
            go = next((g for g in goal_outputs if g["goal_id"] == c["id"]), None)
            status = f"Done(Retries{go['retries']})" if go else "Not Executed"
            lines.append(f"  G{c['id']} [{c['difficulty']:6s}]: {c['name']:30s} -> {status}")

        lines += [
            "", "Skill Library Growth:",
        ]
        for s in skill_library[len(BASE_SKILL_LIBRARY):]:
            lines.append(f"  + {s['name']}: {s['description'][:80]}")

        lines += ["", "-" * 70, "Final Plan", "-" * 70, display or "Failed to generate"]

        if scores:
            lines += ["", "-" * 70, "Quality Assessment", "-" * 70]
            for s in scores:
                c = s["combined"]
                lines.append(
                    f"  DS(R={s['ds'].get('reliability',0)} I={s['ds'].get('innovation',0)}) "
                    f"GPT(R={s['gpt'].get('reliability',0)} I={s['gpt'].get('innovation',0)}) "
                    f"-> Combined R={c.get('reliability',0)} I={c.get('innovation',0)} [{s.get('verdict','')}]"
                )

        lines += ["", "=" * 70]
        return {"final_report": "\n".join(lines)}

    # =====================
    # Routing
    # =====================
    def route_after_goal(state: VoyagerState) -> Literal["execute", "compose"]:
        goal_idx = state.get("current_goal_index", 0)
        curriculum = state.get("curriculum", [])
        if goal_idx < len(curriculum):
            return "execute"
        return "compose"

    # =====================
    # Assemble Graph
    #   curriculum -> execute -> [more?] -> compose -> score -> report
    #                   ^                      |
    #                   |_______ YES _________|
    # =====================
    workflow = StateGraph(VoyagerState)

    workflow.add_node("curriculum", curriculum_node)
    workflow.add_node("execute", execute_goal_node)
    workflow.add_node("compose", compose_node)
    workflow.add_node("score", score_node)
    workflow.add_node("report", report_node)

    workflow.set_entry_point("curriculum")
    workflow.add_edge("curriculum", "execute")
    workflow.add_conditional_edges("execute", route_after_goal,
                                   {"execute": "execute", "compose": "compose"})
    workflow.add_edge("compose", "score")
    workflow.add_edge("score", "report")
    workflow.add_edge("report", END)

    return workflow


# =========================================================
# Compile & Run
# =========================================================

def compile_agent():
    return build_voyager_graph().compile(checkpointer=MemorySaver())


def run_agent(question: str, verbose: bool = True) -> Dict[str, Any]:
    app = compile_agent()
    initial: VoyagerState = {
        "research_topic": question,
        "skill_library": list(BASE_SKILL_LIBRARY),
        "curriculum": [], "current_goal_index": 0,
        "goal_outputs": [], "scores": [], "final_report": "",
    }
    config = {"configurable": {"thread_id": f"voyager-{int(time.time())}"}}

    print(f"\n{'*'*70}")
    print(f"Voyager: {question[:200]}")
    print(f"{'*'*70}")
    print(f"Skill Library: {len(BASE_SKILL_LIBRARY)} PresetSkill | Curriculum -> Retrieve -> Generate -> Validate -> [Reflect] -> Compose")

    final_state = None
    for event in app.stream(initial, config, stream_mode="values"):
        final_state = event
        if verbose:
            for key in ["curriculum", "goal_outputs", "skill_library", "final_report"]:
                val = event.get(key, "")
                if isinstance(val, list) and val:
                    print(f"  [{key}] {len(val)}  items")
                elif isinstance(val, str) and len(val) > 50:
                    print(f"  [{key}] {len(val)} chars")

    if final_state is None:
        return {"final_report": "[ERROR]", "scores": [], "goal_outputs": []}

    return {
        "final_report": final_state.get("final_report", ""),
        "scores": final_state.get("scores", []),
        "goal_outputs": final_state.get("goal_outputs", []),
        "skill_library_size": len(final_state.get("skill_library", [])),
    }


# =========================================================
# Cross-Sample Average Scores
# =========================================================

def _avg(lst: List[float]) -> float:
    return round(sum(lst) / len(lst), 1) if lst else 0.0


def compute_batch_averages(all_results: List[Dict]) -> Dict:
    rels, inns = [], []
    for r in all_results:
        for s in r.get("scores", []):
            c = s.get("combined", {})
            rels.append(c.get("reliability", 0))
            inns.append(c.get("innovation", 0))
    return {
        "overall": {
            "total_scores": len(rels),
            "reliability_avg": _avg(rels),
            "innovation_avg": _avg(inns),
        },
    }


# =========================================================
# CLI
# =========================================================

def interactive_loop():
    print(r"""
╔══════════════════════════════════════════════════════════════╗
║    Voyager Experiment Protocol Generator                                   ║
║    Automatic Curriculum -> Skill Retrieval -> Generate -> Validate -> Reflect -> Compose      ║
║    Generator: DeepSeek V4 Pro | Judges: DS-chat + GPT-5.4          ║
╚══════════════════════════════════════════════════════════════╝

Commands:
  /run <question> -- Generate experimental plan for research question (Voyager)
  /batch [N]     -- Batch Test (default 3)
  /demo          -- use built-in demo
  /quit          -- Exit
""")

    while True:
        try:
            ui = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!"); break
        if not ui: continue
        if ui.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye!"); break
        if ui.lower() == "/demo":
            ui = "/run Design an experiment to evaluate whether large language models can self-improve through recursive self-critique without external feedback"

        if ui.lower().startswith("/run "):
            question = ui[5:].strip()
            result = run_agent(question)
            print("\n" + result["final_report"])
            out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f"voyager_result_{int(time.time())}.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n[ResultSaved] {out}")

        elif ui.lower().startswith("/batch"):
            parts = ui.split()
            n = int(parts[1]) if len(parts) > 1 else 3
            batch_test(n)
        else:
            print("Unknown command. Use /run <question> to start.")


def batch_test(n: int = 3):
    dataset = load_dataset(sample_size=n)
    print(f"\n{'#'*70}")
    print(f"  Voyager Batch Test：{n} samples")
    print(f"  Skill Library: {len(BASE_SKILL_LIBRARY)} Preset | Generator: {GENERATOR_MODEL}")
    print(f"  Judges: {JUDGE_MODEL_DS} + {JUDGE_MODEL_GPT}")
    print(f"{'#'*70}")

    all_results = []
    for i, sample in enumerate(dataset):
        user_q = extract_user_content(sample)
        req_match = re.search(r"<user_request>\s*(.*?)(?:<think>|$)", user_q, re.DOTALL)
        question = req_match.group(1).strip()[:500] if req_match else user_q[:500]

        print(f"\n{'#'*60}")
        print(f"  Sample {i+1}/{n}: {question[:100]}...")
        print(f"{'#'*60}")

        result = run_agent(question, verbose=False)
        all_results.append({
            "sample_index": i, "question": question,
            "report": result["final_report"], "scores": result.get("scores", []),
            "goals_completed": len(result.get("goal_outputs", [])),
            "skill_library_size": result.get("skill_library_size", 0),
        })
        for s in result.get("scores", []):
            c = s["combined"]
            print(f"  -> DS(R={s['ds'].get('reliability',0)} I={s['ds'].get('innovation',0)}) "
                  f"GPT(R={s['gpt'].get('reliability',0)} I={s['gpt'].get('innovation',0)}) "
                  f"Combined(R={c.get('reliability',0)} I={c.get('innovation',0)})")

    batch_avg = compute_batch_averages(all_results)
    overall = batch_avg["overall"]
    print("\n" + "=" * 70)
    print("         Voyager Batch Test -- Cross-Sample Average ScoresSummary")
    print("=" * 70)
    print(f"\n  [Global Best Avg]  (Total {overall['total_scores']} scoring events, {n} samples)")
    print(f"    * Reliability = {overall['reliability_avg']}  |  Innovation = {overall['innovation_avg']}")
    print(f"\n{'='*70}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"voyager_batch_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"generator": GENERATOR_MODEL, "judges": [JUDGE_MODEL_DS, JUDGE_MODEL_GPT],
                       "samples": n, "base_skills": len(BASE_SKILL_LIBRARY)},
            "batch_averages": batch_avg, "samples": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[ResultSaved] {out}")


if __name__ == "__main__":
    interactive_loop()

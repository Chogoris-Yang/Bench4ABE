"""
Experiment_5/stages/stage4_codegen.py — Stage 4: Code Generation.
Generates executable experiment code based on the finalized plan.
Output is saved to Experiment_5/generated_code/question_N/.
"""

import os
import re
import time
from typing import Dict

from pipeline_components import (
    call_llm, generator_client, GENERATOR_MODEL,
    dual_judge_score, parse_generic_score,
)
from config_exp5 import (
    GENERATED_CODE_DIR, TOTAL_ROUNDS,
)


CODE_GEN_PROMPT = """You are an expert bioinformatics software engineer. Based on the experimental plan
below, generate COMPLETE, RUNNABLE Python code that implements the core methodology.

EXPERIMENTAL PLAN:
---
{plan}
---

RESEARCH QUESTION: {question}

REQUIREMENTS:
1. Generate Python code that implements the core algorithm/method described in the plan
2. Include data loading, preprocessing, model architecture, training loop, and evaluation
3. Use standard libraries: numpy, pandas, scikit-learn, pytorch, etc.
4. Add docstrings and comments explaining each section
5. Include a main() function that demonstrates the full pipeline
6. Handle edge cases and add error checking
7. Make it SELF-CONTAINED (no external files needed beyond what's imported)
8. Output should be EXECUTABLE Python code

Generate ONLY the Python code. No markdown explanations."""


CODE_FIX_PROMPT = """The following Python code has issues. Fix ALL problems and output the corrected complete code.

ORIGINAL CODE:
```python
{code}
```

ISSUES TO FIX:
{issues}

Output the complete, corrected Python code. Make it runnable."""


def run_stage4(question: str, plan: str, verbose: bool = True) -> Dict:
    """
    Stage 4: Code Generation.

    1. Generate Python code implementing the experiment
    2. Validate code structure (syntax check)
    3. Retry if code has obvious issues
    4. Score code quality
    5. Save to generated_code/question_N/

    Returns:
        {code, filepath, scores, retries}
    """
    print(f"\n{'='*60}")
    print(f"[Stage 4] Code Generation")
    print(f"{'='*60}")

    code = ""
    retry_count = 0

    for attempt in range(1, TOTAL_ROUNDS + 1):
        print(f"\n  --- Code Generation Attempt {attempt}/{TOTAL_ROUNDS} ---")

        if attempt == 1:
            code = call_llm(
                generator_client, GENERATOR_MODEL,
                "You are an expert bioinformatics software engineer. Output only Python code.",
                CODE_GEN_PROMPT.format(plan=plan[:8000], question=question),
                temperature=0.2, max_tokens=8192,
            )
        else:
            # Analyze issues and fix
            issues = _analyze_code_issues(code)
            print(f"    Issues found: {issues[:200]}")
            code = call_llm(
                generator_client, GENERATOR_MODEL,
                "You fix Python code issues. Output only corrected code.",
                CODE_FIX_PROMPT.format(code=code[:6000], issues=issues),
                temperature=0.1, max_tokens=8192,
            )

        # Basic validation
        clean_code = _clean_code(code)
        is_valid, validation_msg = _validate_code(clean_code)
        print(f"    Validation: {validation_msg}")

        if is_valid:
            code = clean_code
            retry_count = attempt - 1
            break
        else:
            retry_count = attempt
            print(f"    Retrying...")

    # Clean final code
    if not code:
        code = "# [ERROR] Code generation failed after all retries\n"

    clean_code = _clean_code(code)

    # Score code quality
    print(f"\n  [Scoring] Code quality...")
    code_score_text = call_llm(
        generator_client, GENERATOR_MODEL,
        "You evaluate code quality for research experiments.",
        f"Evaluate this code for the research: {question[:200]}\n\nCODE:\n{clean_code[:4000]}\n\n"
        f"Rate Correctness (0-100) and Completeness (0-100).\nOutput: Correctness: xx\\nCompleteness: xx",
        temperature=0.0, max_tokens=128,
    )
    scores = parse_generic_score(code_score_text, "Correctness", "Completeness")
    scores["code_quality"] = round((scores.get("correctness", 0) + scores.get("completeness", 0)) / 2, 1)
    print(f"    Correctness={scores.get('correctness',0)} Completeness={scores.get('completeness',0)} → {scores['code_quality']}")

    # Save to file
    saved_path = _save_code(question, clean_code)

    return {
        "code": clean_code,
        "filepath": saved_path,
        "scores": scores,
        "code_quality": scores.get("code_quality", 0),
        "retries": retry_count,
        "validation": validation_msg if 'validation_msg' in dir() else "ok",
    }


def _clean_code(code: str) -> str:
    """Extract clean Python code from LLM output (strip markdown fences)."""
    # Remove markdown code fences
    code = re.sub(r'^```(?:python)?\s*\n?', '', code, flags=re.MULTILINE)
    code = re.sub(r'\n?```\s*$', '', code, flags=re.MULTILINE)
    # Remove leading/trailing whitespace
    code = code.strip()
    return code


def _validate_code(code: str) -> tuple:
    """Basic Python syntax validation."""
    if not code or len(code) < 100:
        return False, "Code too short (< 100 chars)"
    try:
        compile(code, "<generated>", "exec")
        return True, "Syntax OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return True, f"Compile warning (non-fatal): {e}"


def _analyze_code_issues(code: str) -> str:
    """Identify issues in generated code."""
    issues = []
    if "import" not in code:
        issues.append("No import statements found")
    if "def " not in code:
        issues.append("No function definitions found")
    if len(code) < 500:
        issues.append(f"Code too short ({len(code)} chars) - may be incomplete")
    if code.count("def ") < 2:
        issues.append("Too few functions - likely not a complete pipeline")
    if "=" * 50 in code or "-" * 50 in code:
        issues.append("Contains markdown separators - remove them")
    if "print(" not in code:
        issues.append("No output or logging statements")
    return "; ".join(issues) if issues else "Code looks OK but needs refinement"


def _save_code(question: str, code: str) -> str:
    """Save generated code to a subfolder under generated_code/."""
    # Create a safe directory name from the question
    safe_name = re.sub(r'[^a-zA-Z0-9_\-. ]', '', question[:50]).strip().replace(' ', '_')
    if not safe_name:
        safe_name = f"experiment_{int(time.time())}"

    # Number existing question dirs
    existing = [d for d in os.listdir(GENERATED_CODE_DIR) if d.startswith("question_")]
    q_num = len(existing) + 1

    dir_name = f"question_{q_num:02d}"
    dir_path = os.path.join(GENERATED_CODE_DIR, dir_name)
    os.makedirs(dir_path, exist_ok=True)

    # Save code
    code_path = os.path.join(dir_path, "experiment.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(f"# Experiment Code — Question {q_num}\n")
        f.write(f"# Task: {question[:200]}\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# {'='*60}\n\n")
        f.write(code)

    # Save metadata
    meta_path = os.path.join(dir_path, "metadata.json")
    import json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "question_num": q_num,
            "question": question[:300],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, ensure_ascii=False, indent=2)

    print(f"  [Saved] Code → {code_path}")
    return code_path

import json
from openai import OpenAI
import random
from copy import deepcopy
#from google import genai
#from anthropic import Anthropic
import re
import httpx
import os


# =========================================================
# API CONFIG
# =========================================================

SELF_API_KEY = "enter-your-api-key"
DEEPSEEK_API_KEY = "enter-your-api-key"
OPENAI_API_KEY = "enter-your-api-key"


self_client = OpenAI(
    api_key=SELF_API_KEY,
    base_url="https://api.x.ai/v1",
    timeout=httpx.Timeout(3600.0)
)

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

gpt_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://www.autodl.art/api/v1"
)

# =========================================================
# MODELS
# =========================================================

GENERATOR_MODEL = "grok-4.3"
JUDGE_MODEL_DEEPSEEK = "deepseek-chat"
JUDGE_MODEL_GPT = "gpt-5.4"

# =========================================================
# PROMPTS
# =========================================================

# Prompt 1: Scoring Prompt
SCORING_PROMPT = """
You are an expert scientific reviewer.

Please evaluate the given research methodology from the following two aspects:

1. Method Reliability (0-100)
2. Method Innovation (0-100)

Scoring Criteria:

Method Reliability:
- technical correctness
- experimental rigor
- feasibility
- logical consistency
- reproducibility

Method Innovation:
- originality
- novelty
- creativity
- research contribution
- uniqueness of methodology

Your output format MUST strictly follow:

Reliability: xx
Innovation: xx

Only output the scores.
Do not provide explanations.
"""

# Prompt 2: Experiment Plan Generation Prompt
GENERATION_PROMPT = """
You are a professional AI research scientist.

Based on the user's research objective,
generate a complete, rigorous, and executable experimental plan.

Requirements:

1. Clearly describe the research objective
2. Design the model or methodology
3. Provide detailed experimental procedures
4. Include datasets and evaluation metrics
5. Analyze possible challenges and limitations
6. Maintain rigorous scientific reasoning
7. Encourage methodological innovation
8. Ensure the plan is practical and reproducible

Your response should be well-structured and academically professional.
"""

# Prompt 3: Prompt Iteration Update
UPDATED_PROMPT = ""

# =========================================================
# UTIL FUNCTIONS
# =========================================================

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def random_sample_dataset(dataset, sample_size, seed=42):

    random.seed(seed)

    if sample_size >= len(dataset):
        return dataset

    return random.sample(dataset, sample_size)

# =========================================================
# API CALL
# =========================================================
'''
def call_model(client, model, system_prompt, user_input):

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=0.1
    )

    return response.choices[0].message.content
'''
def call_model(client, model, system_prompt, user_input):

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=0.1
    )

    return response.choices[0].message.content

# =========================================================
# SCORE PARSER
# =========================================================

def parse_score(text):

    reliability = 0
    innovation = 0

    # Prevent None
    if text is None:
        return {
            "reliability": 0,
            "innovation": 0,
        }

    try:

        reliability_match = re.search(
            r"Reliability\s*:\s*(\d+)",
            text,
            re.IGNORECASE
        )

        innovation_match = re.search(
            r"Innovation\s*:\s*(\d+)",
            text,
            re.IGNORECASE
        )

        if reliability_match:
            reliability = int(reliability_match.group(1))

        if innovation_match:
            innovation = int(innovation_match.group(1))

    except Exception as e:

        print("Parse Error:", e)
        print(text)

    return {
        "reliability": reliability,
        "innovation": innovation,
    }

def calculate_average_scores(all_scores):

    stats = {
        "self": {
            "reliability": 0,
            "innovation": 0
        },
        "deepseek": {
            "reliability": 0,
            "innovation": 0
        },
        "gpt": {
            "reliability": 0,
            "innovation": 0
        }
    }
    count = len(all_scores)

    if count == 0:
        return stats

    # =====================================================
    # Accumulate
    # =====================================================

    for score in all_scores:

        for model_name in ["self", "deepseek", "gpt"]:

            stats[model_name]["reliability"] += \
                score[model_name]["reliability"]

            stats[model_name]["innovation"] += \
                score[model_name]["innovation"]

    # =====================================================
    # Average
    # =====================================================

    for model_name in ["self", "deepseek", "gpt"]:

        stats[model_name]["reliability"] /= count

        stats[model_name]["innovation"] /= count

    return stats

# =========================================================
# MULTI MODEL SCORING
# =========================================================

def evaluate_plan(plan_text):

    results = {}

    # Self-Model Scoring
    self_score_text = call_model(
        self_client,
        GENERATOR_MODEL,
        SCORING_PROMPT,
        plan_text
    )

    results["self"] = parse_score(self_score_text)

    # DeepSeek Scoring
    ds_score_text = call_model(
        deepseek_client,
        JUDGE_MODEL_DEEPSEEK,
        SCORING_PROMPT,
        plan_text
    )

    results["deepseek"] = parse_score(ds_score_text)

    # GPT Scoring
    gpt_score_text = call_model(
        gpt_client,
        JUDGE_MODEL_GPT,
        SCORING_PROMPT,
        plan_text
    )

    results["gpt"] = parse_score(gpt_score_text)

    return results

# =========================================================
# PROMPT OPTIMIZATION
# =========================================================

PROMPT_OPTIMIZATION_PROMPT = """
You are an expert prompt optimization engineer.

The current prompt is:
----------------
{old_prompt}
----------------

The generated experimental plan is:
----------------
{generated_plan}
----------------

Please analyze the weaknesses of the current prompt
and generate an improved version of the prompt.

Optimization Goals:

1. Improve methodological reliability
2. Improve methodological innovation
3. Strengthen scientific rigor
4. Encourage deeper reasoning
5. Improve experimental completeness
6. Increase clarity and structure
7. Enhance feasibility and reproducibility

The optimized prompt should remain concise,
clear, and highly effective for scientific research planning.
"""

def optimize_prompt(
        old_prompt,
        generated_plan,
):

    optimize_input = PROMPT_OPTIMIZATION_PROMPT.format(
        old_prompt=old_prompt,
        generated_plan=generated_plan,
    )

    new_prompt = call_model(
        self_client,
        GENERATOR_MODEL,
        "You are a Senior Prompt Engineer.",
        optimize_input
    )

    return new_prompt

# =========================================================
# MAIN PIPELINE
# =========================================================

dataset = load_jsonl(r"train.jsonl")
dataset = random_sample_dataset(dataset, 25)

all_initial_scores = []
all_improved_scores = []

for idx, sample in enumerate(dataset):

    print("=" * 80)
    print(f"SAMPLE {idx}")
    print("=" * 80)

    user_input = ""
    for msg in sample["messages"]:

        if msg["role"] == "user":
            user_input = msg["content"]
            break

    assistant_answer = ""
    for msg in sample["messages"]:

        if msg["role"] == "assistant":
            assistant_answer = msg["content"]
            break


    # =====================================================
    # STEP 1
    # Score the ground truth answer
    # =====================================================

    #print("\n[STEP 1] Ground Truth Scoring")

    #gt_scores = evaluate_plan(assistant_answer)

    #print(json.dumps(gt_scores, indent=2))

    # =====================================================
    # STEP 2
    # Use Prompt2 to generate experiment plan
    # =====================================================

    print("\n[STEP 2] Generate Plan")

    generated_plan = call_model(
        self_client,
        GENERATOR_MODEL,
        GENERATION_PROMPT,
        user_input
    )

    print(generated_plan)

    # =====================================================
    # STEP 3
    # Multi-Model Scoring
    # =====================================================

    print("\n[STEP 3] Evaluate Generated Plan")

    gen_scores = evaluate_plan(generated_plan)

    # Record STEP 3 Average Score
    step3_avg = calculate_average_scores([
        {
            "self": gen_scores["self"],
            "deepseek": gen_scores["deepseek"],
            "gpt": gen_scores["gpt"]
        }
    ])

    print("\n[STEP 3 AVERAGE SCORE]")
    print(json.dumps(step3_avg, indent=2))

    all_initial_scores.append({
        "self": gen_scores["self"],
        "deepseek": gen_scores["deepseek"],
        "gpt": gen_scores["gpt"]
    })

    print(json.dumps(gen_scores, indent=2))

    # =====================================================
    # STEP 4
    # Prompt Iteration
    # =====================================================

    print("\n[STEP 4] Prompt Optimization")

    UPDATED_PROMPT = optimize_prompt(
        GENERATION_PROMPT,
        generated_plan,
    )

    print(UPDATED_PROMPT)

    # =====================================================
    # STEP 5
    # Regenerate using the optimized prompt
    # =====================================================

    print("\n[STEP 5] Regenerate With Updated Prompt")

    improved_plan = call_model(
        self_client,
        GENERATOR_MODEL,
        UPDATED_PROMPT,
        user_input
    )

    print(improved_plan)

    # =====================================================
    # FINAL EVALUATION
    # =====================================================

    print("\n[FINAL EVALUATION]")

    improved_scores = evaluate_plan(improved_plan)

    step5_avg = calculate_average_scores([
        {
            "self": improved_scores["self"],
            "deepseek": improved_scores["deepseek"],
            "gpt": improved_scores["gpt"]
        }
    ])

    print("\n[STEP 5 AVERAGE SCORE]")
    print(json.dumps(step5_avg, indent=2))

    all_improved_scores.append({
        "self": improved_scores["self"],
        "deepseek": improved_scores["deepseek"],
        "gpt": improved_scores["gpt"]
    })

    print(json.dumps(improved_scores, indent=2))

print("\n" + "=" * 80)
print("FINAL AVERAGE SCORES (INITIAL PROMPT)")
print("=" * 80)

avg_initial = calculate_average_scores(all_initial_scores)
print(json.dumps(avg_initial, indent=2))


print("\n" + "=" * 80)
print("FINAL AVERAGE SCORES (IMPROVED PROMPT)")
print("=" * 80)

avg_improved = calculate_average_scores(all_improved_scores)
print(json.dumps(avg_improved, indent=2))
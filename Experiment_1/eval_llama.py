import os
import json
import torch
import traceback
import re

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import PeftModel

from openai import OpenAI

# =========================================================
# Configuration
# =========================================================

# Base Model
BASE_MODEL_PATH = "autodl-tmp/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"

# LoRA Model
LORA_MODEL_PATH = "autodl-tmp/outputs/llama_lora/final/final"

# Test Set
TEST_FILE = "test.jsonl"

# Output
OUTPUT_FILE = "reflection_evaluation_llama.jsonl"

# =========================================================
# OpenAI API
# =========================================================

OPENAI_API_KEY = "enter-your-api-key"

openai_client = OpenAI(
    base_url="https://www.autodl.art/api/v1",
    api_key=OPENAI_API_KEY
)

# =========================================================
# DeepSeek API
# =========================================================

DEEPSEEK_API_KEY = "enter-your-api-key"

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# =========================================================
# tokenizer
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token

# =========================================================
# 4-bit Quantization Config
# =========================================================

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=True
)

# =========================================================
# base model
# =========================================================

base_model = AutoModelForCausalLM.from_pretrained(

    BASE_MODEL_PATH,

    quantization_config=bnb_config,

    device_map="auto",

    torch_dtype=torch.float16,

    trust_remote_code=True
)

# =========================================================
# load lora
# =========================================================

model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL_PATH
)

model.eval()
model.config.use_cache = True

# =========================================================
# Judge Prompt
# =========================================================

JUDGE_PROMPT = """
You are an expert AI researcher.

Please evaluate the following AI research proposal.

You need to give two scores:

1. Reliability Score (0-100)

Evaluate:
- technical correctness
- feasibility
- experimental validity
- logical consistency
- scientific rigor

2. Innovation Score (0-100)

Evaluate:
- novelty
- originality
- creativity
- research value
- uniqueness of methodology

Return ONLY valid JSON.
You must output EXACTLY this format.

Do not output markdown.
Do not output explanations.
Do not output additional text.
Format:

{{
    "reliability": 85,
    "innovation": 78,
    "reason": "short explanation"
}}

Research Proposal:

{proposal}
"""

# =========================================================
# Self Judge Prompt
# =========================================================

SELF_JUDGE_PROMPT = """
You are reviewing your own generated research proposal.

Critically evaluate the proposal.

IMPORTANT:
- Output ONLY plain text
- Do NOT output JSON
- Do NOT output markdown
- Do NOT explain anything before answering

Use EXACTLY this format:

Reliability: <0-100 integer>
Innovation: <0-100 integer>
Reason: <one short sentence>

Proposal:

{proposal}
"""

# =========================================================
# Reflection Prompt
# =========================================================

REFLECTION_PROMPT = """
You previously generated a research proposal.

You also identified weaknesses in the proposal.

Your task is to improve the proposal while keeping the same research topic and objective.

Focus on:
- improving technical correctness
- improving experimental rigor
- improving logical consistency
- improving novelty
- improving feasibility

DO NOT:
- rewrite the proposal completely
- change the research topic
- output explanations
- output self evaluation
- output scores
- output markdown
- output section titles

Output ONLY the improved proposal text.

Previous Proposal:
{proposal}

Weakness Analysis:
{reason}
"""


# =========================================================
# load jsonl
# =========================================================

def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data


# =========================================================
# build input
# =========================================================

def build_input(messages):
    text = ""

    for msg in messages:

        role = msg["role"]
        content = msg["content"]

        # Skip assistant messages
        if role == "assistant":
            continue

        text += f"<|{role}|>\n{content}\n"

    return text


# =========================================================
# generate
# =========================================================

def generate_response(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return text.strip()


# =========================================================
# GPT Judge
# =========================================================

def gpt_judge(proposal):
    prompt = JUDGE_PROMPT.format(
        proposal=proposal
    )

    response = openai_client.chat.completions.create(

        model="gpt-5.4",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0
    )

    content = response.choices[0].message.content

    try:

        start = content.find("{")

        end = content.rfind("}") + 1

        json_text = content[start:end]

        return json.loads(json_text)

    except Exception as e:

        print("Judge Parse Error:", e)

        return {
        "reliability": -1,
        "innovation": -1,
        "reason": "parse error"
    }

# =========================================================
# DeepSeek Judge
# =========================================================

def deepseek_judge(proposal):
    prompt = JUDGE_PROMPT.format(
        proposal=proposal
    )

    response = deepseek_client.chat.completions.create(

        model="deepseek-chat",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0
    )

    content = response.choices[0].message.content

    try:

        start = content.find("{")

        end = content.rfind("}") + 1

        json_text = content[start:end]

        return json.loads(json_text)

    except Exception as e:

        print("Judge Parse Error:", e)

        return {
        "reliability": -1,
        "innovation": -1,
        "reason": "parse error"
    }


# =========================================================
# self judge
# =========================================================

def self_judge(proposal):

    prompt = SELF_JUDGE_PROMPT.format(
        proposal=proposal
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    rel = re.search(
        r"Reliability:\s*(\d+)",
        text
    )

    inn = re.search(
        r"Innovation:\s*(\d+)",
        text
    )

    reason = re.search(
        r"Reason:\s*(.*)",
        text,
        re.DOTALL
    )

    return {

        "reliability":
            int(rel.group(1)) if rel else -1,

        "innovation":
            int(inn.group(1)) if inn else -1,

        "reason":
            reason.group(1).strip() if reason else "parse error"
    }


# =========================================================
# reflection generation
# =========================================================

def reflection_generate(proposal, self_score):

    prompt = REFLECTION_PROMPT.format(

        proposal=proposal,

        reason=self_score["reason"]
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return text.strip()


# =========================================================
# main
# =========================================================

def main():

    test_data = load_jsonl(TEST_FILE)

    results = []

    for sample in tqdm(test_data):

        try:

            prompt = build_input(
                sample["messages"]
            )

            # =================================================
            # V1
            # =================================================

            prediction_v1 = generate_response(prompt)

            gpt_v1 = gpt_judge(prediction_v1)

            deepseek_v1 = deepseek_judge(prediction_v1)

            self_v1 = self_judge(prediction_v1)

            # =================================================
            # Reflection
            # =================================================

            prediction_v2 = reflection_generate(
                prediction_v1,
                self_v1
            )

            # =================================================
            # V2 Judge
            # =================================================

            gpt_v2 = gpt_judge(prediction_v2)

            deepseek_v2 = deepseek_judge(prediction_v2)

            self_v2 = self_judge(prediction_v2)

            # =================================================
            # Improvement
            # =================================================

            item = {

                "input": prompt,

                "prediction_v1": prediction_v1,

                "prediction_v2": prediction_v2,

                "gpt_v1": gpt_v1,

                "gpt_v2": gpt_v2,

                "deepseek_v1": deepseek_v1,

                "deepseek_v2": deepseek_v2,

                "self_score_v1": self_v1,

                "self_score_v2": self_v2
            }

            results.append(item)

            with open(
                OUTPUT_FILE,
                "a",
                encoding="utf-8"
            ) as f:

                f.write(
                    json.dumps(
                        item,
                        ensure_ascii=False
                    ) + "\n"
                )

        except Exception:

            traceback.print_exc()

    # =====================================================
    # Statistics
    # =====================================================

    def avg(x):

        valid = [v for v in x if v >= 0]

        if len(valid) == 0:
            return -1

        return sum(valid) / len(valid)

    gpt_rel_before = []
    gpt_rel_after = []

    gpt_inn_before = []
    gpt_inn_after = []

    ds_rel_before = []
    ds_rel_after = []

    ds_inn_before = []
    ds_inn_after = []

    self_rel_before = []
    self_rel_after = []

    self_inn_before = []
    self_inn_after = []

    for r in results:

        gpt_rel_before.append(
            r["gpt_v1"]["reliability"]
        )

        gpt_rel_after.append(
            r["gpt_v2"]["reliability"]
        )

        gpt_inn_before.append(
            r["gpt_v1"]["innovation"]
        )

        gpt_inn_after.append(
            r["gpt_v2"]["innovation"]
        )

        ds_rel_before.append(
            r["deepseek_v1"]["reliability"]
        )

        ds_rel_after.append(
            r["deepseek_v2"]["reliability"]
        )

        ds_inn_before.append(
            r["deepseek_v1"]["innovation"]
        )

        ds_inn_after.append(
            r["deepseek_v2"]["innovation"]
        )

        self_rel_before.append(
            r["self_score_v1"]["reliability"]
        )

        self_rel_after.append(
            r["self_score_v2"]["reliability"]
        )

        self_inn_before.append(
            r["self_score_v1"]["innovation"]
        )

        self_inn_after.append(
            r["self_score_v2"]["innovation"]
        )

    print("\n================ FINAL RESULTS ================\n")

    print("GPT Reliability Before:", avg(gpt_rel_before))
    print("GPT Reliability After:", avg(gpt_rel_after))

    print()

    print("GPT Innovation Before:", avg(gpt_inn_before))
    print("GPT Innovation After:", avg(gpt_inn_after))

    print()

    print("DeepSeek Reliability Before:", avg(ds_rel_before))
    print("DeepSeek Reliability After:", avg(ds_rel_after))

    print()

    print("DeepSeek Innovation Before:", avg(ds_inn_before))
    print("DeepSeek Innovation After:", avg(ds_inn_after))

    print()

    print("Self Reliability Before:", avg(self_rel_before))
    print("Self Reliability After:", avg(self_rel_after))

    print()

    print("Self Innovation Before:", avg(self_inn_before))
    print("Self Innovation After:", avg(self_inn_after))

# =========================================================
# start
# =========================================================

if __name__ == "__main__":

    main()
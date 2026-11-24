import os
import json
from openai import OpenAI

# ========== 1. 初始化 DeepSeek ==========
client = OpenAI(
    api_key="sk-f375071129cd488093ca55935a9e1afa",
    base_url="https://api.deepseek.com"
)

# ========== 2. Long-CoT Prompt 模板 ==========
PROMPT_TEMPLATE = """
You are an expert research analyst.

You are given a structured JSON of a research paper, where sections such as Abstract, Introduction, Method, Experiment, and Conclusion are already separated.

Your task is NOT to summarize the paper directly.

Instead, you must generate a **Long Chain-of-Thought (Long-CoT)** that reconstructs the reasoning process of the paper in a structured, step-by-step manner.

Additionally, you must generate a **simulated user request** based on the paper, which will be used for training a model to generate research plans.

---

# Requirements

1. The output MUST be structured into clearly defined stages.
2. Each stage must include:

   * A goal
   * Detailed reasoning steps
   * Intermediate conclusions
3. The reasoning must be grounded ONLY in the provided content. Do NOT hallucinate.
4. Avoid redundancy. Each step must add new information.
5. The final output should read like a **progressive reasoning process**, not a flat summary.
6. You MUST generate a **fuzzy user request** that mimics a real user's input.

---

# Long-CoT Structure

## [Stage 1] Problem Identification & Research Background

* Extract the core research problem
* Explain the background and why this problem is important
* Identify limitations of existing work
* Intermediate conclusion: Clearly define the research gap

---

## [Stage 2] Method Understanding & Design Logic

* Explain the main idea of the proposed method
* Break down key components/mechanisms
* Explain WHY this method can address the identified problem
* If applicable, describe model structure / algorithm flow
* Intermediate conclusion: Link method to problem

---

## [Stage 3] Experimental Setup & Evaluation Strategy

* Describe datasets, baselines, and evaluation metrics
* Explain how experiments are designed to validate the method
* Highlight what is being tested (e.g., performance, robustness)
* Intermediate conclusion: Explain what evidence is expected

---

## [Stage 4] Results Analysis & Evidence Integration

* Extract key experimental results
* Compare with baselines
* Explain what the results imply
* Intermediate conclusion: Whether evidence supports the method

---

## [Stage 5] Final Conclusion & Problem Resolution

* Determine whether the original problem is solved
* Summarize the contribution
* Mention any limitations (if present)
* Final conclusion: Overall effectiveness of the work

---

## [Stage 6] Simulated User Request (NEW)

* Generate a **realistic, slightly vague user input**
* It MUST be based on:

  * The core problem (from Stage 1)
  * Implicit or explicit constraints (conditions)
* The request should NOT look like a paper summary
* It should resemble how a real user would ask for help

### Conditions Generation Rules:

* If the paper does not specify constraints, you MUST infer or generate plausible ones, such as:

  * limited compute
  * small dataset
  * real-time requirement
  * low latency
  * limited training time

### Style Requirements:

* Natural language (not structured JSON)
* Slightly incomplete or underspecified
* Do NOT include phrases like "this paper" or "the authors"

---

# Output Format (STRICT)

[Stage 1]
Goal:
Reasoning:
Intermediate Conclusion:

[Stage 2]
Goal:
Reasoning:
Intermediate Conclusion:

[Stage 3]
Goal:
Reasoning:
Intermediate Conclusion:

[Stage 4]
Goal:
Reasoning:
Intermediate Conclusion:

[Stage 5]
Goal:
Reasoning:
Final Conclusion:

[Stage 6]
User Request:
(Write a natural, slightly vague user request that includes problem + conditions)

---

# Input JSON:

{paper_json}
"""

# ========== 3. 单个文件处理 ==========
def generate_long_cot(json_data):
    prompt = PROMPT_TEMPLATE.format(
        paper_json=json.dumps(json_data, ensure_ascii=False, indent=2)
    )

    response = client.chat.completions.create(
        model="deepseek-chat",   # 推荐这个
        messages=[
            {"role": "system", "content": "You are a precise and structured reasoning assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,   # 低一点更稳定
        max_tokens=4000    # ⚠️ 很重要，Long-CoT必须开大
    )

    return response.choices[0].message.content


# ========== 4. 批量处理 ==========
def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".json"):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".json", "_long_cot.txt"))

        print(f"Processing: {filename}")

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        try:
            result = generate_long_cot(data)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)

        except Exception as e:
            print(f"Error processing {filename}: {e}")


# ========== 5. 运行 ==========
if __name__ == "__main__":
    process_folder(r"D:\PycharmProjects\BioResearcher-main\auto_bioresearcher\download\json", "raw_jsons")

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

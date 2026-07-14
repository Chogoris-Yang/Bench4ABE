# BioBench: A Comprehensive LLM Benchmark for Automated Bioinformatics Experimental Protocol Generation

This repository contains the code and data for the paper **"A Comprehensive LLM Benchmark for Automated Bioinformatics Experimental Protocol Generation"**. The project systematically evaluates and optimizes Large Language Models (LLMs) for generating rigorous, innovative experimental plans in bioinformatics research.

## Dataset

The dataset is stored in the `gen_lcot\data\` directory:

| File | Description |
|------|-------------|
| `train.jsonl` | Training set (32 MB, ~60K samples) |
| `test.jsonl` | Test set |
| `eval.jsonl` | Evaluation set |
| `total.jsonl` | Complete dataset (32 MB) |

Each sample follows the `messages` format with `role` (`user`/`assistant`) and `content` fields, where `user` contains the research question and `assistant` contains the reference experimental plan.

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for Experiment_1 fine-tuning)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/biobench.git
cd biobench

# Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### API Keys

Before running any experiment, you must configure your API keys. Replace `enter-your-api-key` in the following files with your actual keys:

- **Experiment_1**: `gemini_IF.py`, `grok_IF.py`, `eval_*.py`
- **Experiment_2**: All `.py` files in the directory
- **Experiment_3**: `config.py`
- **Experiment_4**: `config_rag.py`
- **Experiment_5**: `config_exp5.py`
- **Experiment_6**: `config_exp6.py`

API endpoints used:
- [DeepSeek API](https://platform.deepseek.com/) — Generator & Judge models
- [AutoDL API](https://www.autodl.art/) — GPT, GLM, MiniMax, Qwen, Kimi models
- [xAI API](https://api.x.ai/) — Grok models

## Experiments

### Experiment_1 — Baseline Fine-tuned Models & Instruction Following

QLoRA fine-tuning of 4 open-source LLMs and instruction-following pipelines for 2 proprietary models.

```bash
# Fine-tune models with QLoRA (requires GPU)
python Experiment_1/ft_llama.py
python Experiment_1/ft_mistral.py
python Experiment_1/ft_qwen8b.py
python Experiment_1/ft_gemma.py

# Run instruction-following pipelines (Gemini & Grok)
python Experiment_1/gemini_IF.py
python Experiment_1/grok_IF.py

# Evaluate fine-tuned models with reflection
python Experiment_1/eval_llama.py
python Experiment_1/eval_mistral.py
python Experiment_1/eval_qwen8b.py
python Experiment_1/eval_gemma.py
```

### Experiment_2 — Agent-Based Reasoning Frameworks

11 distinct agent-based reasoning strategies using LangGraph with DeepSeek V4 Pro as the generator.

```bash
# ReAct Agent (Think → Act → Observe loop)
python Experiment_2/react.py

# Reflexion (self-reflection with verbal RL signals)
python Experiment_2/reflexion.py

# Self-Refine (iterative self-improvement)
python Experiment_2/self_refine.py

# Reverse Chain-of-Thought
python Experiment_2/reverse_cot.py

# Least-to-Most decomposition
python Experiment_2/least_to_most.py

# Plan-and-Solve strategy
python Experiment_2/plan_and_solve.py

# Step-Back abstraction reasoning
python Experiment_2/step_back.py

# Language Agent Tree Search
python Experiment_2/lats.py

# Tree of Thoughts
python Experiment_2/tot.py

# Skeleton of Thought
python Experiment_2/sot.py

# Self-Discover reasoning structures
python Experiment_2/self_discover.py

# Voyager-style curriculum exploration
python Experiment_2/voyager.py

# ReWOO (Reasoning Without Observation)
python Experiment_2/rewoo.py

# Multi-Agent Debate
python Experiment_2/debate.py

# Critic feedback optimization
python Experiment_2/critic.py

# DSPy pipeline
python Experiment_2/dspy_pipeline.py
```

### Experiment_3 — Reinforcement Learning Strategy Optimization

Five RL strategies with a multi-model role architecture (6 models with dedicated roles).

```bash
# Interactive mode (supports /run, /batch, /compare, /strategy commands)
python Experiment_3/rl_runner.py

# Run PPO on a single question
python Experiment_3/rl_runner.py --run "Design an experiment for predicting protein-protein interactions"

# Batch test GRPO on 5 questions
python Experiment_3/rl_runner.py --batch 5 --strategy grpo

# Compare all 5 strategies on 3 questions
python Experiment_3/rl_runner.py --compare 3

# Run individual agents directly
python Experiment_3/ppo_agent.py
python Experiment_3/grpo_agent.py
python Experiment_3/gspo_agent.py
python Experiment_3/dapo_agent.py
python Experiment_3/odp_agent.py
```

### Experiment_4 — RAG-Enhanced Pipeline

Retrieval-Augmented Generation integrating PubMed literature into protocol generation.

```bash
# Interactive menu (run sub-experiments 4.1, 4.2, 4.3)
python Experiment_4/rag_runner.py

# Retrieval strategy ablation study
python Experiment_4/rag_runner.py --exp 4.1

# RAG × RL cross experiment
python Experiment_4/rag_runner.py --exp 4.2

# RAG integration depth experiment
python Experiment_4/rag_runner.py --exp 4.3

# Run all sub-experiments
python Experiment_4/rag_runner.py --exp all
```

### Experiment_5 — 4-Stage End-to-End Pipeline

A complete pipeline with literature retrieval, plan generation, prompt optimization, and code generation, running 3 full iterations.

```bash
# Run the full 4-stage × 3-round pipeline
python Experiment_5/pipeline_runner.py

# Run individual stages
python Experiment_5/stages/stage1_retrieval.py
python Experiment_5/stages/stage2_iteration.py
python Experiment_5/stages/stage3_prompt.py
python Experiment_5/stages/stage4_codegen.py
```

### Experiment_6 — RL-Enhanced Pipeline with GLM Optimizer

Extends Experiment_5 with GLM-5.1 as an RL policy optimizer supporting PPO, GRPO, and Direct strategies.

```bash
# Run the RL-enhanced pipeline
python Experiment_6/runner.py
```

## Scoring System

All experiments use a **dual-judge scoring system** evaluating two dimensions:

- **Reliability** (0–100): Technical correctness, experimental rigor, feasibility, logical consistency, reproducibility
- **Innovation** (0–100): Originality, novelty, creativity, research contribution, uniqueness of methodology

The final reward is a weighted combination: **60% Reliability + 40% Innovation**.

## Project Structure

```
biobench/
├── README.md
├── requirements.txt
├── .gitignore
├── prompts_collection.txt          # Complete prompt catalog (50+ prompts)
├── gen_lcot/data/                   # Dataset files
├── Experiment_1/                    # Baseline: fine-tuning & instruction following
├── Experiment_2/                    # 11 agent-based reasoning frameworks
├── Experiment_3/                    # 5 RL strategies (PPO/GRPO/GSPO/DAPO/ODP)
├── Experiment_4/                    # RAG-enhanced retrieval pipeline
├── Experiment_5/                    # 4-stage end-to-end pipeline
└── Experiment_6/                    # RL-enhanced pipeline with GLM optimizer
```

## License

This project is for research purposes. Please cite the paper if you use this code or dataset.

## Citation

```bibtex
@article{biobench2025,
  title={A Comprehensive LLM Benchmark for Automated Bioinformatics Experimental Protocol Generation},
  author={},
  journal={},
  year={2025}
}
```

# Experiment_3: RL-Based Experiment Protocol Generation

Five reinforcement learning strategies adapted for LLM-based experimental plan generation.

## Architecture

```
Experiment_3/
├── __init__.py          # Package init
├── config.py             # Shared configuration & hyperparameters
├── rl_components.py      # Common RL infrastructure
│   ├── LLM clients (DeepSeek + GPT)
│   ├── Dual-judge scoring system
│   ├── Data loading from train.jsonl
│   ├── RL utility functions (advantages, clipping, etc.)
│   └── BaseRLAgent abstract class
├── ppo_agent.py          # PPO: Clipped surrogate objective
├── grpo_agent.py         # GRPO: Group-relative advantage (no Critic)
├── gspo_agent.py         # GSPO: Sequential group sampling + elitism
├── dapo_agent.py         # DAPO: Asymmetric clipping + dynamic filter
├── odp_agent.py          # ODP: Online teacher-student distillation
└── rl_runner.py          # Unified CLI & batch runner
```

## Strategies

| Strategy | Key Mechanism | Math | Unique Feature |
|----------|-------------|------|---------------|
| **PPO** | Clipped surrogate | `min(r·A, clip(r,1-ε,1+ε)·A)` | Needs Critic (Value) model |
| **GRPO** | Group-relative advantage | `A_i = (r_i - μ)/σ` | No Value model needed |
| **GSPO** | Sequential elitism+mutation | Cumulative group advantage | Evolutionary optimization |
| **DAPO** | Asymmetric clipping | `ε_high > ε_low` | Dynamic filter + token-level |
| **ODP** | Teacher-student distill | `α·L_distill + (1-α)·L_reward` | Online GPT-5.4 → DeepSeek |

## Usage

### Interactive Mode
```bash
python rl_runner.py
```

### CLI Commands
```
/run <question>          — Run current strategy on a question
/batch [N]               — Batch test on N samples
/compare [N]             — Compare ALL 5 strategies
/strategy <name>         — Switch: ppo/grpo/gspo/dapo/odp
/strategies              — List all strategies
/demo                    — Built-in demo
```

### Command Line Arguments
```bash
# Run a single question with PPO
python rl_runner.py --run "Design an experiment for..."

# Batch test GRPO on 5 questions
python rl_runner.py --batch 5 --strategy grpo

# Compare all 5 strategies on 3 questions
python rl_runner.py --compare 3
```

### Python API
```python
from rl_runner import generate_experiment, compare_strategies_on_tasks

# Generate protocols using a single strategy
results = generate_experiment(
    task_descriptions=["Task 1", "Task 2", "Task 3"],
    strategy="grpo"
)

# Compare all strategies
comparison = compare_strategies_on_tasks(
    task_descriptions=["Task 1", "Task 2"]
)
```

## Output Format

Each result dict contains:
- `final_report`: Full text report with training history
- `best_plan`: Best experimental plan text
- `best_scores`: {reliability, innovation}
- `best_reward`: Combined scalar reward
- `strategy`: Strategy name
- Strategy-specific metrics (PPO: epochs, GRPO: generations, etc.)

## Scoring

Dual-judge system:
- **Judge 1**: DeepSeek-chat → scores Reliability + Innovation
- **Judge 2**: GPT-5.4 → scores Reliability + Innovation
- **Combined**: Average of both judges
- **Reward**: Weighted combination (60% Reliability + 40% Innovation)

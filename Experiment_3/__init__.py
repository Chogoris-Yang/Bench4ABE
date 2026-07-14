"""
Experiment_3: RL-based Experiment Protocol Generation
=====================================================
Five reinforcement learning strategies (PPO, GRPO, GSPO, DAPO, ODP)
adapted for LLM-based experimental plan generation.

Each strategy formulates plan generation as an RL problem:
- State: research question + current plan
- Action: generate/modify a plan
- Reward: dual-judge score (Reliability + Innovation)
- Policy: LLM generation strategy
"""

__version__ = "1.0.0"

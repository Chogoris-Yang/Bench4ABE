"""
Experiment_4: RAG-Enhanced RL Experiment Protocol Generation
===========================================================
Introduces external knowledge (PubMed papers, prior protocols) via
retrieval-augmented generation into 5 RL strategies from Experiment_3.

Sub-experiments:
  4.1 — Retrieval Strategy Ablation (Dense/Sparse/Hybrid × Papers/Prior/All)
  4.2 — RAG × RL Cross Experiment (5 strategies × with/without RAG)
  4.3 — RAG Integration Depth (pre_gen / post_gen / iterative / full_pipeline)
  (4.4 skipped — knowledge source ablation)

Key innovation: External knowledge → better experimental designs.
"""

__version__ = "1.0.0"

# Deep Reinforcement Learning Coursework

This repository is organized in phases so implementation and reporting can
progress together.

## Phase Roadmap

1. `Phase 1`: Snake environment
2. `Phase 2`: Tabular Q-learning baseline and hyperparameter experiments
3. `Phase 3`: DQN baseline and hyperparameter experiments
4. `Phase 4`: Advanced DRL extensions and final comparison

## Current Scope

The current codebase implements:

- A custom Snake environment with an engineered state representation
- A tabular Q-learning agent
- A Q-learning hyperparameter sweep pipeline
- Report-ready outputs including CSV summaries, plots, and markdown notes

## Run Phase 2

```bash
python3 -m src.snake_rl.experiments.run_q_learning
```

Outputs are written under `outputs/q_learning/`.

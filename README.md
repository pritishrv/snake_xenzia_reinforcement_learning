# Deep Reinforcement Learning Coursework

Pritish Ranjan Verma, Roberto Tsuneki Sa Freire

## Scope

The current codebase implements:

- A custom Snake environment with an engineered state representation
- A tabular Q-learning agent
- A Q-learning hyperparameter sweep pipeline
- Report-ready outputs including CSV summaries, plots, and markdown notes
- A pygame-based live training viewer for Q-learning

## Run tabular Q-learning experiments

```bash
python3 -m src.snake_rl.experiments.run_q_learning
```

## Run tabular Q-learning experiments

```bash
python3 -m src.snake_rl.experiments.run_dqn.py
```

## Run tabular Q-learning experiments

```bash
python3 -m src.snake_rl.experiments.run_ppo.py
```

Outputs are written under `outputs/q_learning/`.

## Watch Training Live

Install `pygame`, then run:

```bash
python3 -m src.snake_rl.experiments.watch_q_learning --experiment epsilon_fast_decay_dense
```

This renders training live and then plays back the learned greedy policy.

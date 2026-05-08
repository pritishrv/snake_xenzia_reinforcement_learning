# Phase 3 Report: Deep Q-Network (DQN) Analysis

This report documents the experimental results for the DQN implementation on the Snake Xenzia environment.

## 1. Objective
The goal of Phase 3 was to transition from tabular Q-learning to a Deep Q-Network (DQN) to evaluate improvements in learning capacity, final policy performance, and scalability within the engineered state space.

## 2. Experimental Setup

### Neural Network Architecture
- **Type**: Multi-Layer Perceptron (MLP)
- **Input Dimension**: 11 (Engineered state vector)
- **Hidden Layers**: 2 layers with 128 neurons each
- **Activation**: ReLU
- **Output Dimension**: 3 (Straight, Left, Right)

### Core Hyperparameters (Baseline)
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Gamma (Discount Factor)**: 0.99
- **Replay Buffer**: 10,000 capacity
- **Batch Size**: 64
- **Target Network Update**: Every 100 steps
- **Episodes**: 500

## 3. Comparative Results Table

| Experiment | Avg Reward (Last 50) | Avg Score (Last 50) | Max Score | Avg Steps (Last 50) |
|---|---|---|---|---|
| **DQN Baseline (LR 1e-3)** | **80.97** | **8.38** | **22** | **72.72** |
| DQN Low LR (1e-4) | 65.99 | 6.96 | 20 | 61.52 |
| DQN Large Batch (128) | 54.16 | 5.84 | 18 | 52.44 |
| *Tabular Q-Learning (Phase 2 Ref)* | *20.93* | *2.86* | *7* | *29.88* |

## 4. Analysis of Results

### 4.1. DQN vs. Tabular Q-Learning
The transition to DQN resulted in a massive performance leap:
- **Score Improvement**: Average score increased from **~2.9** to **~8.4** (a 190% improvement).
- **Peak Performance**: The maximum score achieved by DQN (**22**) tripled the tabular best (**7**).
- **Survival Capability**: Average steps per episode increased from **~30** to **~73**, indicating significantly better obstacle avoidance and longer survival.

### 4.2. Learning Rate Sensitivity
- The baseline learning rate of **0.001** performed significantly better than **0.0001**.
- With the lower learning rate, the agent learned more stably but much slower, failing to reach the same peak performance within the 500-episode limit.

### 4.3. Batch Size Impact
- Increasing the batch size to **128** unexpectedly resulted in lower performance (**Avg Score 5.84**) compared to the baseline of **64**.
- This suggests that with a relatively small state space (11 features), larger batches might be over-smoothing the gradient updates or requiring more episodes to converge than the allocated 500.

## 5. Visual Observations
- **Early Phase (0-150)**: The agent primarily learns to avoid walls. Rewards are negative but improving.
- **Mid Phase (150-350)**: Navigation towards food becomes consistent. The "closer_to_food" dense rewards provide strong signals that accelerate convergence.
- **Late Phase (350-500)**: The agent develops complex maneuvers to avoid its own body as the snake grows longer, a behavior barely seen in the tabular version.

## 6. Conclusion
DQN has successfully demonstrated superior generalization over tabular methods. Even with the same engineered features, the function approximation of the neural network allows the agent to handle the spatial relationships between the snake, food, and walls more effectively.

---
*Report generated on 2026-05-08*
*Data source: snake_xenzia_reinforcement_learning/outputs/dqn/*

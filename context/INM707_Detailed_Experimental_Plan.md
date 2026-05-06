# INM707 Detailed Experimental Plan
## Self-Learning Snake Xenzia using Reinforcement Learning

This document presents the detailed experimental execution plan for the INM707 Deep Reinforcement Learning coursework project.

---

# 1. Research Questions

1. Can a tabular Q-learning agent learn an effective policy for Snake Xenzia?
2. How do reward shaping and exploration strategies affect learning performance?
3. Does DQN outperform tabular Q-learning in larger environments?
4. Do advanced DRL algorithms improve stability and performance compared to standard DQN?

---

# 2. Global Hypothesis

Deep Reinforcement Learning methods are expected to outperform classical Q-learning in terms of:
- Scalability
- Stability
- Learning efficiency
- Generalisation capability

---

# 3. Experimental Pipeline

```text
Environment → Q-Learning → DQN → Advanced DRL Extensions
```

The project follows a progressive reinforcement learning pipeline moving from classical RL methods to modern DRL techniques.

---

# 4. Phase 1 — Environment Development

## Objective
Build a custom Snake Xenzia environment using Python.

## Tasks
- Implement snake movement mechanics
- Food spawning
- Collision detection
- Score tracking
- Gym-style RL API integration

## State Representation
The state vector will include:
- Danger straight
- Danger left
- Danger right
- Current movement direction
- Relative food direction

Example:
```python
[
 danger_straight,
 danger_left,
 danger_right,
 moving_up,
 moving_down,
 moving_left,
 moving_right,
 food_up,
 food_down,
 food_left,
 food_right
]
```

## Hypothesis
A compact engineered state representation will:
- Reduce state-space complexity
- Improve convergence speed
- Improve learning stability

## Expected Outcome
- Stable RL environment
- Faster learning
- Reduced memory requirements

---

# 5. Phase 2 — Tabular Q-Learning

## Objective
Train a baseline RL agent using tabular Q-learning.

## Tasks
- Implement Q-table
- Implement epsilon-greedy exploration
- Implement reward shaping
- Train over thousands of episodes

## Q-Learning Update Rule

```text
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') − Q(s,a)]
```

## Hypothesis
The Q-learning agent will learn:
- Basic survival behaviour
- Food-seeking navigation
- Simple obstacle avoidance

## Expected Outcome
Expected:
- Gradual improvement in rewards
- Better survival time
- Increasing snake length

Limitations expected:
- Poor scalability
- Slow convergence in larger environments

---

# 6. Hyperparameter Experiments

## Experiment A — Exploration Rate

### Variables
- Fast epsilon decay
- Slow epsilon decay
- Constant exploration

### Hypothesis
Slower epsilon decay will improve long-term learning by encouraging broader exploration.

### Expected Results
- Better final policy
- Slower early convergence

---

## Experiment B — Reward Shaping

### Variables
- Sparse rewards
- Dense rewards

### Hypothesis
Dense rewards will accelerate learning by providing continuous feedback.

### Expected Results
- Faster convergence
- Higher average reward

---

## Experiment C — Environment Size

### Variables
- 6×6 grid
- 10×10 grid
- 15×15 grid

### Hypothesis
Tabular Q-learning performance will decrease significantly as state-space size increases.

### Expected Results
- Small environments: stable learning
- Large environments: poor scalability

---

# 7. Phase 3 — Deep Q-Network (DQN)

## Objective
Replace the Q-table with a neural network.

## Improvements Implemented
- Experience Replay
- Target Network

## Hypothesis
DQN will outperform tabular Q-learning in larger state spaces due to function approximation.

## Expected Outcome
Expected:
- Better scalability
- Improved generalisation
- Higher rewards
- Faster convergence

---

# 8. Individual Extension — Pritish (Double DQN)

## Objective
Implement Double DQN to reduce Q-value overestimation.

## Hypothesis
Double DQN will:
- Reduce overestimation bias
- Improve training stability
- Improve convergence behaviour

## Expected Outcome
Expected:
- More stable rewards
- Lower variance
- Better long-term performance

---

# 9. Individual Extension — Roberto (PPO)

## Objective
Implement Proximal Policy Optimisation (PPO).

## Hypothesis
PPO will:
- Improve exploration behaviour
- Learn smoother policies
- Produce stable policy optimisation

## Expected Outcome
Expected:
- Stable training
- Smooth reward progression
- Strong exploration capability

---

# 10. Evaluation Metrics

| Metric | Purpose |
|---|---|
| Average Reward | Overall performance |
| Maximum Score | Peak capability |
| Episode Length | Survival behaviour |
| Training Stability | Variance analysis |
| Convergence Speed | Learning efficiency |
| Loss Curves | Optimisation monitoring |

---

# 11. Final Comparative Analysis

| Algorithm | Avg Score | Stability | Scalability | Learning Speed |
|---|---|---|---|---|
| Q-Learning |  |  |  |  |
| DQN |  |  |  |  |
| Double DQN |  |  |  |  |
| PPO |  |  |  |  |

---

# 12. Conclusion

This project creates a strong academic progression from classical reinforcement learning to advanced deep reinforcement learning techniques.

The project also clearly separates:
- Shared team contributions
- Individual advanced DRL investigations

This structure aligns strongly with the coursework assessment requirements and creates a coherent MSc-level reinforcement learning investigation.

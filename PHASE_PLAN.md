# Coursework Phase Plan

This project should be executed and reported in tightly coupled phases.

## Phase 1: Environment

### Coding

- Implement custom Snake environment
- Define action space and engineered state representation
- Add reward configuration support

### Report Output

- Problem framing
- Environment mechanics
- State representation rationale
- Reward design rationale

## Phase 2: Tabular Q-learning

### Coding

- Implement Q-table agent
- Add epsilon-greedy exploration
- Add training loop
- Add hyperparameter sweep runner
- Save CSV metrics, plots, and markdown summaries

### Experiments

- Fast epsilon decay
- Slow epsilon decay
- Constant exploration
- Dense vs sparse rewards

### Report Output

- Q-learning method section
- Hyperparameter table
- Reward and score curves
- Early analysis of exploration and reward shaping

## Phase 3: DQN

### Coding

- Replace Q-table with neural network
- Add replay buffer
- Add target network
- Add configurable hidden sizes and training hyperparameters

### Experiments

- Learning rate comparison
- Batch size comparison
- Replay buffer size comparison
- Target update frequency comparison
- Reward shaping comparison

### Report Output

- DQN architecture and training description
- DQN learning curves
- Q-learning vs DQN comparison table

## Phase 4: Advanced Extension

### Coding

- Implement Double DQN
- Keep the same environment and evaluation pipeline for fair comparison

### Experiments

- DQN vs Double DQN
- Stability and overestimation analysis

### Report Output

- Individual contribution section
- Stability comparison figures
- Final comparative analysis

## Shared Reporting Rule

Every phase should produce:

- A reproducible config
- Raw per-episode metrics
- A compact summary table
- At least one report-ready plot
- Short written notes on what changed and what was learned

# Q-Learning Phase Report

## 1. Overview

This report documents the first reinforcement learning stage of the Snake
coursework project: a tabular Q-learning baseline on a custom Snake
environment.

The goal of this phase is to answer three practical questions:

1. Can a tabular Q-learning agent learn useful Snake behaviour at all?
2. How sensitive is the agent to the exploration schedule?
3. Does reward shaping improve the quality or speed of learning?

At this stage, all experiments were run on a `10x10` grid and are best treated
as a controlled baseline rather than the final coursework conclusion.

## 2. Environment Design

The environment is implemented in
[snake_env.py](/Users/pritishrv/Documents/Courseworks/Deep_Reinforcement_Learning/src/snake_rl/env/snake_env.py:1).

### 2.1 Game Mechanics

- The snake starts near the centre of the board with length `3`
- Food spawns randomly on an unoccupied grid cell
- The episode ends when the snake collides with a wall or itself
- The episode also terminates if the snake takes too many steps without eating
  food

For the current implementation, the step-without-food cutoff is tied to the
 configured maximum steps per episode, which was set to `250`.

### 2.2 State Representation

The Q-learning agent does not observe the full board directly. Instead, it uses
an engineered `11`-dimensional binary state:

1. danger straight
2. danger left
3. danger right
4. moving up
5. moving down
6. moving left
7. moving right
8. food up
9. food down
10. food left
11. food right

This state abstraction is important. Food appears at random locations, so the
agent cannot simply memorize a fixed path. Instead, it learns reusable local
decision patterns such as:

- move toward food when safe
- avoid immediate collision
- choose left or right turns based on nearby danger

This compression is what makes tabular Q-learning feasible in the first place.

### 2.3 Action Space

The action space contains `3` actions:

- `0`: keep moving straight
- `1`: turn left relative to the current direction
- `2`: turn right relative to the current direction

This relative action design simplifies learning because the agent reasons in
terms of local navigation, not absolute orientation.

## 3. Reward Design

Two reward schemes were used.

### 3.1 Dense Reward

- food: `+10.0`
- death: `-10.0`
- step penalty: `-0.1`
- moving closer to food: `+0.3`
- moving farther from food: `-0.3`

The dense scheme gives continuous feedback and is intended to accelerate
learning by rewarding progress toward food even before the snake actually eats.

### 3.2 Sparse Reward

- food: `+10.0`
- death: `-10.0`
- step penalty: `0.0`
- moving closer to food: `0.0`
- moving farther from food: `0.0`

The sparse scheme only rewards success and failure, making the task more
difficult because intermediate navigation choices receive no guidance.

## 4. Q-Learning Method

The tabular agent is implemented in
[q_learning.py](/Users/pritishrv/Documents/Courseworks/Deep_Reinforcement_Learning/src/snake_rl/agents/q_learning.py:1).

The update rule is the standard Q-learning equation:

`Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]`

### 4.1 Meaning of the Main Hyperparameters

- `alpha`: learning rate. Controls how quickly new experience overrides old
  estimates.
- `gamma`: discount factor. Controls how much future reward matters.
- `epsilon`: exploration rate. Controls how often the agent acts randomly.
- `epsilon_decay`: reduces exploration over time.
- `epsilon_min`: lower bound on exploration.

### 4.2 Training Procedure

Training is implemented in
[q_learning_training.py](/Users/pritishrv/Documents/Courseworks/Deep_Reinforcement_Learning/src/snake_rl/experiments/q_learning_training.py:1).

For each episode:

1. Reset the environment
2. Repeatedly choose an action using epsilon-greedy selection
3. Step the environment
4. Update the Q-table using the observed transition
5. Continue until collision or the step budget is exhausted
6. Decay epsilon at the end of the episode

The outputs of each run are saved under
[outputs/q_learning](/Users/pritishrv/Documents/Courseworks/Deep_Reinforcement_Learning/outputs/q_learning).

## 5. Experimental Setup

The experiment configuration is stored in
[q_learning_base.json](/Users/pritishrv/Documents/Courseworks/Deep_Reinforcement_Learning/configs/q_learning_base.json:1).

### 5.1 Shared Settings

- grid size: `10x10`
- episodes: `300`
- maximum steps per episode: `250`
- moving average window for plots: `25`
- random seed: `7`

### 5.2 Hyperparameter Variants

Four experiments were run:

1. `epsilon_fast_decay_dense`
2. `epsilon_slow_decay_dense`
3. `constant_exploration_dense`
4. `epsilon_slow_decay_sparse`

The current sweep changes exploration behaviour and reward design while keeping
`alpha = 0.1` and `gamma = 0.95` fixed.

## 6. Results

The current summary table comes from
[summary.csv](/Users/pritishrv/Documents/Courseworks/Deep_Reinforcement_Learning/outputs/q_learning/summary.csv:1).

| Experiment | Reward Scheme | Epsilon Behaviour | Final Epsilon | Avg Reward Last 50 | Avg Score Last 50 | Avg Steps Last 50 | Max Score | Visited States |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `epsilon_fast_decay_dense` | dense | decay `1.0 -> 0.05` with `0.985` | `0.0500` | `87.538` | `8.90` | `69.46` | `28` | `241` |
| `epsilon_slow_decay_dense` | dense | decay `1.0 -> 0.05` with `0.995` | `0.2223` | `20.926` | `2.86` | `29.88` | `7` | `211` |
| `constant_exploration_dense` | dense | fixed `0.2` | `0.2000` | `26.836` | `3.44` | `32.38` | `14` | `228` |
| `epsilon_slow_decay_sparse` | sparse | decay `1.0 -> 0.05` with `0.995` | `0.2223` | `20.600` | `3.06` | `31.48` | `11` | `202` |

## 7. Interpretation of Each Result

### 7.1 Fast Decay with Dense Reward

This was the strongest run by a large margin.

Observations:

- highest average reward
- highest average score
- longest survival time
- highest maximum score

Interpretation:

The agent appears to benefit from two conditions at the same time:

- informative dense reward shaping
- aggressive reduction in exploration

Because epsilon reached its minimum value of `0.05`, the agent spent more of
the later episodes exploiting what it had already learned. In a small and
engineered state space like this one, that is enough to produce visibly better
behaviour. The longer average episode length also suggests that the snake
learned both survival and food-seeking behaviour rather than only one of the
two.

### 7.2 Slow Decay with Dense Reward

This run performed much worse than the fast-decay dense run.

Observations:

- average reward dropped from `87.538` to `20.926`
- average score dropped from `8.90` to `2.86`
- max score only reached `7`

Interpretation:

The most likely explanation is not that slow decay is always bad. It is that
the training budget of `300` episodes is relatively short. With `epsilon_decay`
set to `0.995`, the agent was still exploring heavily at the end, with final
epsilon around `0.2223`. That means many later actions were still random, so
the learned policy could not fully dominate behaviour during evaluation.

This result should therefore be interpreted as:

- slow exploration decay was too conservative for the current number of
  episodes
- a longer run may change this conclusion

### 7.3 Constant Exploration with Dense Reward

This run sits between the two decaying-dense settings.

Observations:

- better than slow-decay dense in score
- much worse than fast-decay dense
- max score of `14`, which shows some useful learning

Interpretation:

Keeping epsilon fixed at `0.2` preserves exploration permanently. That can be
useful early in training, but it prevents the agent from cleanly converging to
a stable policy. Even if the Q-table contains good action estimates, one in
five decisions remains exploratory, which limits final performance.

In practical terms, this run shows that:

- constant exploration still allows learning
- persistent randomness suppresses the best achievable late-stage performance

### 7.4 Slow Decay with Sparse Reward

This run is the sparse-reward counterpart to slow-decay dense.

Observations:

- average reward is very close to slow-decay dense
- average score is slightly higher than slow-decay dense
- average survival steps are also slightly higher
- visited states are slightly lower

Interpretation:

The sparse and dense versions are closer than expected from theory. Normally,
dense shaping should make learning easier. The fact that the difference is
small here likely means one or both of the following:

- the short training budget makes the difference harder to separate
- the current shaping values are not strong enough to create a dramatic
  advantage

This is therefore an interesting but still preliminary result. It does not
prove that reward shaping is unimportant. It only shows that, under the current
settings, exploration schedule had a stronger visible effect than the sparse
versus dense reward distinction.

## 8. What the Agent Is Actually Learning

Since food spawns randomly, the agent is not learning a fixed route. Instead,
it learns values for abstract local situations defined by the engineered state.

For example, a learned preference may emerge for a state like:

- danger straight = `0`
- danger left = `1`
- danger right = `0`
- food right = `1`

If this kind of state appears in many different board locations, the same
Q-values are reused. This is why tabular Q-learning is still possible despite
random food placement.

In qualitative terms, the successful agent is learning:

- immediate collision avoidance
- directional bias toward food
- better survival in open space

It is not learning:

- full-board strategic planning
- long-horizon trap avoidance in a large state space

Those limitations motivate the DQN phase later.

## 9. What the Current Results Suggest

From this first phase, the strongest conclusions are:

1. Tabular Q-learning works on the custom Snake environment.
2. Exploration scheduling matters a lot.
3. The best current setup is dense reward with faster epsilon decay.
4. The current evidence for reward-shaping superiority is weaker than expected.
5. The compact engineered state representation is sufficient for early learning
   on `10x10`.

## 10. Limitations of the Current Report

This report should be used carefully in the coursework because it is still a
baseline study.

Current limitations:

- only one grid size: `10x10`
- only one random seed
- only `300` episodes per run
- only one learning rate and discount factor pair
- no direct statistical averaging across repeated runs

Because of that, these results are good for methodology and early discussion,
but not yet enough for strong final comparative claims.

## 11. Recommended Next Steps

To complete the Q-learning phase more rigorously, the next experiments should
be:

1. Grid-size comparison: `6x6`, `10x10`, `15x15`
2. Multi-seed repeats for each configuration
3. Longer training for slow-decay experiments
4. Small sweep over `alpha` and `gamma`

These additions would make the tabular baseline section substantially stronger
for the final report.

## 12. Conclusion

The first Q-learning phase successfully establishes a working RL baseline for
Snake. The agent can learn meaningful behaviour using an engineered local state
representation and epsilon-greedy exploration. Among the tested settings, the
combination of dense rewards and faster epsilon decay produced by far the best
performance on the `10x10` environment.

The main practical lesson from this phase is that tabular Q-learning is viable
for a compact version of Snake, but its behaviour is highly sensitive to the
training schedule and state abstraction. This creates a strong basis for the
next project stage, where DQN can be introduced to improve scalability and
generalization.

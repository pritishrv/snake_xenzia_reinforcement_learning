# double_dqn_baseline_dense

## Hyperparameters

- learning_rate: `0.001`
- gamma: `0.99`
- epsilon_start: `1.0`
- epsilon_min: `0.05`
- epsilon_decay: `0.995`
- batch_size: `64`
- buffer_capacity: `10000`
- target_update_freq: `100`
- hidden_dims: `[128, 128]`
- reward_scheme: `dense`

## Results

- final_epsilon: `0.0816`
- average_reward_last_50: `71.162`
- average_score_last_50: `7.460`
- max_score: `21`
- average_steps_last_50: `64.50`

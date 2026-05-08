# dqn_lr_low_dense

## Hyperparameters

- learning_rate: `0.0001`
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
- average_reward_last_50: `65.992`
- average_score_last_50: `6.960`
- max_score: `20`
- average_steps_last_50: `61.52`

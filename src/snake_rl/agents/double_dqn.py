from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Any

from snake_rl.agents.dqn import DQNAgent, DQNNetwork


class DoubleDQNAgent(DQNAgent):
    """Double DQN Agent to reduce Q-value overestimation."""

    def update(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        done: bool,
    ) -> float | None:
        self.memory.push(np.array(state), action, reward, np.array(next_state), done)

        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute Q(s_t, a)
        current_q_values = self.policy_net(states).gather(1, actions)

        # Double DQN update rule:
        # 1. Use policy_net to choose the best action for next_state
        # 2. Use target_net to evaluate that action
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1).unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        # Compute MSE loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

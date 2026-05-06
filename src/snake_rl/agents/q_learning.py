from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class QLearningAgent:
    action_size: int
    alpha: float
    gamma: float
    epsilon: float
    epsilon_min: float
    epsilon_decay: float
    q_table: dict[tuple[int, ...], np.ndarray] = field(default_factory=dict)

    def select_action(self, state: tuple[int, ...]) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_size))
        return int(np.argmax(self._get_q_values(state)))

    def update(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        done: bool,
    ) -> None:
        current_q = self._get_q_values(state)
        next_q = self._get_q_values(next_state)
        target = reward if done else reward + self.gamma * float(np.max(next_q))
        current_q[action] = current_q[action] + self.alpha * (target - current_q[action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def snapshot(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "visited_states": len(self.q_table),
        }

    def _get_q_values(self, state: tuple[int, ...]) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size, dtype=np.float32)
        return self.q_table[state]

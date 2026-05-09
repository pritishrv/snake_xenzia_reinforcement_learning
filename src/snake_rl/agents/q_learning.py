from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class QLearningAgent:
    actionSize: int
    alpha: float
    gamma: float
    epsilon: float
    minEpsilon: float
    epsilonDecay: float
    q_table: dict[tuple[int, ...], np.ndarray] = field(default_factory=dict)

    def chooseAction(self, state: tuple[int, ...]) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.actionSize))
        return int(np.argmax(self.get_q_values(state)))

    def chooseGreedyAction(self, state: tuple[int, ...]) -> int:
        return int(np.argmax(self.get_q_values(state)))

    def update(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...],
        done: bool,
    ) -> None:
        current_q = self.get_q_values(state)
        next_q = self.get_q_values(next_state)
        target = reward if done else reward + self.gamma * float(np.max(next_q))
        current_q[action] = current_q[action] + self.alpha * (target - current_q[action])

    def decayEpsilon(self) -> None:
        self.epsilon = max(self.minEpsilon, self.epsilon * self.epsilonDecay)

    def snapshot(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.minEpsilon,
            "epsilon_decay": self.epsilonDecay,
            "visited_states": len(self.q_table),
        }

    def export_q_table(self) -> dict[str, list[float]]:
        return {",".join(str(bit) for bit in key): values.tolist() for key, values in self.q_table.items()}

    def get_q_values(self, state: tuple[int, ...]) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.actionSize, dtype=np.float32)
        return self.q_table[state]

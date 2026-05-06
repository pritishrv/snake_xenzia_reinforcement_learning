from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque

import numpy as np


UP = (0, -1)
RIGHT = (1, 0)
DOWN = (0, 1)
LEFT = (-1, 0)
DIRECTIONS = [UP, RIGHT, DOWN, LEFT]


@dataclass(frozen=True)
class StepResult:
    state: tuple[int, ...]
    reward: float
    done: bool
    score: int


class SnakeEnv:
    """Minimal Snake environment with engineered tabular state features."""

    def __init__(
        self,
        grid_size: int = 10,
        reward_config: dict[str, float] | None = None,
        max_steps_without_food: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.grid_size = grid_size
        self.reward_config = reward_config or {
            "food": 10.0,
            "death": -10.0,
            "step": -0.1,
            "closer_to_food": 0.3,
            "farther_from_food": -0.3,
        }
        self.max_steps_without_food = max_steps_without_food or (grid_size * grid_size)
        self.rng = random.Random(seed)
        self.snake: Deque[tuple[int, int]] = deque()
        self.direction: tuple[int, int] = RIGHT
        self.food: tuple[int, int] = (0, 0)
        self.score = 0
        self.steps_since_food = 0
        self.total_steps = 0
        self.reset()

    def reset(self) -> tuple[int, ...]:
        center = self.grid_size // 2
        self.snake = deque([(center, center), (center - 1, center), (center - 2, center)])
        self.direction = RIGHT
        self.score = 0
        self.steps_since_food = 0
        self.total_steps = 0
        self._spawn_food()
        return self.get_state()

    def step(self, action: int) -> StepResult:
        self.direction = self._next_direction(action)
        head_x, head_y = self.snake[0]
        move_x, move_y = self.direction
        new_head = (head_x + move_x, head_y + move_y)
        self.total_steps += 1
        old_distance = self._food_distance(self.snake[0])

        if self._is_collision(new_head):
            return StepResult(
                state=self.get_state(),
                reward=self.reward_config["death"],
                done=True,
                score=self.score,
            )

        self.snake.appendleft(new_head)
        reward = self.reward_config["step"]
        done = False

        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            reward += self.reward_config["food"]
            self._spawn_food()
        else:
            self.snake.pop()
            self.steps_since_food += 1
            new_distance = self._food_distance(new_head)
            if new_distance < old_distance:
                reward += self.reward_config["closer_to_food"]
            elif new_distance > old_distance:
                reward += self.reward_config["farther_from_food"]

        if self.steps_since_food >= self.max_steps_without_food:
            done = True

        return StepResult(
            state=self.get_state(),
            reward=reward,
            done=done,
            score=self.score,
        )

    def get_state(self) -> tuple[int, ...]:
        head = self.snake[0]
        straight = self.direction
        left = self._turn_left(self.direction)
        right = self._turn_right(self.direction)

        state = [
            int(self._is_collision(self._next_point(head, straight))),
            int(self._is_collision(self._next_point(head, left))),
            int(self._is_collision(self._next_point(head, right))),
            int(self.direction == UP),
            int(self.direction == DOWN),
            int(self.direction == LEFT),
            int(self.direction == RIGHT),
            int(self.food[1] < head[1]),
            int(self.food[1] > head[1]),
            int(self.food[0] < head[0]),
            int(self.food[0] > head[0]),
        ]
        return tuple(state)

    def _next_direction(self, action: int) -> tuple[int, int]:
        direction_index = DIRECTIONS.index(self.direction)
        if action == 0:
            next_index = direction_index
        elif action == 1:
            next_index = (direction_index - 1) % len(DIRECTIONS)
        elif action == 2:
            next_index = (direction_index + 1) % len(DIRECTIONS)
        else:
            raise ValueError(f"Unsupported action: {action}")
        return DIRECTIONS[next_index]

    def _is_collision(self, point: tuple[int, int]) -> bool:
        x, y = point
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        return point in self.snake

    def _spawn_food(self) -> None:
        occupied = set(self.snake)
        candidates = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in occupied
        ]
        self.food = self.rng.choice(candidates)

    @staticmethod
    def _next_point(point: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
        return point[0] + direction[0], point[1] + direction[1]

    @staticmethod
    def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
        return DIRECTIONS[(DIRECTIONS.index(direction) - 1) % len(DIRECTIONS)]

    @staticmethod
    def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
        return DIRECTIONS[(DIRECTIONS.index(direction) + 1) % len(DIRECTIONS)]

    def _food_distance(self, point: tuple[int, int]) -> int:
        return abs(point[0] - self.food[0]) + abs(point[1] - self.food[1])

    @property
    def observation_size(self) -> int:
        return 11

    @property
    def action_size(self) -> int:
        return 3

    @staticmethod
    def decode_action(action: int) -> str:
        return {0: "straight", 1: "left", 2: "right"}[action]

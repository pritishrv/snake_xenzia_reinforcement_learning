from __future__ import annotations

from typing import Any

from snake_rl.env.snake_env import SnakeEnv


class PygameSnakeRenderer:
    """Simple pygame renderer for visualizing learning episodes live."""

    def __init__(self, grid_size: int, cell_size: int = 32, fps: int = 20) -> None:
        try:
            import pygame
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pygame is not installed. Install it with `python3 -m pip install pygame`."
            ) from exc

        self.pygame = pygame
        pygame.init()
        pygame.font.init()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.fps = fps
        self.width = grid_size * cell_size
        self.height = grid_size * cell_size + 90
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake Q-Learning Live Training")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Menlo", 20)
        self.small_font = pygame.font.SysFont("Menlo", 16)

    def render(
        self,
        env: SnakeEnv,
        episode: int,
        step: int,
        reward: float,
        epsilon: float,
        mode: str,
    ) -> bool:
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill((18, 18, 18))
        self._draw_board(env)
        self._draw_stats(env, episode, step, reward, epsilon, mode)
        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def close(self) -> None:
        self.pygame.quit()

    def _draw_board(self, env: SnakeEnv) -> None:
        pygame = self.pygame
        board_height = env.grid_size * self.cell_size
        for x in range(env.grid_size):
            for y in range(env.grid_size):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                color = (38, 38, 38) if (x + y) % 2 == 0 else (32, 32, 32)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (52, 52, 52), rect, width=1)

        food_rect = pygame.Rect(
            env.food[0] * self.cell_size + 4,
            env.food[1] * self.cell_size + 4,
            self.cell_size - 8,
            self.cell_size - 8,
        )
        pygame.draw.rect(self.screen, (220, 60, 60), food_rect, border_radius=6)

        for idx, segment in enumerate(env.snake):
            rect = pygame.Rect(
                segment[0] * self.cell_size + 3,
                segment[1] * self.cell_size + 3,
                self.cell_size - 6,
                self.cell_size - 6,
            )
            color = (90, 210, 120) if idx == 0 else (60, 170, 90)
            pygame.draw.rect(self.screen, color, rect, border_radius=6)

        pygame.draw.line(self.screen, (70, 70, 70), (0, board_height), (self.width, board_height), 2)

    def _draw_stats(
        self,
        env: SnakeEnv,
        episode: int,
        step: int,
        reward: float,
        epsilon: float,
        mode: str,
    ) -> None:
        board_height = env.grid_size * self.cell_size
        lines = [
            f"Mode: {mode}",
            f"Episode: {episode}",
            f"Step: {step}",
            f"Score: {env.score}",
            f"Epsilon: {epsilon:.3f}",
            f"Reward: {reward:.2f}",
        ]
        for idx, line in enumerate(lines):
            font = self.font if idx == 0 else self.small_font
            text = font.render(line, True, (235, 235, 235))
            self.screen.blit(text, (12, board_height + 10 + idx * 12))

        hint = self.small_font.render("Close the window to stop training.", True, (180, 180, 180))
        self.screen.blit(hint, (self.width - hint.get_width() - 12, board_height + 12))

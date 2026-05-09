from typing import Any
from snake_rl.env.snake_env import SnakeEnv
import pygame


class PygameSnakeRenderer:

    def __init__(self, gridSize: int, cellSize: int = 32, fps: int = 20) -> None:
        self.pygame = pygame
        pygame.init()
        pygame.font.init()
        self.gridSize = gridSize
        self.cellSize = cellSize
        self.fps = fps
        self.width = gridSize * cellSize
        self.height = gridSize * cellSize + 90
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake Q-Learning Live Training")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Menlo", 20)
        self.smallFont = pygame.font.SysFont("Menlo", 16)

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
        self.drawBoard(env)
        self.drawStats(env, episode, step, reward, epsilon, mode)
        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def close(self) -> None:
        self.pygame.quit()

    def drawBoard(self, env: SnakeEnv) -> None:
        pygame = self.pygame
        boardHeight = env.grid_size * self.cellSize
        for x in range(env.grid_size):
            for y in range(env.grid_size):
                rect = pygame.Rect(
                    x * self.cellSize,
                    y * self.cellSize,
                    self.cellSize,
                    self.cellSize,
                )
                color = (38, 38, 38) if (x + y) % 2 == 0 else (32, 32, 32)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (52, 52, 52), rect, width=1)

        food_rect = pygame.Rect(
            env.food[0] * self.cellSize + 4,
            env.food[1] * self.cellSize + 4,
            self.cellSize - 8,
            self.cellSize - 8,
        )
        pygame.draw.rect(self.screen, (220, 60, 60), food_rect, border_radius=6)

        for idx, segment in enumerate(env.snake):
            rect = pygame.Rect(
                segment[0] * self.cellSize + 3,
                segment[1] * self.cellSize + 3,
                self.cellSize - 6,
                self.cellSize - 6,
            )
            color = (90, 210, 120) if idx == 0 else (60, 170, 90)
            pygame.draw.rect(self.screen, color, rect, border_radius=6)

        pygame.draw.line(self.screen, (70, 70, 70), (0, boardHeight), (self.width, boardHeight), 2)

    def drawStats(
        self,
        env: SnakeEnv,
        episode: int,
        step: int,
        reward: float,
        epsilon: float,
        mode: str,
    ) -> None:
        boardHeight = env.grid_size * self.cellSize
        lines = [
            f"Mode: {mode}",
            f"Episode: {episode}",
            f"Step: {step}",
            f"Score: {env.score}",
            f"Epsilon: {epsilon:.3f}",
            f"Reward: {reward:.2f}",
        ]
        for idx, line in enumerate(lines):
            font = self.font if idx == 0 else self.smallFont
            text = font.render(line, True, (235, 235, 235))
            self.screen.blit(text, (12, boardHeight + 10 + idx * 12))

        hint = self.smallFont.render("Close the window to stop training.", True, (180, 180, 180))
        self.screen.blit(hint, (self.width - hint.get_width() - 12, boardHeight + 12))

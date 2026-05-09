import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snake_rl.utils.io import dumpJSON, checkDir


def movingAverage(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    averages = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        averages.append(float(np.mean(values[start : idx + 1])))
    return averages


def saveEpisodeHistory(outputFolder: Path, history: list[dict[str, Any]]) -> None:
    checkDir(outputFolder)
    if not history:
        return
    fieldnames = list(history[0].keys())
    path = outputFolder / "episode_history.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def saveSummaryTable(outputFolder: Path, rows: list[dict[str, Any]]) -> None:
    checkDir(outputFolder)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path = outputFolder / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def plotRewards(outputFolder: Path, history: list[dict[str, Any]], window: int) -> None:
    rewards = [row["reward"] for row in history]
    smooth = movingAverage(rewards, window)
    episodes = [row["episode"] for row in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, alpha=0.35, label="Episode reward")
    ax.plot(episodes, smooth, linewidth=2, label=f"Moving average ({window})")
    ax.set_title("DQN Reward Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outputFolder / "reward_curve.png", dpi=150)
    plt.close(fig)


def plotScores(outputFolder: Path, history: list[dict[str, Any]], window: int) -> None:
    scores = [row["score"] for row in history]
    smooth = movingAverage(scores, window)
    episodes = [row["episode"] for row in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, scores, alpha=0.35, label="Episode score")
    ax.plot(episodes, smooth, linewidth=2, label=f"Moving average ({window})")
    ax.set_title("DQN Score Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outputFolder / "score_curve.png", dpi=150)
    plt.close(fig)


def plotLoss(outputFolder: Path, history: list[dict[str, Any]], window: int) -> None:
    losses = [row["loss"] for row in history if row["loss"] is not None]
    if not losses:
        return
    smooth = movingAverage(losses, window)
    episodes = list(range(len(losses)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, losses, alpha=0.35, label="Training loss")
    ax.plot(episodes, smooth, linewidth=2, label=f"Moving average ({window})")
    ax.set_yscale("log")
    ax.set_title("DQN Training Loss (Log Scale)")
    ax.set_xlabel("Update Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outputFolder / "loss_curve.png", dpi=150)
    plt.close(fig)


def saveSummary(outputFolder: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# {summary['name']}",
        "",
        "## Hyperparameters",
        "",
    ]
    for key in ["learning_rate", "gamma", "epsilon_start", "epsilon_min", "epsilon_decay", 
                "batch_size", "buffer_capacity", "target_update_freq", "eps_clip", "k_epochs", 
                "hidden_dims", "reward_scheme"]:
        if key in summary:
            lines.append(f"- {key}: `{summary[key]}`")
    
    lines.extend([
        "",
        "## Results",
        "",
        f"- average_reward_last_50: `{summary['average_reward_last_50']:.3f}`",
        f"- average_score_last_50: `{summary['average_score_last_50']:.3f}`",
        f"- max_score: `{summary['max_score']}`",
        f"- average_steps_last_50: `{summary['average_steps_last_50']:.2f}`",
    ])
    if "final_epsilon" in summary:
        lines.append(f"- final_epsilon: `{summary['final_epsilon']:.4f}`")
        
    (outputFolder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

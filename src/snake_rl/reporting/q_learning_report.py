from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from snake_rl.utils.io import dump_json, ensure_dir


def moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    averages = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        averages.append(float(np.mean(values[start : idx + 1])))
    return averages


def save_episode_history(output_dir: Path, history: list[dict[str, Any]]) -> None:
    ensure_dir(output_dir)
    fieldnames = ["episode", "reward", "score", "steps", "epsilon"]
    path = output_dir / "episode_history.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_summary_table(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(output_dir)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path = output_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_run_metadata(output_dir: Path, payload: dict[str, Any]) -> None:
    dump_json(output_dir / "metadata.json", payload)


def plot_rewards(output_dir: Path, history: list[dict[str, Any]], window: int) -> None:
    rewards = [row["reward"] for row in history]
    smooth = moving_average(rewards, window)
    episodes = [row["episode"] for row in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, alpha=0.35, label="Episode reward")
    ax.plot(episodes, smooth, linewidth=2, label=f"Moving average ({window})")
    ax.set_title("Q-learning Reward Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "reward_curve.png", dpi=150)
    plt.close(fig)


def plot_scores(output_dir: Path, history: list[dict[str, Any]], window: int) -> None:
    scores = [row["score"] for row in history]
    smooth = moving_average(scores, window)
    episodes = [row["episode"] for row in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, scores, alpha=0.35, label="Episode score")
    ax.plot(episodes, smooth, linewidth=2, label=f"Moving average ({window})")
    ax.set_title("Q-learning Score Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "score_curve.png", dpi=150)
    plt.close(fig)


def write_markdown_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# {summary['name']}",
        "",
        "## Hyperparameters",
        "",
        f"- alpha: `{summary['alpha']}`",
        f"- gamma: `{summary['gamma']}`",
        f"- epsilon_start: `{summary['epsilon_start']}`",
        f"- epsilon_min: `{summary['epsilon_min']}`",
        f"- epsilon_decay: `{summary['epsilon_decay']}`",
        f"- reward_scheme: `{summary['reward_scheme']}`",
        "",
        "## Results",
        "",
        f"- final_epsilon: `{summary['final_epsilon']:.4f}`",
        f"- average_reward_last_50: `{summary['average_reward_last_50']:.3f}`",
        f"- average_score_last_50: `{summary['average_score_last_50']:.3f}`",
        f"- max_score: `{summary['max_score']}`",
        f"- average_steps_last_50: `{summary['average_steps_last_50']:.2f}`",
        f"- visited_states: `{summary['visited_states']}`",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snake_rl.agents.q_learning import QLearningAgent
from snake_rl.env.snake_env import SnakeEnv
from snake_rl.reporting.q_learning_report import (
    plot_rewards,
    plot_scores,
    save_episode_history,
    save_run_metadata,
    save_summary_table,
    write_markdown_summary,
)
from snake_rl.utils.io import PROJECT_ROOT as ROOT, ensure_dir, load_json
from snake_rl.utils.randomness import set_global_seed


def train_single_run(base_config: dict, experiment: dict) -> dict:
    reward_scheme_name = experiment["reward_scheme"]
    reward_config = base_config["reward_schemes"][reward_scheme_name]

    env = SnakeEnv(
        grid_size=base_config["grid_size"],
        reward_config=reward_config,
        max_steps_without_food=base_config["max_steps_per_episode"],
        seed=base_config["seed"],
    )
    agent = QLearningAgent(
        action_size=env.action_size,
        alpha=experiment["alpha"],
        gamma=experiment["gamma"],
        epsilon=experiment["epsilon_start"],
        epsilon_min=experiment["epsilon_min"],
        epsilon_decay=experiment["epsilon_decay"],
    )

    history = []
    for episode in range(1, base_config["episodes"] + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < base_config["max_steps_per_episode"]:
            action = agent.select_action(state)
            result = env.step(action)
            agent.update(state, action, result.reward, result.state, result.done)
            state = result.state
            total_reward += result.reward
            done = result.done
            steps += 1

        history.append(
            {
                "episode": episode,
                "reward": round(total_reward, 6),
                "score": env.score,
                "steps": steps,
                "epsilon": round(agent.epsilon, 6),
            }
        )
        agent.decay_epsilon()

    reward_tail = [row["reward"] for row in history[-50:]]
    score_tail = [row["score"] for row in history[-50:]]
    step_tail = [row["steps"] for row in history[-50:]]

    summary = {
        "name": experiment["name"],
        "alpha": experiment["alpha"],
        "gamma": experiment["gamma"],
        "epsilon_start": experiment["epsilon_start"],
        "epsilon_min": experiment["epsilon_min"],
        "epsilon_decay": experiment["epsilon_decay"],
        "reward_scheme": reward_scheme_name,
        "final_epsilon": agent.epsilon,
        "average_reward_last_50": mean(reward_tail),
        "average_score_last_50": mean(score_tail),
        "average_steps_last_50": mean(step_tail),
        "max_score": max(row["score"] for row in history),
        "visited_states": len(agent.q_table),
    }
    return {"history": history, "summary": summary, "agent": agent}


def main() -> None:
    config = load_json(ROOT / "configs" / "q_learning_base.json")
    set_global_seed(config["seed"])

    output_root = ensure_dir(ROOT / "outputs" / "q_learning")
    all_summaries = []

    for experiment in config["experiments"]:
        print(f"Running experiment: {experiment['name']}")
        run_output_dir = ensure_dir(output_root / experiment["name"])
        result = train_single_run(config, experiment)

        save_episode_history(run_output_dir, result["history"])
        save_run_metadata(
            run_output_dir,
            {
                "base_config": config,
                "experiment": experiment,
                "agent_snapshot": result["agent"].snapshot(),
            },
        )
        plot_rewards(run_output_dir, result["history"], config["moving_average_window"])
        plot_scores(run_output_dir, result["history"], config["moving_average_window"])
        write_markdown_summary(run_output_dir, result["summary"])
        all_summaries.append(result["summary"])

    save_summary_table(output_root, all_summaries)
    print(f"Completed {len(all_summaries)} Q-learning experiments.")
    print(f"Outputs written to: {output_root}")


if __name__ == "__main__":
    main()

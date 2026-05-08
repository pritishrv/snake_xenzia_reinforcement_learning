from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snake_rl.agents.dqn import DQNAgent
from snake_rl.agents.double_dqn import DoubleDQNAgent
from snake_rl.env.snake_env import SnakeEnv
from snake_rl.reporting.dqn_report import (
    plot_loss,
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent_type = experiment.get("agent_type", "dqn")
    agent_cls = DoubleDQNAgent if agent_type == "double_dqn" else DQNAgent
    
    agent = agent_cls(
        state_size=env.observation_size,
        action_size=env.action_size,
        hidden_dims=experiment["hidden_dims"],
        learning_rate=experiment["learning_rate"],
        gamma=experiment["gamma"],
        epsilon=experiment["epsilon_start"],
        epsilon_min=experiment["epsilon_min"],
        epsilon_decay=experiment["epsilon_decay"],
        batch_size=experiment["batch_size"],
        buffer_capacity=experiment["buffer_capacity"],
        target_update_freq=experiment["target_update_freq"],
        device=device,
    )

    history = []
    losses = []
    
    for episode in range(1, base_config["episodes"] + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        episode_loss = []

        while not done and steps < base_config["max_steps_per_episode"]:
            action = agent.select_action(state)
            result = env.step(action)
            loss = agent.update(state, action, result.reward, result.state, result.done)
            if loss is not None:
                episode_loss.append(loss)
                losses.append(loss)
            
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
                "loss": mean(episode_loss) if episode_loss else None,
            }
        )
        agent.decay_epsilon()
        
        if episode % 50 == 0:
            avg_reward = mean([row["reward"] for row in history[-50:]])
            avg_score = mean([row["score"] for row in history[-50:]])
            print(f"Episode {episode} | Avg Reward: {avg_reward:.2f} | Avg Score: {avg_score:.2f} | Epsilon: {agent.epsilon:.3f}")

    reward_tail = [row["reward"] for row in history[-50:]]
    score_tail = [row["score"] for row in history[-50:]]
    step_tail = [row["steps"] for row in history[-50:]]

    summary = {
        "name": experiment["name"],
        "learning_rate": experiment["learning_rate"],
        "gamma": experiment["gamma"],
        "epsilon_start": experiment["epsilon_start"],
        "epsilon_min": experiment["epsilon_min"],
        "epsilon_decay": experiment["epsilon_decay"],
        "batch_size": experiment["batch_size"],
        "buffer_capacity": experiment["buffer_capacity"],
        "target_update_freq": experiment["target_update_freq"],
        "hidden_dims": experiment["hidden_dims"],
        "reward_scheme": reward_scheme_name,
        "final_epsilon": agent.epsilon,
        "average_reward_last_50": mean(reward_tail),
        "average_score_last_50": mean(score_tail),
        "average_steps_last_50": mean(step_tail),
        "max_score": max(row["score"] for row in history),
    }
    
    # Create a history object for losses specifically for plotting
    loss_history = [{"loss": l} for l in losses]
    
    return {"history": history, "loss_history": loss_history, "summary": summary, "agent": agent}


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "dqn_base.json"
    config_path = ROOT / "configs" / config_name
    print(f"Loading config from: {config_path}")
    config = load_json(config_path)
    set_global_seed(config["seed"])

    output_root = ensure_dir(ROOT / "outputs" / config_name.replace(".json", ""))
    all_summaries = []

    for experiment in config["experiments"]:
        print(f"\nRunning experiment: {experiment['name']}")
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
        plot_loss(run_output_dir, result["loss_history"], config["moving_average_window"] * 10) # More updates than episodes
        write_markdown_summary(run_output_dir, result["summary"])
        all_summaries.append(result["summary"])

    save_summary_table(output_root, all_summaries)
    print(f"\nCompleted {len(all_summaries)} DQN/Double DQN experiments.")
    print(f"Outputs written to: {output_root}")


if __name__ == "__main__":
    main()

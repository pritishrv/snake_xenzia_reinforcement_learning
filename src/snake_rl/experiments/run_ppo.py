from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snake_rl.agents.ppo import PPOAgent
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
    
    agent = PPOAgent(
        state_size=env.observation_size,
        action_size=env.action_size,
        hidden_dims=experiment["hidden_dims"],
        learning_rate=experiment["learning_rate"],
        gamma=experiment["gamma"],
        eps_clip=experiment["eps_clip"],
        k_epochs=experiment["k_epochs"],
        device=device,
    )

    history = []
    losses = []
    time_step = 0
    update_timestep = base_config["update_timestep"]
    
    for episode in range(1, base_config["episodes"] + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        
        while not done and steps < base_config["max_steps_per_episode"]:
            time_step += 1
            action = agent.select_action(state)
            result = env.step(action)
            
            agent.store_transition(result.reward, result.done)
            
            # Update PPO agent
            if time_step % update_timestep == 0:
                loss = agent.update()
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
                "loss": losses[-1] if losses else None,
            }
        )
        
        if episode % 50 == 0:
            avg_reward = mean([row["reward"] for row in history[-50:]])
            avg_score = mean([row["score"] for row in history[-50:]])
            print(f"Episode {episode} | Avg Reward: {avg_reward:.2f} | Avg Score: {avg_score:.2f}")

    reward_tail = [row["reward"] for row in history[-50:]]
    score_tail = [row["score"] for row in history[-50:]]
    step_tail = [row["steps"] for row in history[-50:]]

    summary = {
        "name": experiment["name"],
        "learning_rate": experiment["learning_rate"],
        "gamma": experiment["gamma"],
        "eps_clip": experiment["eps_clip"],
        "k_epochs": experiment["k_epochs"],
        "hidden_dims": experiment["hidden_dims"],
        "reward_scheme": reward_scheme_name,
        "average_reward_last_50": mean(reward_tail),
        "average_score_last_50": mean(score_tail),
        "average_steps_last_50": mean(step_tail),
        "max_score": max(row["score"] for row in history),
    }
    
    loss_history = [{"loss": l} for l in losses]
    
    return {"history": history, "loss_history": loss_history, "summary": summary, "agent": agent}


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "ppo_base.json"
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
        plot_loss(run_output_dir, result["loss_history"], 10)
        write_markdown_summary(run_output_dir, result["summary"])
        all_summaries.append(result["summary"])

    save_summary_table(output_root, all_summaries)
    print(f"\nCompleted {len(all_summaries)} PPO experiments.")
    print(f"Outputs written to: {output_root}")


if __name__ == "__main__":
    main()

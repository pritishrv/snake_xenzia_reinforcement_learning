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
    plotLoss,
    plotRewards,
    plotScores,
    saveEpisodeHistory,
    saveSummaryTable,
    saveSummary,
)
from snake_rl.utils.io import PROJECT_ROOT as ROOT, checkDir, loadJSON
from snake_rl.utils.randomness import setGlobalSeed


def train_single_run(base_config: dict, experiment: dict) -> dict:
    rewardScheme = experiment["reward_scheme"]
    rewardConfig = base_config["reward_schemes"][rewardScheme]

    env = SnakeEnv(
        gridSize=base_config["grid_size"],
        rewardConfig=rewardConfig,
        maxStepsNoFood=base_config["max_steps_per_episode"],
        seed=base_config["seed"],
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = PPOAgent(
        stateSize=env.observation_size,
        actionSize=env.action_size,
        hiddenDims=experiment["hidden_dims"],
        learningRate=experiment["learning_rate"],
        gamma=experiment["gamma"],
        epsClip=experiment["eps_clip"],
        k_epochs=experiment["k_epochs"],
        device=device,
    )

    history = []
    losses = []
    timeStep = 0
    updateTimestep = base_config["update_timestep"]
    
    for episode in range(1, base_config["episodes"] + 1):
        state = env.reset()
        totalReward = 0.0
        done = False
        steps = 0
        
        while not done and steps < base_config["max_steps_per_episode"]:
            timeStep += 1
            action = agent.select_action(state)
            result = env.step(action)
            
            agent.storeTransition(result.reward, result.done)
            
            # Update PPO agent
            if timeStep % updateTimestep == 0:
                loss = agent.update()
                losses.append(loss)
            
            state = result.state
            totalReward += result.reward
            done = result.done
            steps += 1

        history.append(
            {
                "episode": episode,
                "reward": round(totalReward, 6),
                "score": env.score,
                "steps": steps,
                "loss": losses[-1] if losses else None,
            }
        )
        
        if episode % 50 == 0:
            avgReward = mean([row["reward"] for row in history[-50:]])
            avgScore = mean([row["score"] for row in history[-50:]])
            print(f"Episode {episode} | Avg Reward: {avgReward:.2f} | Avg Score: {avgScore:.2f}")

    rewardTail = [row["reward"] for row in history[-50:]]
    scoreTail = [row["score"] for row in history[-50:]]
    stepTail = [row["steps"] for row in history[-50:]]

    summary = {
        "name": experiment["name"],
        "learning_rate": experiment["learning_rate"],
        "gamma": experiment["gamma"],
        "eps_clip": experiment["eps_clip"],
        "k_epochs": experiment["k_epochs"],
        "hidden_dims": experiment["hidden_dims"],
        "reward_scheme": rewardScheme,
        "average_reward_last_50": mean(rewardTail),
        "average_score_last_50": mean(scoreTail),
        "average_steps_last_50": mean(stepTail),
        "max_score": max(row["score"] for row in history),
    }
    
    lossHistory = [{"loss": l} for l in losses]
    
    return {"history": history, "loss_history": lossHistory, "summary": summary, "agent": agent}


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "ppo_base.json"
    config_path = ROOT / "configs" / config_name
    print(f"Loading config from: {config_path}")
    config = loadJSON(config_path)
    setGlobalSeed(config["seed"])

    outputFolderRoot = checkDir(ROOT / "outputs" / config_name.replace(".json", ""))
    summaries = []

    for experiment in config["experiments"]:
        print(f"\nRunning experiment: {experiment['name']}")
        outputFolder = checkDir(outputFolderRoot / experiment["name"])
        result = train_single_run(config, experiment)

        saveEpisodeHistory(outputFolder, result["history"])
        plotRewards(outputFolder, result["history"], config["moving_average_window"])
        plotScores(outputFolder, result["history"], config["moving_average_window"])
        plotLoss(outputFolder, result["loss_history"], 10)
        saveSummary(outputFolder, result["summary"])
        summaries.append(result["summary"])

    saveSummaryTable(outputFolderRoot, summaries)
    print(f"\nCompleted {len(summaries)} PPO experiments.")
    print(f"Outputs written to: {outputFolderRoot}")


if __name__ == "__main__":
    main()

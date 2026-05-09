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
    plotLoss,
    plotRewards,
    plotScores,
    saveEpisodeHistory,
    saveSummaryTable,
    saveSummary,
)
from snake_rl.utils.io import PROJECT_ROOT as ROOT, checkDir, loadJSON
from snake_rl.utils.randomness import setGlobalSeed


def singleTrainRun(baseConfig: dict, experiment: dict) -> dict:
    rewardScheme = experiment["reward_scheme"]
    rewardConfig = baseConfig["reward_schemes"][rewardScheme]

    env = SnakeEnv(
        gridSize=baseConfig["grid_size"],
        rewardConfig=rewardConfig,
        maxStepsNoFood=baseConfig["max_steps_per_episode"],
        seed=baseConfig["seed"],
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agentType = experiment.get("agent_type", "dqn")
    agentCLS = DoubleDQNAgent if agentType == "double_dqn" else DQNAgent
    
    agent = agentCLS(
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
    
    for episode in range(1, baseConfig["episodes"] + 1):
        state = env.reset()
        totalReward = 0.0
        done = False
        steps = 0
        loss = []

        while not done and steps < baseConfig["max_steps_per_episode"]:
            action = agent.chooseAction(state)
            result = env.step(action)
            loss = agent.update(state, action, result.reward, result.state, result.done)
            if loss is not None:
                loss.append(loss)
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
                "epsilon": round(agent.epsilon, 6),
                "loss": mean(loss) if loss else None,
            }
        )
        agent.decay_epsilon()
        
        if episode % 50 == 0:
            avgReward = mean([row["reward"] for row in history[-50:]])
            avgScore = mean([row["score"] for row in history[-50:]])
            print(f"Episode {episode} | Avg Reward: {avgReward:.2f} | Avg Score: {avgScore:.2f} | Epsilon: {agent.epsilon:.3f}")

    rewardTail = [row["reward"] for row in history[-50:]]
    scoreTail = [row["score"] for row in history[-50:]]
    stepTail = [row["steps"] for row in history[-50:]]

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
        "reward_scheme": rewardScheme,
        "final_epsilon": agent.epsilon,
        "average_reward_last_50": mean(rewardTail),
        "average_score_last_50": mean(scoreTail),
        "average_steps_last_50": mean(stepTail),
        "max_score": max(row["score"] for row in history),
    }
    
    # Create a history object for losses specifically for plotting
    lossHistory = [{"loss": l} for l in losses]
    
    return {"history": history, "loss_history": lossHistory, "summary": summary, "agent": agent}


def main() -> None:
    configName = sys.argv[1] if len(sys.argv) > 1 else "dqn_base.json"
    configPath = ROOT / "configs" / configName
    print(f"Loading config from: {configPath}")
    config = loadJSON(configPath)
    setGlobalSeed(config["seed"])

    outputFolderRoot = checkDir(ROOT / "outputs" / configName.replace(".json", ""))
    summaries = []

    for experiment in config["experiments"]:
        print(f"\nRunning experiment: {experiment['name']}")
        outputFolder = checkDir(outputFolderRoot / experiment["name"])
        result = singleTrainRun(config, experiment)

        saveEpisodeHistory(outputFolder, result["history"])
        plotRewards(outputFolder, result["history"], config["moving_average_window"])
        plotScores(outputFolder, result["history"], config["moving_average_window"])
        plotLoss(outputFolder, result["loss_history"], config["moving_average_window"] * 10) # More updates than episodes
        saveSummary(outputFolder, result["summary"])
        summaries.append(result["summary"])

    saveSummaryTable(outputFolderRoot, summaries)
    print(f"\nCompleted {len(summaries)} DQN/Double DQN experiments.")
    print(f"Outputs written to: {outputFolderRoot}")


if __name__ == "__main__":
    main()

from collections.abc import Callable
from statistics import mean
from typing import Any

from snake_rl.agents.q_learning import QLearningAgent
from snake_rl.env.snake_env import SnakeEnv


RenderCallback = Callable[[SnakeEnv, QLearningAgent, int, int, float], bool]


def singleTrainRun(
    config: dict[str, Any],
    experiment: dict[str, Any],
    renderCallback: RenderCallback | None = None,
) -> dict[str, Any]:
    reward_scheme_name = experiment["reward_scheme"]
    reward_config = config["reward_schemes"][reward_scheme_name]

    env = SnakeEnv(
        gridSize=config["grid_size"],
        rewardConfig=reward_config,
        maxStepsNoFood=config["max_steps_per_episode"],
        seed=config["seed"],
    )
    agent = QLearningAgent(
        actionSize=env.action_size,
        alpha=experiment["alpha"],
        gamma=experiment["gamma"],
        epsilon=experiment["epsilon_start"],
        minEpsilon=experiment["epsilon_min"],
        epsilonDecay=experiment["epsilon_decay"],
    )

    history = []
    stopped_early = False
    for episode in range(1, config["episodes"] + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < config["max_steps_per_episode"]:
            action = agent.chooseAction(state)
            result = env.step(action)
            agent.update(state, action, result.reward, result.state, result.done)
            state = result.state
            total_reward += result.reward
            done = result.done
            steps += 1

            if renderCallback is not None:
                keep_running = renderCallback(env, agent, episode, steps, total_reward)
                if not keep_running:
                    stopped_early = True
                    done = True
                    break

        history.append(
            {
                "episode": episode,
                "reward": round(total_reward, 6),
                "score": env.score,
                "steps": steps,
                "epsilon": round(agent.epsilon, 6),
            }
        )
        agent.decayEpsilon()

        if stopped_early:
            break

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
        "episodes_completed": len(history),
        "stopped_early": stopped_early,
    }
    return {"history": history, "summary": summary, "agent": agent}

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snake_rl.agents.q_learning import QLearningAgent
from snake_rl.env.snake_env import SnakeEnv
from snake_rl.experiments.q_learning_training import singleTrainRun
from snake_rl.reporting.pygame_renderer import PygameSnakeRenderer
from snake_rl.utils.io import PROJECT_ROOT as ROOT, loadJSON
from snake_rl.utils.randomness import setGlobalSeed


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch Q-learning train live with pygame.")
    parser.add_argument(
        "--experiment",
        default="epsilon_fast_decay_dense",
        help="Experiment name from configs/q_learning_base.json",
    )
    parser.add_argument("--episodes", type=int, default=150, help="Override episode count for the visual run.")
    parser.add_argument("--fps", type=int, default=18, help="Render frames per second.")
    parser.add_argument("--cell-size", type=int, default=36, help="Grid cell size in pixels.")
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Render every N episodes. Use larger values to speed up training.",
    )
    parser.add_argument(
        "--playback-episodes",
        type=int,
        default=3,
        help="After training, run greedy-policy playback for this many episodes.",
    )
    return parser.parse_args()


def getExperiment(config: dict, name: str) -> dict:
    for experiment in config["experiments"]:
        if experiment["name"] == name:
            return experiment
    raise ValueError(f"Unknown experiment '{name}'.")


def main() -> None:
    args = parseArgs()
    config = loadJSON(ROOT / "configs" / "q_learning_base.json")
    experiment = getExperiment(config, args.experiment)
    visualConfig = dict(config)
    visualConfig["episodes"] = args.episodes

    setGlobalSeed(visualConfig["seed"])
    renderer = PygameSnakeRenderer(gridSize=visualConfig["grid_size"], cellSize=args.cell_size, fps=args.fps)

    try:
        def render_callback(env: SnakeEnv, agent: QLearningAgent, episode: int, step: int, reward: float) -> bool:
            if episode % max(1, args.render_every) != 0:
                return True
            return renderer.render(
                env=env,
                episode=episode,
                step=step,
                reward=reward,
                epsilon=agent.epsilon,
                mode="training",
            )

        result = singleTrainRun(visualConfig, experiment, renderCallback=render_callback)
        print("Training finished.")
        print(result["summary"])

        reward_scheme = config["reward_schemes"][experiment["reward_scheme"]]
        eval_env = SnakeEnv(
            gridSize=config["grid_size"],
            rewardConfig=reward_scheme,
            maxStepsNoFood=config["max_steps_per_episode"],
            seed=config["seed"],
        )
        for episode in range(1, args.playback_episodes + 1):
            state = eval_env.reset()
            done = False
            total_reward = 0.0
            step = 0
            while not done and step < config["max_steps_per_episode"]:
                action = result["agent"].select_greedy_action(state)
                outcome = eval_env.step(action)
                state = outcome.state
                total_reward += outcome.reward
                done = outcome.done
                step += 1
                keep_running = renderer.render(
                    env=eval_env,
                    episode=episode,
                    step=step,
                    reward=total_reward,
                    epsilon=result["agent"].epsilon,
                    mode="greedy playback",
                )
                if not keep_running:
                    return
        print("Playback finished.")
    finally:
        renderer.close()


if __name__ == "__main__":
    main()

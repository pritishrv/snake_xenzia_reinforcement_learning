from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snake_rl.experiments.q_learning_training import train_single_run
from snake_rl.reporting.q_learning_report import (
    plot_rewards,
    plot_scores,
    save_episode_history,
    save_q_table,
    save_run_metadata,
    save_summary_table,
    write_markdown_summary,
)
from snake_rl.utils.io import PROJECT_ROOT as ROOT, ensure_dir, load_json
from snake_rl.utils.randomness import set_global_seed


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
        save_q_table(run_output_dir, result["agent"].export_q_table())
        plot_rewards(run_output_dir, result["history"], config["moving_average_window"])
        plot_scores(run_output_dir, result["history"], config["moving_average_window"])
        write_markdown_summary(run_output_dir, result["summary"])
        all_summaries.append(result["summary"])

    save_summary_table(output_root, all_summaries)
    print(f"Completed {len(all_summaries)} Q-learning experiments.")
    print(f"Outputs written to: {output_root}")


if __name__ == "__main__":
    main()

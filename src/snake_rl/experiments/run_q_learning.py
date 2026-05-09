import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from snake_rl.experiments.q_learning_training import singleTrainRun
from snake_rl.reporting.q_learning_report import (
    plotRewards,
    plotScores,
    saveEpisodeHistory,
    saveSummaryTable,
    saveSummary,
)
from snake_rl.utils.io import PROJECT_ROOT as ROOT, checkDir, loadJSON
from snake_rl.utils.randomness import setGlobalSeed


def main() -> None:
    config = loadJSON(ROOT / "configs" / "q_learning_base.json")
    setGlobalSeed(config["seed"])

    outputFolderRoot = checkDir(ROOT / "outputs" / "q_learning")
    summaries = []

    for experiment in config["experiments"]:
        print(f"Running experiment: {experiment['name']}")
        outputFolder = checkDir(outputFolderRoot / experiment["name"])
        result = singleTrainRun(config, experiment)

        saveEpisodeHistory(outputFolder, result["history"])
        plotRewards(outputFolder, result["history"], config["moving_average_window"])
        plotScores(outputFolder, result["history"], config["moving_average_window"])
        saveSummary(outputFolder, result["summary"])
        summaries.append(result["summary"])

    saveSummaryTable(outputFolderRoot, summaries)
    print(f"Completed {len(summaries)} Q-learning experiments.")
    print(f"Outputs written to: {outputFolderRoot}")


if __name__ == "__main__":
    main()

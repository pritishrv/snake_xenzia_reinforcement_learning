import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def checkDir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def loadJSON(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dumpJSON(path: Path, payload: dict[str, Any]) -> None:
    checkDir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

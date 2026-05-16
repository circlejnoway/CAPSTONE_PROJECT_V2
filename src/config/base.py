from pathlib import Path
import yaml


def load_settings(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

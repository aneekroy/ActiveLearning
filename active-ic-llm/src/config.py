"""Configuration loading for ActiveLLM."""

from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class Config(dict):
    """Simple dict-like access via attributes."""

    def __getattr__(self, item):
        return self[item]


with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = Config(yaml.safe_load(f))

"""Dataset loading utilities."""

from pathlib import Path
from typing import List
import json
import pandas as pd

from .config import cfg


class CrossFitDataset:
    def __init__(self, task: str, split: str):
        data_dir = Path(cfg.standardized_dir)
        file_path = data_dir / f"{task}_{split}.csv"
        self.df = pd.read_csv(file_path)
        self.task = task

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        if "question" in self.df.columns:
            return row["question"], json.loads(row["choices"]), row["label"]
        return row["text"], row["label"]

    def get_all_texts(self) -> List[str]:
        if "text" in self.df.columns:
            return self.df["text"].tolist()
        return self.df["question"].tolist()

    def get_all_labels(self) -> List[str]:
        return self.df["label"].astype(str).tolist()

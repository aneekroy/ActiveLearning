"""Standardise CrossFit datasets into CSV pool/test splits."""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import random


def _read_jsonl(jsonl_path: Path):
    return [json.loads(line) for line in jsonl_path.read_text().splitlines()]


def prepare_classification(records) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(len(records)),
            "text": [r.get("text", r.get("question", "")) for r in records],
            "label": [r.get("label", r.get("answer")) for r in records],
        }
    )


def prepare_multichoice(records) -> pd.DataFrame:
    questions = []
    choices = []
    labels = []
    for r in records:
        questions.append(
            r.get("question")
            or r.get("ctx")
            or f"{r.get('ctx_a','')}{r.get('ctx_b','')}"
        )
        ch = r.get("choices") or r.get("endings") or r.get("options")
        choices.append(json.dumps(ch))
        labels.append(r.get("label"))
    return pd.DataFrame({"id": range(len(records)), "question": questions, "choices": choices, "label": labels})


def split_and_save(df: pd.DataFrame, out_prefix: Path, seed: int = 42, test_size: float = 0.2) -> None:
    df_shuf = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pool, test = train_test_split(df_shuf, test_size=test_size, random_state=seed)
    pool.to_csv(f"{out_prefix}_pool.csv", index=False)
    test.to_csv(f"{out_prefix}_test.csv", index=False)


def process_task(task_dir: Path, out_dir: Path, seed: int = 42) -> None:
    for split_file in task_dir.glob("*.jsonl"):
        records = _read_jsonl(split_file)
        if any("choices" in r or "endings" in r for r in records):
            df = prepare_multichoice(records)
        else:
            df = prepare_classification(records)
        out_prefix = out_dir / f"{task_dir.name}"
        split_and_save(df, out_prefix, seed=seed)
        break  # assume single split file per task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/standardized")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for category_dir in raw_dir.iterdir():
        if not category_dir.is_dir():
            continue
        for task_dir in category_dir.iterdir():
            if task_dir.is_dir():
                process_task(task_dir, out_dir)


if __name__ == "__main__":
    main()

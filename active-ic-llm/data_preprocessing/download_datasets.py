"""Download CrossFit datasets from HuggingFace and save to JSONL."""

import argparse
from datasets import load_dataset
from pathlib import Path


CATEGORIES = {
    "classification": [
        "sst2",
    ],
    "multichoice": [
        "hellaswag",
    ],
}


def save_split(ds_split, out_path: Path) -> None:
    """Write a dataset split to JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_split.to_json(str(out_path))


def download_task(task_name, out_dir):
    ds = load_dataset(task_name)
    for split, dset in ds.items():
        save_split(dset, Path(out_dir) / task_name / f"{split}.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/raw")
    args = parser.parse_args()

    for cat, tasks in CATEGORIES.items():
        for task in tasks:
            download_task(task, Path(args.out_dir) / cat)


if __name__ == "__main__":
    main()

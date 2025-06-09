"""Download CrossFit datasets from HuggingFace and save to JSONL."""

import argparse
from datasets import load_dataset
from pathlib import Path


CATEGORIES = {
    "classification": [
        "sst2",
        "boolq",
    ],
    "multichoice": [
        "hellaswag",
        ("allenai/ai2_arc", "ARC-Challenge"),  # specify config explicitly
        ("allenai/ai2_arc", "ARC-Easy"),
        ("winogrande", "winogrande_xl"),  # choose appropriate config
        "piqa",
        "social_i_qa",  # corrected 'siqa' to 'social_i_qa'
        "openbookqa",   # corrected 'obqa' to 'openbookqa'
    ],
    "math_reasoning": [
        ("gsm8k", "main"),       # specify config explicitly
        "aqua_rat",
        "AddSub",                # corrected 'addsub' to 'AddSub'
        "MultiArith",            # corrected 'multiarith' to 'MultiArith'
        "SingleEq",              # corrected 'singleeq' to 'SingleEq'
        "SVAMP",                 # corrected 'svamp' to 'SVAMP'
    ],
    "instruction_following": [
        "databricks/databricks-dolly-15k",  # corrected 'dolly_eval'
        "lmsys/vicuna-eval",                # corrected 'vicuna_eval'
        "yizhongw/self_instruct",           # corrected 'self_instruct'
        "super_ni",                         # corrected 's_ni'
        "ultrachat",                        # corrected 'un_ni'
    ],
}


def save_split(ds_split, out_path: Path) -> None:
    """Write a dataset split to JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_split.to_json(str(out_path))


def is_dataset_downloaded(task_name, out_dir):
    """Check if at least one split file exists for the dataset."""
    dataset_dir = Path(out_dir) / task_name
    if not dataset_dir.exists():
        return False
    # Check for at least one .jsonl file (split)
    return any(dataset_dir.glob("*.jsonl"))


def download_task(task_name, out_dir):
    if is_dataset_downloaded(task_name, out_dir):
        print(f"Skipping {task_name}: already downloaded.")
        return
    try:
        ds = load_dataset(task_name)
        for split, dset in ds.items():
            save_split(dset, Path(out_dir) / task_name / f"{split}.jsonl")
        print(f"Downloaded {task_name} successfully.")
    except Exception as e:
        print(f"Failed to download {task_name}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/raw")
    args = parser.parse_args()

    for cat, tasks in CATEGORIES.items():
        for task in tasks:
            download_task(task, Path(args.out_dir) / cat)


if __name__ == "__main__":
    main()

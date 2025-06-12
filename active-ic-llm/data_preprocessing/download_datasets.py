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
        ("allenai/ai2_arc", "ARC-Challenge"),
        ("allenai/ai2_arc", "ARC-Easy"),
        ("winogrande", "winogrande_xl"),
        "piqa",
        "social_i_qa",
        "openbookqa", 
    ],
    "math_reasoning": [
        ("gsm8k", "main"),
        ("openai/gsm8k","socratic"),
        "aqua_rat",
        "allenai/lila", #AddSub
        "ChilleD/MultiArith",
        # "SingleEq", 
        # "SVAMP",
        # "deepmind/math_dataset"
    ],
    "instruction_following": [
        "databricks/databricks-dolly-15k",
        # "lmsys/vicuna-eval",
        "yizhongw/self_instruct",
        "Dynosaur/dynosaur-sub-superni",
        "HuggingFaceH4/ultrachat_200k",
    ],
}

def save_split(ds_split, out_path: Path) -> None:
    """Write a dataset split to JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_split.to_json(str(out_path))

def get_task_folder_name(task):
    """Get the folder name for a task, handling both strings and tuples."""
    if isinstance(task, tuple):
        dataset_name, config_name = task
        # Use dataset name + config name for folder
        return f"{dataset_name.replace('/', '_')}_{config_name}"
    else:
        return task.replace('/', '_')

def is_dataset_downloaded(task, out_dir):
    """Check if at least one split file exists for the dataset."""
    folder_name = get_task_folder_name(task)
    dataset_dir = Path(out_dir) / folder_name
    if not dataset_dir.exists():
        return False
    # Check for at least one .jsonl file (split)
    return any(dataset_dir.glob("*.jsonl"))

def download_task(task, out_dir):
    folder_name = get_task_folder_name(task)
    
    if is_dataset_downloaded(task, out_dir):
        print(f"Skipping {folder_name}: already downloaded.")
        return
    
    try:
        if isinstance(task, tuple):
            dataset_name, config_name = task
            ds = load_dataset(dataset_name, config_name)
        else:
            ds = load_dataset(task)
            
        for split, dset in ds.items():
            save_split(dset, Path(out_dir) / folder_name / f"{split}.jsonl")
        print(f"Downloaded {folder_name} successfully.")
    except Exception as e:
        print(f"Failed to download {folder_name}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/raw")
    args = parser.parse_args()
    
    for cat, tasks in CATEGORIES.items():
        for task in tasks:
            download_task(task, Path(args.out_dir) / cat)

if __name__ == "__main__":
    main()
"""DDP variant of run_experiment."""

import argparse
import os
import json
from pathlib import Path
from typing import List
from sklearn.metrics import f1_score

import torch
import torch.distributed as dist

try:  # prefer package relative imports
    from .config import cfg
    from .data_loader import CrossFitDataset
    from .prompt_builder import build_classification_prompt, build_multichoice_prompt
    from .models.model_utils import ModelUtils
    from .al import RandomSampler, DiversitySampler, UncertaintySampler, SimilaritySampler
except ImportError:  # allow execution as a standalone script
    import sys

    current_dir = Path(__file__).resolve().parent
    sys.path.append(str(current_dir))
    sys.path.append(str(current_dir.parent))

    from config import cfg
    from data_loader import CrossFitDataset
    from prompt_builder import build_classification_prompt, build_multichoice_prompt
    from models.model_utils import ModelUtils
    from al import RandomSampler, DiversitySampler, UncertaintySampler, SimilaritySampler

from tqdm import tqdm


def get_sampler(name: str):
    if name == "random":
        return RandomSampler(seed=cfg.seed)
    if name == "diversity":
        return DiversitySampler()
    if name == "uncertainty":
        return UncertaintySampler()
    if name == "similarity":
        return SimilaritySampler()
    raise ValueError(f"Unknown sampler {name}")


def setup_ddp() -> int:
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    cfg.device = f"cuda:{local_rank}"
    cfg.num_gpus = 1
    return local_rank


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def run(task: str, al_method: str, model_name: str, num_shots: int) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    pool_dataset = CrossFitDataset(task, "pool")
    test_dataset = CrossFitDataset(task, "test")
    sampler = get_sampler(al_method)
    mu = ModelUtils(model_name, device=cfg.device, num_gpus=1)
    label_space = sorted(set(pool_dataset.get_all_labels()))
    batch_size = cfg.inference_batch_size
    classification = "text" in pool_dataset.df.columns

    if al_method != "similarity":
        demo_indices = None
        if rank == 0:
            demo_indices = sampler.select(pool_dataset, num_shots)
        obj_list = [demo_indices]
        dist.broadcast_object_list(obj_list, src=0)
        demo_indices = obj_list[0]
        demos = [pool_dataset[i] for i in demo_indices]

    results = []
    indices = list(range(rank, len(test_dataset), world_size))
    for start in tqdm(range(0, len(indices), batch_size), desc="Evaluating", unit="batch", disable=rank != 0):
        batch_ids = indices[start:start + batch_size]
        items = [test_dataset[i] for i in batch_ids]
        prompts = []
        golds = []
        choices_list = []

        for item in items:
            if classification:
                text, gold = item
                if al_method == "similarity":
                    demo_ids = sampler.select_for_one_test(pool_dataset, text, num_shots)
                    demos_batch = [pool_dataset[i] for i in demo_ids]
                else:
                    demos_batch = demos
                prompts.append(build_classification_prompt(demos_batch, text))
                golds.append(gold)
            else:
                question, choices, gold = item
                if al_method == "similarity":
                    demo_ids = sampler.select_for_one_test(pool_dataset, question, num_shots)
                    demos_batch = [pool_dataset[i] for i in demo_ids]
                else:
                    demos_batch = demos
                prompts.append(build_multichoice_prompt(demos_batch, question, choices))
                choices_list.append(choices)
                golds.append(gold)

        if classification:
            preds = mu.predict_classification_batch(
                prompts, label_space, batch_size=batch_size
            )
        else:
            preds = mu.predict_multichoice_batch(
                prompts, choices_list, batch_size=batch_size
            )

        for idx, p, g in zip(batch_ids, preds, golds):
            results.append({"id": idx, "prediction": p, "gold": g, "correct": p == g})

    gather_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_list, results)

    if rank == 0:
        flat = [item for sublist in gather_list for item in sublist]
        flat.sort(key=lambda x: x["id"])  # restore original order
        accuracy = sum(r["correct"] for r in flat) / len(flat)
        print(f"Accuracy: {accuracy:.4f}")
        predictions = [r["prediction"] for r in flat]
        golds = [r["gold"] for r in flat]
        f1 = f1_score(golds, predictions, average="macro") if len(label_space) == 2 else None

        kind = "classification" if "text" in pool_dataset.df.columns else "multichoice"
        out_dir = Path(cfg.outputs_dir) / kind / task / model_name / al_method
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "task": task,
            "model": model_name,
            "al_method": al_method,
            "num_shots": num_shots,
            "accuracy": accuracy,
        }
        if f1 is not None:
            metrics["f1"] = f1
        metrics["per_example"] = flat
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--al_method", default=cfg.al_method)
    parser.add_argument("--model_name", default=cfg.model_name)
    parser.add_argument("--num_shots", type=int, default=cfg.num_shots)
    parser.add_argument("--perplexity_batch_size", type=int, default=cfg.perplexity_batch_size)
    parser.add_argument("--inference_batch_size", type=int, default=cfg.inference_batch_size)
    parser.add_argument("--local_rank", type=int, default=None, help="Provided by torchrun")
    args = parser.parse_args()

    cfg.perplexity_batch_size = args.perplexity_batch_size
    cfg.inference_batch_size = args.inference_batch_size

    _ = setup_ddp()
    try:
        run(args.task, args.al_method, args.model_name, args.num_shots)
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()

"""Main experiment driver."""

import argparse
import json
from pathlib import Path
from sklearn.metrics import f1_score

from .config import cfg
from .data_loader import CrossFitDataset
from .prompt_builder import build_classification_prompt, build_multichoice_prompt
from .models.model_utils import ModelUtils
from .al import RandomSampler, DiversitySampler, UncertaintySampler, SimilaritySampler
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


def run(task: str, al_method: str, model_name: str, num_shots: int) -> None:
    pool_dataset = CrossFitDataset(task, "pool")
    test_dataset = CrossFitDataset(task, "test")
    sampler = get_sampler(al_method)
    mu = ModelUtils(model_name, device=cfg.device, num_gpus=cfg.num_gpus)
    label_space = sorted(set(pool_dataset.get_all_labels()))
    batch_size = cfg.inference_batch_size
    classification = "text" in pool_dataset.df.columns

    if al_method != "similarity":
        demo_indices = sampler.select(pool_dataset, num_shots)
        demos = [pool_dataset[i] for i in demo_indices]

    results = []
    for start in tqdm(range(0, len(test_dataset), batch_size), desc="Evaluating", unit="batch"):
        batch_indices = range(start, min(start + batch_size, len(test_dataset)))
        items = [test_dataset[i] for i in batch_indices]
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

        for p, g in zip(preds, golds):
            results.append({"prediction": p, "gold": g, "correct": p == g})

    accuracy = sum(r["correct"] for r in results) / len(results)
    print(f"Accuracy: {accuracy:.4f}")

    predictions = [r["prediction"] for r in results]
    golds = [r["gold"] for r in results]
    if len(label_space) == 2:
        f1 = f1_score(golds, predictions, average="macro")
    else:
        f1 = None

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
    metrics["per_example"] = [
        {"id": i, "prediction": p, "gold": g, "correct": c}
        for i, (p, g, c) in enumerate(zip(predictions, golds, [r["correct"] for r in results]))
    ]
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--al_method", default=cfg.al_method)
    parser.add_argument("--model_name", default=cfg.model_name)
    parser.add_argument("--num_shots", type=int, default=cfg.num_shots)
    parser.add_argument("--num_gpus", type=int, default=cfg.num_gpus)

    parser.add_argument(
        "--perplexity_batch_size",
        type=int,
        default=cfg.perplexity_batch_size,
    )
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=cfg.inference_batch_size,
    )

    args = parser.parse_args()

    # Update config in case command-line overrides were provided
    cfg.num_gpus = args.num_gpus

    cfg.perplexity_batch_size = args.perplexity_batch_size
    cfg.inference_batch_size = args.inference_batch_size


    run(args.task, args.al_method, args.model_name, args.num_shots)


if __name__ == "__main__":
    main()

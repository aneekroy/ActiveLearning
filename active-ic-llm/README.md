# ActiveLLM Codebase

This directory contains the implementation of ActiveLLM. It is organised into several modules:

- `data_preprocessing/` — scripts for downloading and preparing datasets, including CrossFit tasks, math reasoning benchmarks, commonsense QA and instruction following corpora.
- `src/` — core package containing configuration loading, dataset classes, prompt builders and active learning strategies.
- `experiments/` — helper scripts for running batch experiments across tasks.
- `docs/` — additional documentation including a design overview.

## Installation

```bash
pip install -r requirements.txt
```

## Preparing Data

Run the preprocessing scripts to download and standardise the datasets mentioned above:

```bash
python data_preprocessing/download_datasets.py
python data_preprocessing/prepare_crossfit.py
```

## Running Experiments

Example usage:

Run the experiment module from within the `active-ic-llm` directory:

```bash
cd active-ic-llm
python -m src.run_experiment --task sst2 --al_method random --model_name bert-base-uncased --num_shots 8
# Optionally control batching for perplexity-based sampling
python -m src.run_experiment --task sst2 --al_method uncertainty --model_name bert-base-uncased --num_shots 8 --perplexity_batch_size 16
```

Batch experiments for all tasks are available under `experiments/`.
Outputs including per-example predictions and accuracy/F1 scores are written to the `outputs/` directory. The metrics file for a run can be found at `outputs/<task_type>/<task>/<model>/<al_method>/metrics.json`.
To average metrics across multiple runs you can use `../scripts/aggregate_metrics.py`.


## Category-wise Experiments

Two convenience scripts run ActiveLLM across groups of tasks using different
active learning strategies:

- `experiments/classification_exp.py` iterates over all
  `classification_tasks`, `math_reasoning_tasks` and `instruction_following_tasks`
  defined in `config.yaml`.
- `experiments/multichoice_exp.py` covers the `multichoice_tasks` and
  `commonsense_tasks` lists.

Both scripts accept a space separated list of strategies via the `--al_methods`
argument (defaults to `random diversity uncertainty similarity`). For every
combination of task and method they invoke `src/run_experiment.py`.

Example running every classification task with all strategies:

```bash
python experiments/classification_exp.py --model_name bert-base-uncased --num_shots 8
```

You can limit the strategies to a subset, e.g.:

```bash
python experiments/classification_exp.py --al_methods random diversity --num_shots 8
```

The same interface applies to `multichoice_exp.py`.

A combined script `experiments/all_tasks_exp.py` runs every task from all five categories (classification, multichoice, commonsense, math reasoning and instruction following) in one go. It accepts the same `--al_methods`, `--model_name` and `--num_shots` arguments and sequentially calls `src/run_experiment.py` for each combination.

### Active learning methods

ActiveLLM includes four samplers:

| Method | Description |
|--------|-------------|
| `random` | Randomly pick `k` pool examples. |
| `diversity` | Embed the pool with SentenceTransformer and select diverse examples using k-means clustering. |
| `uncertainty` | Choose examples with highest perplexity under the current model. |
| `similarity` | For each test instance select the pool items with most similar embeddings. |

`run_experiment.py` loads the pool and test splits, applies the sampler to obtain
demonstrations and builds prompts for either classification or multiple choice
tasks. Predictions and metrics are then written under `outputs/`.

### Llama-3.2 training examples

If `--model_name` is a path on disk, the model will be loaded entirely from that
directory without downloading files.

For sampling strategies that rely on sentence embeddings, you can set the
`SBERT_MODEL` environment variable to a local directory to load the embeddings
model from disk.



cd active-ic-llm
python -m src.run_experiment --task gsm8k --al_method random --model_name llama-3.2-1b --num_shots 8
python -m src.run_experiment --task MultiArith --al_method random --model_name llama-3.2-1b --num_shots 8
python -m src.run_experiment --task AddSub --al_method random --model_name llama-3.2-1b --num_shots 8
python -m src.run_experiment --task gsm8k --al_method random --model_name llama-3.2-3b --num_shots 8
python -m src.run_experiment --task MultiArith --al_method random --model_name llama-3.2-3b --num_shots 8
python -m src.run_experiment --task AddSub --al_method random --model_name llama-3.2-3b --num_shots 8

```bash
# 1B model
python src/run_experiment.py --task gsm8k --al_method random --model_name llama-3.2-1b --num_shots 8
python src/run_experiment.py --task MultiArith --al_method random --model_name llama-3.2-1b --num_shots 8
python src/run_experiment.py --task AddSub --al_method random --model_name llama-3.2-1b --num_shots 8

# 3B model
python src/run_experiment.py --task gsm8k --al_method random --model_name llama-3.2-3b --num_shots 8
python src/run_experiment.py --task MultiArith --al_method random --model_name llama-3.2-3b --num_shots 8
python src/run_experiment.py --task AddSub --al_method random --model_name llama-3.2-3b --num_shots 8
```


### Evaluation script

To reproduce the evaluation for both models on the math benchmarks run:

```bash
python ../scripts/run_llama_eval.py
```


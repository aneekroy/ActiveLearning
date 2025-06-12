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
To enable optional training and inference optimisations using [unsloth.ai](https://www.unsloth.ai), install the package and pass the `--use_unsloth` flag when running experiments. To avoid pulling heavy dependencies you may run `pip install --no-deps unsloth`.

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
```

Batch experiments for all tasks are available under `experiments/`.
Outputs including per-example predictions and accuracy/F1 scores are written to the `outputs/` directory. The metrics file for a run can be found at `outputs/<task_type>/<task>/<model>/<al_method>/metrics.json`.
To average metrics across multiple runs you can use `../scripts/aggregate_metrics.py`.

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


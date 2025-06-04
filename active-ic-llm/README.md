# ActiveLLM Codebase

This directory contains the implementation of ActiveLLM. It is organised into several modules:

- `data_preprocessing/` — scripts for downloading and preparing CrossFit datasets.
- `src/` — core package containing configuration loading, dataset classes, prompt builders and active learning strategies.
- `experiments/` — helper scripts for running batch experiments across tasks.
- `docs/` — additional documentation including a design overview.

## Installation

```bash
pip install -r requirements.txt
```

## Preparing Data

Run the preprocessing scripts to download and standardise the CrossFit datasets:

```bash
python data_preprocessing/download_datasets.py
python data_preprocessing/prepare_crossfit.py
```

## Running Experiments

Example usage:

```bash
python src/run_experiment.py --task sst2 --al_method random --model_name bert-base-uncased --num_shots 8
```

Batch experiments for all tasks are available under `experiments/`.
Outputs including per-example predictions and accuracy/F1 scores are written to the `outputs/` directory. The metrics file for a run can be found at `outputs/<task_type>/<task>/<model>/<al_method>/metrics.json`.

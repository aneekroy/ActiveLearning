# Step-by-Step Walkthrough

This guide demonstrates how to set up the ActiveLLM codebase and run a minimal experiment.

## 1. Clone the repository

```bash
git clone <repo_url>
cd ActiveLearning
```

## 2. Install dependencies

Use the requirements file provided with the main package:

```bash
pip install -r active-ic-llm/requirements.txt
```

## 3. Download and prepare datasets

Run the preprocessing scripts to fetch and standardise datasets:

```bash
python active-ic-llm/data_preprocessing/download_datasets.py
python active-ic-llm/data_preprocessing/prepare_crossfit.py
```

## 4. Run an experiment

Experiments are driven by the `src.run_experiment` module. Change into the package directory and invoke it with `-m`:

```bash
cd active-ic-llm
python -m src.run_experiment --task sst2 --al_method random --model_name bert-base-uncased --num_shots 8
```

Outputs including predictions and metrics will be written under `outputs/<task_type>/<task>/<model>/<al_method>/`.

If you provide a local directory to `--model_name`, the model is loaded from
that path without contacting HuggingFace. You can also set the `SBERT_MODEL`
environment variable to point to a local sentence transformer so that
diversity/similarity sampling works offline.

## 5. Aggregate metrics across runs (optional)

If you perform multiple runs of the same configuration you can average the metrics using the helper script:

```bash
python scripts/aggregate_metrics.py --task sst2 --model bert-base-uncased --al_method random
```

The aggregated accuracy and F1 scores are stored in `outputs/<task>/<model>/<al_method>/avg_metrics.json`.

cd active-ic-llm
python -m src.run_experiment --task gsm8k --al_method random --model_name llama-3.2-1b --num_shots 8
python -m src.run_experiment --task MultiArith --al_method random --model_name llama-3.2-1b --num_shots 8
python -m src.run_experiment --task AddSub --al_method random --model_name llama-3.2-1b --num_shots 8

python -m src.run_experiment --task gsm8k --al_method random --model_name llama-3.2-3b --num_shots 8
python -m src.run_experiment --task MultiArith --al_method random --model_name llama-3.2-3b --num_shots 8
python -m src.run_experiment --task AddSub --al_method random --model_name llama-3.2-3b --num_shots 8

```bash
# 1B model
python active-ic-llm/src/run_experiment.py --task gsm8k --al_method random --model_name llama-3.2-1b --num_shots 8
python active-ic-llm/src/run_experiment.py --task MultiArith --al_method random --model_name llama-3.2-1b --num_shots 8
python active-ic-llm/src/run_experiment.py --task AddSub --al_method random --model_name llama-3.2-1b --num_shots 8

# 3B model
python active-ic-llm/src/run_experiment.py --task gsm8k --al_method random --model_name llama-3.2-3b --num_shots 8
python active-ic-llm/src/run_experiment.py --task MultiArith --al_method random --model_name llama-3.2-3b --num_shots 8
python active-ic-llm/src/run_experiment.py --task AddSub --al_method random --model_name llama-3.2-3b --num_shots 8
```


### 7. Evaluating the trained models

After running the training commands you can evaluate both models on the same set of tasks using:

```bash
python scripts/run_llama_eval.py
```


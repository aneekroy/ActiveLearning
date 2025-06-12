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

Optional features such as the `unsloth` library can be installed if you plan on using the `--use_unsloth` flag during training.

## 3. Download and prepare datasets

Run the preprocessing scripts to fetch and standardise datasets:

```bash
python active-ic-llm/data_preprocessing/download_datasets.py
python active-ic-llm/data_preprocessing/prepare_crossfit.py
```

## 4. Run an experiment

Experiments are driven by `active-ic-llm/src/run_experiment.py`. For example, to train on SST-2 with random sampling:

```bash
python active-ic-llm/src/run_experiment.py --task sst2 --al_method random --model_name bert-base-uncased --num_shots 8
```

Outputs including predictions and metrics will be written under `outputs/<task_type>/<task>/<model>/<al_method>/`.

## 5. Aggregate metrics across runs (optional)

If you perform multiple runs of the same configuration you can average the metrics using the helper script:

```bash
python scripts/aggregate_metrics.py --task sst2 --model bert-base-uncased --al_method random
```

The aggregated accuracy and F1 scores are stored in `outputs/<task>/<model>/<al_method>/avg_metrics.json`.
If your runs only produce log files, edit `TASK` and `MODE` in `scripts/calc_avg.py` and run that script instead.


## 6. Training Llama-3.2 models on math benchmarks

To fine-tune the lightweight Llama-3.2 models on math reasoning datasets you can invoke `run_experiment.py` with the desired task name and model. The examples below train with random sampling and eight demonstration shots:

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

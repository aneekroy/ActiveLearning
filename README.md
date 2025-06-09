# ActiveLLM

This repository provides an implementation of **ActiveLLM**, a large language model based active learning approach for textual few-shot scenarios as described in the accompanying paper.

The code is organised as a Python package under `active-ic-llm` and contains scripts for dataset preparation, running experiments with different active learning strategies and models, and utilities for embeddings and clustering.

See `active-ic-llm/README.md` for details on how to install dependencies, prepare data and reproduce the experiments.

The codebase optionally integrates the [unsloth.ai](https://www.unsloth.ai) library
for faster model loading when the `--use_unsloth` flag is supplied.

Prepared datasets span multiple benchmarks including math reasoning
(e.g. GSM8k, AQUA-RAT), commonsense QA (BoolQ, HellaSwag, ARC, Winogrande, PiQA
and others) and instruction following evaluations such as DollyEval and VicunaEval.

## Aggregating Results

Use `scripts/aggregate_metrics.py` to compute the average accuracy and F1 across multiple runs. It expects the default metrics produced by `run_experiment.py` and writes the aggregated values next to them.

```bash
python scripts/aggregate_metrics.py --task <task> --model <model_name> --al_method <strategy>
```

## Step-by-Step Walkthrough

A concise walkthrough covering setup, dataset preparation and running a sample experiment is provided in [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md).

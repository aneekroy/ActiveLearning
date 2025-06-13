# ActiveLLM

This repository provides an implementation of [**ActiveLLM**](https://arxiv.org/abs/2405.10808), a large language model based active learning approach for textual few-shot scenarios as described in the accompanying paper.

The code is organised as a Python package under `active-ic-llm` and contains scripts for dataset preparation, running experiments with different active learning strategies and models, and utilities for embeddings and clustering.

See `active-ic-llm/README.md` for details on how to install dependencies, prepare data and reproduce the experiments.


Prepared datasets span multiple benchmarks including math reasoning
(e.g. GSM8k, AQUA-RAT), commonsense QA (BoolQ, HellaSwag, ARC, Winogrande, PiQA
and others) and instruction following evaluations such as DollyEval and VicunaEval.

## Aggregating Results

Use `scripts/aggregate_metrics.py` to compute the average accuracy and F1 across multiple runs. It expects the default metrics produced by `run_experiment.py` and writes the aggregated values next to them.

```bash
python scripts/aggregate_metrics.py --task <task> --model <model_name> --al_method <strategy>
```

If your experiments write log files instead of metrics JSON, `scripts/calc_avg.py` replicates the averaging used in the original ActiveLLM repo.
Edit the `TASK` and `MODE` constants inside the script and run:
```bash
python scripts/calc_avg.py
```

If you pass a filesystem path to `--model_name`, the loader reads the model
directly from that directory without accessing the HuggingFace Hub.


Embeddings required for diversity and similarity sampling use the
`SentenceTransformer` library. Set the `SBERT_MODEL` environment variable to a
local path if you wish to avoid downloading the default model.


## Step-by-Step Walkthrough

A concise walkthrough covering setup, dataset preparation and running a sample experiment is provided in [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md).

### Example: training Llama-3.2 on math tasks

See [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md#6-training-llama-3-2-models-on-math-benchmarks) for commands that fine-tune the 1B and 3B Llama-3.2 models on GSM8k, MultiArith and AddSub.


### Example: evaluating the math benchmarks

Use `scripts/run_llama_eval.py` to automatically run the evaluation for both models on the same three datasets:

```bash
python scripts/run_llama_eval.py
```

### Avoiding GPU 0 OOM in multi-GPU runs

`run_experiment.py` relies on PyTorch **DataParallel** when more than one GPU is
available. The first listed device keeps an extra copy of the model and gathers
the outputs, which can quickly exhaust the memory of GPU 0.

To balance usage you can hide GPU 0 or change the gather device:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 python -m src.run_experiment ...

model = torch.nn.DataParallel(model, device_ids=[0,1,2,3], output_device=3)
model = model.cuda(0)
```

For large models switching to `torch.distributed`'s
`DistributedDataParallel` is recommended. Launch the experiment with:

```bash
torchrun --nproc_per_node=4 src/run_experiment_ddp.py \
    --task <task> --al_method <method> --model_name <path> --num_shots 8
```

See [docs/GPU_PARALLELISM.md](docs/GPU_PARALLELISM.md) for details.


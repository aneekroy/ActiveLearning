# ActiveLLM Design

This document outlines the structure of the ActiveLLM implementation.

The repository separates data preparation, active learning strategies and model utilities into distinct modules. Preprocessing scripts download and convert datasets to a common CSV format. The `src` package implements samplers such as random, diversity, uncertainty and similarity based selection. Experiments are run via `src/run_experiment.py` or the convenience wrappers in the `experiments/` directory.
The experiment driver records predictions for every test example and stores summary metrics in a JSON file.
Outputs are organised under `outputs/{classification|multichoice}/{task}/{model}/{al_method}/metrics.json`.

import subprocess

TASKS = ["gsm8k", "MultiArith", "AddSub"]
MODELS = ["llama-3.2-1b", "llama-3.2-3b"]

for model in MODELS:
    for task in TASKS:
        cmd = [
            "python",
            "-m",
            "src.run_experiment",
            "--task", task,
            "--al_method", "random",
            "--model_name", model,
            "--num_shots", "8",
        ]
        subprocess.run(cmd, check=True, cwd="active-ic-llm")


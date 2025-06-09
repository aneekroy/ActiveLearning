import argparse
from subprocess import call
from src.config import cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name")
    parser.add_argument("--al_method")
    parser.add_argument("--num_shots", type=int)
    args = parser.parse_args()

    task_lists = [
        cfg.classification_tasks,
        cfg.get("math_reasoning_tasks", []),
        cfg.get("instruction_following_tasks", []),
    ]
    for task_list in task_lists:
        for task in task_list:
            cmd = ["python", "src/run_experiment.py", "--task", task]
            if args.model_name:
                cmd += ["--model_name", args.model_name]
            if args.al_method:
                cmd += ["--al_method", args.al_method]
            if args.num_shots is not None:
                cmd += ["--num_shots", str(args.num_shots)]
            call(cmd)


if __name__ == "__main__":
    main()

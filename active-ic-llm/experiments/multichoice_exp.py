import argparse
from subprocess import call
from src.config import cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name")
    parser.add_argument(
        "--al_methods",
        nargs="+",
        default=["random", "diversity", "uncertainty", "similarity"],
        help="Space separated list of active learning methods to run",
    )
    parser.add_argument("--num_shots", type=int)
    args = parser.parse_args()

    task_lists = [
        cfg.multichoice_tasks,
        cfg.get("commonsense_tasks", []),
    ]
    for task_list in task_lists:
        for task in task_list:
            for method in args.al_methods:
                cmd = ["python", "src/run_experiment.py", "--task", task, "--al_method", method]
                if args.model_name:
                    cmd += ["--model_name", args.model_name]
                if args.num_shots is not None:
                    cmd += ["--num_shots", str(args.num_shots)]
                call(cmd)


if __name__ == "__main__":
    main()

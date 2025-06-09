import argparse
import json
from pathlib import Path
from statistics import mean


def load_metrics(files):
    metrics = []
    for path in files:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                metrics.append(json.load(f))
        except Exception:
            continue
    return metrics


def aggregate(metrics):
    acc = [m.get('accuracy') for m in metrics if 'accuracy' in m]
    f1 = [m.get('f1') for m in metrics if 'f1' in m]
    result = {}
    if acc:
        result['accuracy'] = mean(acc)
    if f1:
        result['f1'] = mean(f1)
    return result


def main():
    parser = argparse.ArgumentParser(description='Aggregate metrics across runs')
    parser.add_argument('--outputs_dir', default='outputs')
    parser.add_argument('--task', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--al_method', required=True)
    args = parser.parse_args()

    base = Path(args.outputs_dir)
    pattern = f'*/*/{args.task}/{args.model}/{args.al_method}/metrics.json'
    files = list(base.glob(pattern))
    metrics = load_metrics(files)
    if not metrics:
        print('No metrics found for pattern', pattern)
        return
    agg = aggregate(metrics)
    out_file = base / args.task / args.model / args.al_method / 'avg_metrics.json'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))


if __name__ == '__main__':
    main()

import argparse
from functools import partial
from pathlib import Path
from pprint import pprint

import pandas as pd
from tqdm.contrib.concurrent import process_map

from guidedsum.evaluation import calculate_rouge, load_summaries


def evaluate_trial(trial: Path, step):
    threshold = float(str(trial).split("_")[-1])
    predictions = load_summaries(trial, step=step)
    metrics = calculate_rouge(predictions["candidate"], predictions["gold"]).mean()
    metrics.name = threshold
    return metrics


def main(args):
    base_path = Path(args.base_path)
    trials = list(base_path.glob("validate_threshold_*"))
    print(f"Evaluate {len(trials)} trials")
    print("First 5 trials:")
    pprint(trials[:5])

    _evaluate = partial(evaluate_trial, step=args.step)
    metrics = process_map(_evaluate, trials, max_workers=args.n_jobs)
    df_metrics = pd.concat(metrics, axis=1).T.sort_index()
    df_metrics.index.name = "threshold"
    df_metrics[["rouge1", "rouge1_p", "rouge1_r", "rouge2", "rougeLsum"]].to_csv(
        base_path / "scores_all.csv"
    )

    ix_best = df_metrics["rouge1"].argmax()
    threshold = df_metrics.index[ix_best]
    metrics = df_metrics.iloc[ix_best]
    metrics.to_csv(base_path / "scores_best.csv")
    print("=" * 10)
    print(f"ROUGE-1 F-measure optimal threshold: {threshold}")
    print(metrics.to_string())
    print("=" * 10)
    with open(base_path / "best.txt", "w") as fout:
        fout.write(str(threshold) + "\n")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base_path",
        type=str,
        help="Path where thresholded predictions are written to. Each trial should be a subdirectory following name: `validate_threshold_{threshold}`",
    )
    parser.add_argument(
        "step",
        type=int,
        help="The training step of the BertExt checkpoint that generated summaries.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        help="Evaluate n trials in parallel.",
        default=25,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(arg_parser())

import argparse
from pathlib import Path

import pandas as pd


def main(args):
    in_path = Path(args.input_path)
    out_path = Path(args.output_path)
    out_path.mkdir(exist_ok=True, parents=True)

    if args.dataset_name in [
        "mimic",
        "mimic-bg",
        "mimic-official",
        "mimic-official-bg",
        "openi",
        "openi-bg",
    ]:
        # We use the same tokenizer for mimic and OpenI
        from guidedsum.mimic.base import preprocess
    else:
        raise ValueError("Dataset not supported.")

    splits = ["reports.train.json", "reports.valid.json", "reports.test.json"]
    for split in splits:
        df = pd.read_json(in_path / split)

        src_col = "findings+bg" if args.include_background else "findings"
        df["src"] = df[src_col].apply(preprocess)
        df["tgt"] = df["impression"].apply(preprocess)
        df = df[["id", "src", "tgt"]]
        df.to_json(out_path / split)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    parser.add_argument(
        "--include_background",
        action="store_true",
        help="When given, the source text will include both background and findings. Otherwise only findings.",
    )
    parser.add_argument(
        "--dataset_name",
        help="Name of dataset",
        choices=[
            "mimic",
            "mimic-bg",
            "mimic-official",
            "mimic-official-bg",
            "openi",
            "openi-bg",
        ],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(arg_parser())

from pathlib import Path

import pandas as pd

from guidedsum.mimic.base import preprocess


def convert_to_wgsum_format(in_path, out_path, include_background=False):
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)

    for split in ["train", "valid", "test"]:
        out_file = out_path / f"{split}.jsonl"
        if out_file.exists():
            print(f"{out_file} exists. Skip.")
            continue
        else:
            print(f"Write results to {out_file}")

        df = pd.read_json(in_path / f"reports.{split}.json")
        flatten = lambda sents: [token for sent in sents for token in sent]
        if include_background:
            df["findings"] = df["findings+bg"].apply(preprocess).apply(flatten)
        else:
            df["findings"] = df["findings"].apply(preprocess).apply(flatten)
        df["impression"] = df["impression"].apply(preprocess)
        df = df[["findings", "impression", "id"]]
        df.to_json(out_file, lines=True, orient="records")


def main():
    convert_to_wgsum_format("data/processed/mimic/", "data/processed/mimic-wgsum/")
    convert_to_wgsum_format(
        "data/processed/mimic/",
        "data/processed/mimic-bg-wgsum/",
        include_background=True,
    )
    convert_to_wgsum_format(
        "data/processed/mimic-official/", "data/processed/mimic-official-wgsum/"
    )
    convert_to_wgsum_format(
        "data/processed/mimic-official/",
        "data/processed/mimic-official-bg-wgsum/",
        include_background=True,
    )

    convert_to_wgsum_format("data/processed/openi/", "data/processed/openi-wgsum/")
    convert_to_wgsum_format(
        "data/processed/openi/",
        "data/processed/openi-bg-wgsum/",
        include_background=True,
    )


if __name__ == "__main__":
    main()

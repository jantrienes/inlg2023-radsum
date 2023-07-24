import argparse
from pathlib import Path

import pandas as pd


def write_chunks(chunked_path, prefix, df, chunksize):
    chunked_path = Path(chunked_path)
    chunked_path.mkdir(parents=True, exist_ok=True)
    for i, start in enumerate(range(0, len(df), chunksize)):
        chunk = df.iloc[start : start + chunksize]
        chunk.to_json(chunked_path / f"{prefix}.{i}.json", orient="records")


def main(args):
    input_path = Path(args.input_path)
    chunked_path = input_path / "chunked"
    chunksize = args.chunksize

    train = pd.read_json(input_path / "reports.train.json")
    valid = pd.read_json(input_path / "reports.valid.json")
    test = pd.read_json(input_path / "reports.test.json")

    write_chunks(chunked_path, prefix="reports.train", df=train, chunksize=chunksize)
    write_chunks(chunked_path, prefix="reports.valid", df=valid, chunksize=chunksize)
    write_chunks(chunked_path, prefix="reports.test", df=test, chunksize=chunksize)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--chunksize", default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    main(arg_parser())

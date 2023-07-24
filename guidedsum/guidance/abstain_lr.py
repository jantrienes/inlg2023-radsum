import argparse
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from guidedsum.mimic.base import is_abstractive


def main(args):
    data_path = Path(args.data_path)
    out_path = Path(args.output_path)
    out_path.mkdir(exist_ok=True, parents=True)

    df = pd.concat(
        [
            pd.read_json(data_path / "reports.train.json").assign(split="train"),
            pd.read_json(data_path / "reports.valid.json").assign(split="valid"),
            pd.read_json(data_path / "reports.test.json").assign(split="test"),
        ]
    )
    df["x"] = df["src"].apply(lambda x: " ".join(sum(x, [])))
    df["y"] = (
        df["tgt"]
        .apply(lambda x: " ".join(sum(x, [])))
        .apply(is_abstractive)
        .astype(int)
    )
    df = df.set_index(["split", "id"])

    X_train, y_train = df.loc["train", "x"], df.loc["train", "y"]
    X_valid, y_valid = df.loc["valid", "x"], df.loc["valid", "y"]
    X_test, y_test = df.loc["test", "x"], df.loc["test", "y"]

    pipe = Pipeline(
        [
            ("vect", CountVectorizer(ngram_range=(1, 2), min_df=5)),
            ("tfidf", TfidfTransformer()),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    num_params = len(pipe["vect"].get_feature_names())  # pylint: disable=E1101
    logger.info(f"Vocabulary size/parameters: {num_params}")

    def predict(X, y_true, split_name):
        # Predict and evaluate
        y_pred = pipe.predict(X)
        report = classification_report(y_true, y_pred)
        logger.info(f"Evaluate {split_name}\n{report}")

        # Save predictions
        with open(out_path / f"y_true_{split_name}.txt", "w") as fout:
            for i in y_true:
                fout.write(str(i) + "\n")
        with open(out_path / f"y_pred_{split_name}.txt", "w") as fout:
            for i in y_pred:
                fout.write(str(i) + "\n")

    predict(X_train, y_train, "train")
    predict(X_valid, y_valid, "valid")
    predict(X_test, y_test, "test")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--output_path")
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = arg_parser()
    logger.add(Path(ARGS.output_path) / "train.log")
    main(ARGS)

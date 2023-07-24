import argparse
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from guidedsum.guidance.extractive_guidance import greedy_selection_parallel


def load_dataset(train_path, valid_path, test_path):
    df = pd.concat(
        [
            pd.read_json(train_path).assign(split="train"),
            pd.read_json(valid_path).assign(split="valid"),
            pd.read_json(test_path).assign(split="test"),
        ]
    )

    df["x"] = df["src"].apply(lambda x: " ".join(sum(x, [])))
    df["y"] = [len(oracle) for oracle in greedy_selection_parallel(df)]
    df = df.set_index(["split", "id"])
    return df


def predict(pipe, df, split_name: str, out_path):
    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True, parents=True)
    y_pred = pipe.predict(df.loc[split_name, "x"])
    y_true = df.loc[split_name, "y"]
    logger.info(f"Evaluate {split_name}\n" + classification_report(y_true, y_pred))

    with open(out_path / f"y_true_{split_name}.txt", "w") as fout:
        for i in y_true:
            fout.write(str(i) + "\n")

    with open(out_path / f"y_pred_{split_name}.txt", "w") as fout:
        for i in y_pred:
            fout.write(str(i) + "\n")


def main(args):
    data_path = Path(args.data_path)
    df = load_dataset(
        data_path / "reports.train.json",
        data_path / "reports.valid.json",
        data_path / "reports.test.json",
    )
    logger.info("dataset head:\n{}", df.head())

    pipe = Pipeline(
        [
            ("vect", CountVectorizer(ngram_range=(1, 1), min_df=5)),
            ("tfidf", TfidfTransformer()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    solver="saga",
                    multi_class="multinomial",
                ),
            ),
        ]
    )

    pipe.fit(df.loc["train", "x"], df.loc["train", "y"])
    train_accuracy = pipe.score(df.loc["train", "x"], df.loc["train", "y"])
    num_params = len(pipe["vect"].get_feature_names())  # pylint: disable=E1101
    logger.info(f"Vocabulary size/parameters: {num_params}")
    logger.info(f"Train accuracy: {train_accuracy:.2f}")
    predict(pipe, df, "train", args.output_path)
    predict(pipe, df, "valid", args.output_path)
    predict(pipe, df, "test", args.output_path)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--output_path")
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = arg_parser()
    logger.add(Path(ARGS.output_path) / "train.log")
    main(ARGS)

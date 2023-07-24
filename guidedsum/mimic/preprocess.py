import argparse
import glob
import math
import multiprocessing
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from guidedsum.utils import train_val_test_split


def label_command(csv_path):
    csv_path = Path(csv_path)
    folder = csv_path.parent.absolute()
    in_csv = csv_path.name
    out_csv = f"{csv_path.stem}.labeled.csv"

    command = [
        "docker",
        "run",
        "-v",
        f"{folder}:/data",
        "chexpert-labeler:latest",
        "python",
        "label.py",
        "--reports_path",
        f"/data/{in_csv}",
        "--output_path",
        f"/data/{out_csv}",
    ]

    return command


def run_labeler(csv_path):
    command = label_command(csv_path)
    return subprocess.run(command, capture_output=True, check=True)


def write_chunks(texts: pd.Series, chunked_path: Path, chunksize):
    n_chunks = math.ceil(len(texts) / chunksize)
    width = len(str(n_chunks))
    files = []
    for i, start in enumerate(range(0, len(texts), chunksize)):
        chunk = texts.iloc[start : start + chunksize]
        chunk_csv = chunked_path / f"data.{i:0{width}}.csv"
        chunk.to_csv(chunk_csv, header=False, index=False)
        files.append(chunk_csv)
    return files


def label_texts(texts: pd.Series, chunksize=2500):
    with tempfile.TemporaryDirectory() as tmpdir:
        chunk_path = Path(tmpdir)
        chunks = write_chunks(texts, chunk_path, chunksize=chunksize)
        with multiprocessing.Pool() as p:
            tasks = p.imap(run_labeler, chunks)
            tasks = tqdm(tasks, total=len(chunks), desc="Label chunks")
            tasks = list(tasks)  # run and wait for completion
        labeled_chunks = glob.glob(str(chunk_path / "data.*.labeled.csv"))
        df = pd.concat(
            [pd.read_csv(f) for f in sorted(labeled_chunks)], ignore_index=True
        )
        df = df.drop(
            "Reports", axis=1
        )  # Reports column includes the text that was labeled. Obsolete, so drop.
        return df.to_dict(orient="records")


def clean_text(text):
    """Clean up the impression string.
    This mainly removes bullet numbers for consistency.
    """
    text = text.strip().replace("\n", "")
    # remove bullet numbers
    text = re.sub(r"^[0-9]\.\s+", "", text)
    text = re.sub(r"\s[0-9]\.\s+", " ", text)
    text = re.sub(r"\s\s+", " ", text)
    text = re.sub(r"_+", "_", text)
    return text


def load_reports(reports_csv):
    df = pd.read_csv(reports_csv)
    for col in [
        "findings",
        "impression",
        "comparison",
        "examination",
        "history",
        "technique",
        "indication",
    ]:
        df[col] = df[col].fillna("").apply(clean_text)
    df["id"] = df["study"].str[1:].astype(int)
    df = df.drop(["study", "last_paragraph"], axis=1)
    df = df.sort_values("id")
    return df


def filter_reports(df):
    df_filtered = df[
        (df["findings"].str.split().apply(len) >= 10)
        & (df["impression"].str.split().apply(len) >= 2)
        & (
            df["findings"].str.split().apply(len)
            > df["impression"].str.split().apply(len)
        )
    ]
    df_filtered = df_filtered.copy()
    return df_filtered


def combine_background_findings(report):
    """
    Combine background and findings in a standardized format.
    Order of section resembles the order in original reports.
    """

    txt = ""
    sections = [
        ("examination", "Examination"),
        ("indication", "Indication"),
        ("technique", "Technique"),
        ("comparison", "Comparison"),
        ("history", "History"),
        ("findings", "Findings"),
    ]

    for key, title in sections:
        if key not in report:
            continue

        section_text = report[key]
        if section_text.strip():
            txt += f"{title}:\n{section_text}"
            txt += "\n\n"
    txt = txt.strip()
    return txt


def main(args):
    print("Load MIMIC-CXR reports...")
    df = load_reports(args.reports_path)
    print(f"Sectioned Reports: {len(df):,}")
    df = filter_reports(df)
    print(f"Filtered Reports: {len(df):,}")

    # Add text column which includes both background and findings.
    df["findings+bg"] = df.apply(combine_background_findings, axis=1)

    print("Run CheXpert labeler on findings and impression.")
    df["chexpert_labels_findings"] = label_texts(df["findings"])
    df["chexpert_labels_impression"] = label_texts(df["impression"])

    # Generate random split
    train, valid, test = train_val_test_split(
        df,
        frac_train=0.8,
        frac_val=0.1,
        frac_test=0.1,
        random_state=42,
    )

    print(f"Random split (train/val/test): {len(train)}/{len(valid)}/{len(test)}")
    out_path = Path(args.random_split_output_path)
    out_path.mkdir(exist_ok=True, parents=True)
    train.to_json(out_path / "reports.train.json", orient="records")
    valid.to_json(out_path / "reports.valid.json", orient="records")
    test.to_json(out_path / "reports.test.json", orient="records")
    train["id"].to_csv(out_path / "train.ids", header=False, index=False)
    valid["id"].to_csv(out_path / "valid.ids", header=False, index=False)
    test["id"].to_csv(out_path / "test.ids", header=False, index=False)

    # Generate official split
    split_map = pd.read_csv(args.splits_csv_path)
    split_map = (
        split_map.drop_duplicates("study_id").set_index("study_id")["split"].to_dict()
    )
    split = df["id"].apply(lambda x: split_map[x])
    train = df[split == "train"]
    valid = df[split == "validate"]
    test = df[split == "test"]

    print(f"Official split (train/val/test): {len(train)}/{len(valid)}/{len(test)}")
    out_path = Path(args.official_split_output_path)
    out_path.mkdir(exist_ok=True, parents=True)
    train.to_json(out_path / "reports.train.json", orient="records")
    valid.to_json(out_path / "reports.valid.json", orient="records")
    test.to_json(out_path / "reports.test.json", orient="records")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_csv_path", help="Path to official MIMIC-CXR splits.")
    parser.add_argument("--reports_path", help="Path to sectioned MIMIC reports.")
    parser.add_argument(
        "--random_split_output_path",
        help="Path to write processed dataset to (random split)",
    )
    parser.add_argument(
        "--official_split_output_path",
        help="Path to write processed dataset to (official split)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(arg_parser())

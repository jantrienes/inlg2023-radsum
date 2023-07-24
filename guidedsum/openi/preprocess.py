import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from guidedsum.mimic.preprocess import (
    clean_text,
    combine_background_findings,
    filter_reports,
    label_texts,
)
from guidedsum.utils import train_val_test_split


def load_reports(reports_path):
    report_files = Path(reports_path).glob("*.xml")
    reports = []
    for f in report_files:
        tree = ET.parse(f)
        root = tree.getroot()

        report = {}
        report["id"] = root.find("pmcId").get("id")
        sections = (
            root.find("MedlineCitation")
            .find("Article")
            .find("Abstract")
            .findall("AbstractText")
        )
        for section in sections:
            name = section.get("Label").lower()
            text = clean_text(section.text) if section.text is not None else ""
            report[name] = text
        reports.append(report)
    return pd.DataFrame(reports)


def main(args):
    print("Load OpenI reports...")
    df = load_reports(args.reports_path)
    print(f"N: {len(df):,}")
    df = filter_reports(df)
    print(f"N Filtered: {len(df):,}")

    # Add column which includes both background and findings.
    df["findings+bg"] = df.apply(combine_background_findings, axis=1)

    print("Run CheXpert labeler on findings and impression.")
    df["chexpert_labels_findings"] = label_texts(df["findings"], chunksize=500)
    df["chexpert_labels_impression"] = label_texts(df["impression"], chunksize=500)

    train, valid, test = train_val_test_split(
        df,
        frac_train=0.7,
        frac_val=0.1,
        frac_test=0.2,
        random_state=42,
    )

    print(f"Length train/val/test: {len(train)}/{len(valid)}/{len(test)}")
    out_path = Path(args.output_path)
    out_path.mkdir(exist_ok=True, parents=True)
    train.to_json(out_path / "reports.train.json", orient="records")
    valid.to_json(out_path / "reports.valid.json", orient="records")
    test.to_json(out_path / "reports.test.json", orient="records")
    train["id"].to_csv(out_path / "train.ids", header=False, index=False)
    valid["id"].to_csv(out_path / "valid.ids", header=False, index=False)
    test["id"].to_csv(out_path / "test.ids", header=False, index=False)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports_path", help="Path to raw OpenI XML files.")
    parser.add_argument("--output_path", help="Path to write processed dataset to.")
    return parser.parse_args()


if __name__ == "__main__":
    main(arg_parser())

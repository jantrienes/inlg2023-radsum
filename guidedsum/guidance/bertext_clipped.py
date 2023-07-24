"""
Output 1: Clipped BertExt summaries
Output 2: IDs of those BertExt sentences
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from guidedsum.evaluation import load_summaries
from guidedsum.guidance.extractive_guidance import greedy_selection_parallel


def load_clips(clips_path):
    # each line contains an integer
    with open(clips_path) as fin:
        clips = [int(line.strip()) for line in fin]
    return clips


def extract_k(candidate, k):
    # Format: `sent1<q>sent2<q>sent3`
    sents = candidate.split("<q>")
    sents = sents[:k]
    sents = "<q>".join(sents)
    return sents


def write_clipped(df, clips, out_path, step):
    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)

    with open(out_path / f"summaries.{step}.id", "w") as fout:
        for id_ in df["id"]:
            fout.write(str(id_) + "\n")

    with open(out_path / f"summaries.{step}.gold", "w") as fout:
        for gold in df["gold"]:
            fout.write(str(gold) + "\n")

    with open(out_path / f"summaries.{step}.candidate", "w") as fout:
        for candidate, clip in zip(df["candidate"], clips):
            candidate = extract_k(candidate, k=clip)
            fout.write(str(candidate) + "\n")

    with open(out_path / f"summaries.{step}.candidate_ids_orig", "w") as fout:
        for ids, clip in zip(df["candidate_ids_orig"], clips):
            ids = ids[:clip]
            fout.write(str(ids[:clip]) + "\n")


def main(args):
    step = args.bertext_step
    df_reports = pd.read_json(args.reports_json)
    df_summaries = load_summaries(args.bertext_ranks, step)
    with open(Path(args.bertext_ranks) / f"summaries.{step}.candidate_ids_orig") as fin:
        bertext_ranks = [json.loads(line) for line in fin]

    # id, src, tgt, gold, candidate, candidate_ids_orig
    df = pd.concat(
        [
            df_reports,
            df_summaries[["gold", "candidate"]],
            pd.Series(bertext_ranks, name="candidate_ids_orig"),
        ],
        axis=1,
    )

    out_path = Path(args.out_path_base)
    # k = {1,...,5}
    for k in range(1, 6):
        clips = [k] * len(df)
        write_clipped(df, clips, out_path / f"bertext-default-clip-k{k}", step=step)

    # k = |oracle|
    clips = [len(o) for o in greedy_selection_parallel(df, summary_size=3)]
    write_clipped(df, clips, out_path / "bertext-default-clip-oracle/", step=step)

    # k = |lr_approx|
    clips = load_clips(args.lr_approx_clips)
    write_clipped(df, clips, out_path / "bertext-default-clip-lrapprox/", step=step)

    # k = |bert_approx|
    clips = load_clips(args.bert_approx_clips)
    write_clipped(df, clips, out_path / "bertext-default-clip-bertapprox/", step=step)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports_json",
        help="The raw reports (e.g., reports.test.json). Needed for oracle clipping.",
    )
    parser.add_argument(
        "--bertext_ranks",
        help="Path to BertExt run with full output ranks. This output will be clipped.",
    )
    parser.add_argument(
        "--bertext_step",
        help="The step number of the BertExt. Used to load the summaries.",
    )
    parser.add_argument(
        "--lr_approx_clips",
        help="Predicted summary lengths by LR-Approx. One integer per line.",
    )
    parser.add_argument(
        "--bert_approx_clips",
        help="Predicted summary lengths by BERT-Approx. One integer per line.",
    )
    parser.add_argument(
        "--out_path_base",
        help="Directory where clipped BertExt runs will be stored at.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(arg_parser())

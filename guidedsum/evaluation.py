import contextlib
import os
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from bert_score import BERTScorer
from nltk import ngrams
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
from tqdm.auto import tqdm


def load_summaries(run_path, step):
    run_path = Path(run_path)
    cand_path = run_path / f"summaries.{step}.candidate"
    gold_path = run_path / f"summaries.{step}.gold"

    with open(cand_path) as fin:
        candidates = [line.strip() for line in fin.readlines()]
    with open(gold_path) as fin:
        references = [line.strip() for line in fin.readlines()]

    df_summaries = pd.DataFrame(data={"gold": references, "candidate": candidates})
    return df_summaries


def evaluate_run(
    docs: Iterable[str],
    cands: Iterable[str],
    refs: Iterable[str],
    bertscorer: BERTScorer,
):
    stats = calculate_statistics(docs, cands)
    rouge = calculate_rouge(cands, refs)
    bscore = calculate_bertscore(cands, refs, bertscorer)
    len_delta = calculate_length_delta(cands, refs)
    return pd.concat([stats, rouge, bscore, len_delta], axis=1)


class Aggregator:
    def __init__(self):
        self.stats = defaultdict(list)

    def add(self, scores):
        for metric, value in scores.items():
            self.stats[metric].append(value)


def clean(s):
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s\s+", " ", s)
    s = s.lower()
    return s


def novel_ngrams(a: str, b: str, n=2):
    """Count number of n-grams in b but not in a."""
    a, b = clean(a), clean(b)
    a = set(ngrams(a.split(), n=n))
    b = set(ngrams(b.split(), n=n))
    novel = len(b - a)
    total = len(b)
    return novel, total


def novelty(a, b, n):
    """
    Fraction of n-grams in b but not in a.

    If there are no n-grams in b, this is novelty=0.
    """
    novel, total = novel_ngrams(a, b, n=n)
    if total == 0:
        return 0
    return novel / total


def calculate_statistics(docs: Iterable[str], summaries: Iterable[str]):
    """Surface level statistics comparing documents with summaries.

    Sentence boundaries should be indicated with `<q>` and text is expected to be pre-tokenized.
    """

    agg = Aggregator()

    for doc, summary in zip(docs, summaries):
        n_words_doc = len(doc.replace("<q>", " ").split())
        n_words_summary = len(summary.replace("<q>", " ").split())
        cmp_w = 1 - n_words_summary / n_words_doc

        n_sents_doc = len(doc.split("<q>"))
        n_sents_summary = len(summary.split("<q>"))
        cmp_s = 1 - n_sents_summary / n_sents_doc

        doc_str = doc.replace("<q>", " ")
        summary_str = summary.replace("<q>", " ")
        novelty_uni = novelty(doc_str, summary_str, n=1)
        novelty_bi = novelty(doc_str, summary_str, n=2)

        agg.add(
            {
                "n_words_doc": n_words_doc,
                "n_sents_doc": n_sents_doc,
                "n_words_summary": n_words_summary,
                "n_sents_summary": n_sents_summary,
                "cmp_w": cmp_w,
                "cmp_s": cmp_s,
                "novelty_uni": novelty_uni,
                "novelty_bi": novelty_bi,
            }
        )

    return pd.DataFrame(agg.stats)


def calculate_rouge(cands, refs):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeLsum"], use_stemmer=True
    )
    scores = []
    for c, r in zip(cands, refs):
        c = c.replace("<q>", "\n")
        r = r.replace("<q>", "\n")
        s = scorer.score(r, c)

        scores.append(
            {
                "rouge1": s["rouge1"].fmeasure,
                "rouge1_p": s["rouge1"].precision,
                "rouge1_r": s["rouge1"].recall,
                "rouge2": s["rouge2"].fmeasure,
                "rouge2_p": s["rouge2"].precision,
                "rouge2_r": s["rouge2"].recall,
                "rougeLsum": s["rougeLsum"].fmeasure,
                "rougeLsum_p": s["rougeLsum"].precision,
                "rougeLsum_r": s["rougeLsum"].recall,
            }
        )
    return pd.json_normalize(scores)


def calculate_length_delta(cands: List[str], refs: List[str]):
    """
    Calculates relative difference in length between candidate and reference.

    delta = (|ŷ| - |y|)/|y|
    """
    len_cands = [len(s.replace("<q>", " ").split()) for s in cands]
    len_refs = [len(s.replace("<q>", " ").split()) for s in refs]
    len_cands = np.array(len_cands)
    len_refs = np.array(len_refs)
    deltas = (len_cands - len_refs) / len_refs
    deltas = pd.Series(deltas, name="length_delta")
    return deltas


def calculate_bertscore(cands, refs, scorer):
    cands = [c.replace("<q>", " ") for c in cands]
    refs = [r.replace("<q>", " ") for r in refs]
    empty = [len(c) == 0 for c in cands]

    with open(os.devnull, "w") as f, contextlib.redirect_stderr(f):
        # Suppress print to stderr:
        # "Warning: Empty candidate sentence detected; setting raw BERTscores to 0."
        p, r, f1 = scorer.score(cands, refs)
    df_scores = pd.DataFrame(
        data={
            "bertscore_precision": p.numpy(),
            "bertscore_recall": r.numpy(),
            "bertscore_f1": f1.numpy(),
        }
    )
    # Raw bert scores are set to zero on empty candidate sequence.
    # However, those zero scores get still rescaled when scorer.rescale_with_baseline=True
    # Therefore, we again set them to zero afterwards.
    df_scores.loc[empty, :] = 0
    return df_scores


def chexpert_labels_to_frame(labels: List[dict]):
    """Converts a list of dictionaries to a DataFrame, where one row corresponds to a sample, and the columns are equal to the keys in labels."""
    labels = pd.DataFrame(labels)
    labels = labels.replace(
        {np.nan: "N/A", 1: "Positive", 0: "Negative", -1: "Uncertain"}
    )
    return labels


def calculate_factual_f1(
    candidate_labels: List[dict], reference_labels: List[dict]
) -> pd.Series:
    """Factual F1 for English radiology reports based on facts extracted from CheXpert.

    Given a radiology report, CheXperts labels 14 clinical observations as Positive/Negative/Uncertain. The factual F1 metric compares the labels extracted from the reference and candidate summary. A macro-averaged F1 is computed for each observation. The system total is the mean of all observation F1 scores.

    Possible CheXpert values:
    *  1     = positive: observation is present
    *  0     = negative: observation is not present
    * -1     = uncertain: observation may be present
    * np.nan = observation is not mentioned

    Parameters
    ==========
    candidate_labels: List[dict]
        Labels are expected to be in the form of `{observation: label}` where label in [1, 0, -1, np.nan].

    reference_labels: List[dict]
        See above.

    References
    ==========
    * Irvin, J., et al. (2019). CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison. AAAI 2019. https://doi.org/10.1609/aaai.v33i01.3301590
    * Zhang, Y., Merck, D., Tsai, E. B., Manning, C. D., & Langlotz, C. P. (2020). Optimizing the Factual Correctness of a Summary: A Study of Summarizing Radiology Reports. ACL 2020. https://doi.org/10.18653/v1/2020.acl-main.458
    """
    candidate_labels = chexpert_labels_to_frame(candidate_labels)
    reference_labels = chexpert_labels_to_frame(reference_labels)
    scores = {}
    for c in reference_labels.columns:
        scores[c] = f1_score(reference_labels[c], candidate_labels[c], average="macro")
    return pd.Series(scores)


def bootstrap(y_true, run_a, run_b, R, metric):
    """
    Repeat R times: randomly create new samples from the data with repetitions, calculate delta(A,B).

    Let r be the number of times that delta(A,B) > 2*orig_delta(A,B).
    Significance level: r/R

    Reference:
    * Berg-Kirkpatrick et al. (2012). An Empirical Investigation of Statistical Significance in NLP.
    * Jurafsky, D., Martin, J. (2023). Speech and Language Processing. Chapter 4, https://web.stanford.edu/~jurafsky/slp3/
    """
    delta_orig = metric(y_true, run_a) - metric(y_true, run_b)
    r = 0
    n = len(y_true)
    for x in tqdm(range(0, R), total=R):
        samples = np.random.randint(
            0, n, n
        )  # which samples to add to the subsample with repetitions
        temp_a = run_a[samples]
        temp_b = run_b[samples]
        x = metric(y_true, temp_a)
        y = metric(y_true, temp_b)
        delta = x - y
        if delta > 2 * delta_orig:
            r = r + 1
    pval = float(r) / (R)
    return pval

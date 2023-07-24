"""
Adds sentence-level guidance through an extractive oracle [1]. Implementation based on [2]. At test time, a pre-selected list of sentences can be used (e.g., from an extractive summarization method).

It differs from [1,2] in the sense that oracle sentences are extracted before any source document filtering is done. In the original implementation, `data_builder.py` removes sentences that are shorter than N words from the input document, even if they would be part of the oracle. In our experience, the resulting ROUGE scores are similar.

References
----------
[1] Dou et al. (2021). GSum: A General Framework for Guided Neural Abstractive Summarization. NAACL 2021.
[2] https://github.com/nlpyang/PreSumm/blob/70b810e0f06d179022958dd35c1a3385fe87f28c/src/prepro/data_builder.py#L309
"""
import argparse
import json
import multiprocessing as mp
import re
from itertools import repeat
from pathlib import Path

import pandas as pd


def _get_ngrams(n, text):
    """Calcualtes n-grams.

    Args:
      n: which n-grams to calculate
      text: An array of tokens

    Returns:
      A set of n-grams
    """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i : i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    """Calculates word n-grams for multiple sentences."""
    assert len(sentences) > 0
    assert n > 0
    words = sum(sentences, [])
    return _get_ngrams(n, words)


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    """Greedily select sentences in source document that maximize ROUGE score with respect to the target summary. This routine creates ground truth sentence labels for extractive summarization.

    The summary is build up iteratively up to a length of `summary_size`. At each iteration, a sentence is only added if the additon improves the ROUGE score with respect to the target summary.

    References:
    - Nallapati, R. et al. (2017). SummaRuNNer: A Recurrent Neural Network Based Sequence Model for Extractive Summarization of Documents. AAAI-17.
    - Liu, Y., & Lapata, M. (2019). Text Summarization with Pretrained Encoders. EMNLP-IJCNLP 2019.

    Parameters
    ----------
    doc_sent_list : List[List[str]]
        Tokenized sentences in source document.
    abstract_sent_list : List[List[str]]
        Tokenized sentences in target summary.
    summary_size : int
        Maximum number of sentences to select.

    Returns
    -------
    List[int]
        Indexes of the selected sentences in `doc_sent_list`.
    """

    def _rouge_clean(s):
        s = s.lower()
        return re.sub(r"[^a-zA-Z0-9 ]", "", s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(" ".join(abstract)).split()
    sents = [_rouge_clean(" ".join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    for _ in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if i in selected:
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)["f"]
            rouge_2 = cal_rouge(candidates_2, reference_2grams)["f"]
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if cur_id == -1:
            break
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)


def select_sentences(args):
    src, tgt, n = args
    return greedy_selection(src, tgt, n)


def greedy_selection_parallel(df, summary_size=3):
    with mp.Pool(mp.cpu_count()) as pool:
        return pool.map(
            select_sentences, zip(df["src"], df["tgt"], repeat(summary_size))
        )


def main(args):
    in_path = Path(args.input_path)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    df = pd.read_json(in_path)
    if "z" in df.columns and "z_ids" in df.columns:
        # if df already has guidance, skip and only handle abstain.
        pass
    else:
        if args.use_oracle:
            df["z_ids"] = greedy_selection_parallel(df)
        else:
            with open(args.z_ids_path) as fin:
                df["z_ids"] = [json.loads(line) for line in fin]
        select_by_index = lambda row: [row.src[i] for i in row.z_ids]
        df["z"] = df.apply(select_by_index, axis=1)

    if args.abstain_labels_path:
        with open(args.abstain_labels_path) as fin:
            abstain = [bool(int(line.strip())) for line in fin]

        # z should be a list of sentences of tokens (List[List[str]])
        z_abstain = [args.z_abstain.split()]
        # We repeat z the number of times we abstain. For z_ids we just repeat an empty list.
        z_abstain = pd.Series([z_abstain] * sum(abstain)).values
        z_ids_abstain = pd.Series([[]] * sum(abstain)).values
        df.loc[abstain, "z"] = z_abstain
        df.loc[abstain, "z_ids"] = z_ids_abstain

    cols = ["id", "src", "tgt", "z", "z_ids"]
    df[cols].to_json(out_path, orient="records")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", help="Path to input json (cols = [id, src, tgt])"
    )
    parser.add_argument(
        "--output_path", help="Path to output json (cols = [id, src, tgt, z, z_ids])"
    )
    parser.add_argument(
        "--use_oracle",
        help="Use extractive oracle as guidance signal.",
        action="store_true",
    )
    parser.add_argument(
        "--z_ids_path",
        help="Use these src sentences as guidance signal. One line per instance with a list of sentence IDs.",
    )
    parser.add_argument(
        "--abstain_labels_path",
        help="Instances where no guidance should be added. One line per instance with a boolean (true=abstain).",
    )
    parser.add_argument(
        "--z_abstain",
        type=str,
        help="Fixed string to use as guidance signals for instances which are abstained.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(arg_parser())

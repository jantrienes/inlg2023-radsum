import dataclasses
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np


@dataclasses.dataclass(frozen=True)
class Annotation:
    start: int
    end: int
    label: str
    annotator: str
    document_id: object = None


def group_spans(annotations: Dict[str, List[Annotation]]) -> List[List[Annotation]]:
    """
    Get groups of (partially) overlapping annotations.

    `annotations` should a dictionary mapping annotator name to the list of annotations.
    """
    all_spans = []
    for annotator in annotations.values():
        all_spans += annotator
    sorted_spans = sorted(all_spans, key=lambda s: s.start)

    groups = []
    current_group = []
    current_end = -1
    for span in sorted_spans:
        if span.start >= current_end:
            if current_group:
                groups.append(current_group)
                current_group = []
            current_end = span.end
        current_group.append(span)
        current_end = max(current_end, span.end)
    if current_group:
        groups.append(current_group)

    return groups


def label_groups(groups: List[List[Annotation]]) -> List[str]:
    labels = []

    for group in groups:
        offsets = [(ann.start, ann.end) for ann in group]
        annotators = Counter(ann.annotator for ann in group)
        if len(annotators) == 1:
            # Case 1: only one annotator
            label = "SINGLE"
        elif len(set(offsets)) == 1:
            # Case 2: all annotations align perfectly
            label = "ACCEPT"
        elif any(count > 1 for count in annotators.values()):
            # Case 3: at least one annotator has more than one annotation involved
            label = "REVIEW"
        else:
            # Case 4: partially overlap can extend
            label = "EXTEND"
        labels.append(label)

    return labels


def expand_to_borders(annotations: List[Annotation]) -> List[Annotation]:
    min_start = min(ann.start for ann in annotations)
    max_end = max(ann.end for ann in annotations)

    adjusted = []
    for ann in annotations:
        copy = dataclasses.replace(ann, start=min_start, end=max_end)
        adjusted.append(copy)

    return adjusted


def exact_vote(annotations: Dict[str, List[Annotation]], expand=False):
    """Exact voting: only one span per annotator and all spans offsets have to match exactly.

    Parameters
    ----------
    annotations : Dict[str, List[Annotation]]
        Annotations per annotator
    expand : bool, optional
        If True, this relaxes the offset criterion by setting the start/end offsets to [min(start),max(end)].

    Returns
    -------
    Dict[str, int]
        A dictionary with {label: count}. In exact voting, this can either be empty or {label: 1}.
    """
    anns_flat = []
    for anns in annotations.values():
        if len(anns) != 1:
            # Each annotator must have exactly one annotation.
            return {}

        for ann in anns:
            anns_flat.append(Annotation(ann.start, ann.end, ann.label, ""))

    if expand:
        anns_flat = expand_to_borders(anns_flat)

    distinct = set(anns_flat)
    if len(distinct) == 1:
        return {anns_flat[0].label: 1}
    return {}


def span_blind_majority_vote(annotations: Dict[str, List[Annotation]]):
    """
    Majority vote (span blind): count the support for each label. Return the maximum number of times the label has support by at least two annotators.

    Completely disregards annotation spans (we only apply it to partially overlapping annotations to reduce false-positives).

    Example 1:
    a1: e1, e1
    a2: e1
    a3: a2
    ----> {e1: 1}

    Example 2:
    a1: e1, e1
    a2: e1
    a3: e1, e1, e1
    ----> {e1: 2}
    """
    anns_flat = [ann for anns in annotations.values() for ann in anns]

    # Accumulate annotators by label
    labels = defaultdict(list)
    for a in anns_flat:
        labels[a.label].append(a.annotator)

    maximum_with_support = {}
    for label, annotators in labels.items():
        votes = Counter(
            annotators
        ).values()  # Count how often each annotator assigned the label.
        votes = sorted(votes)
        # Get the maximum number of times this label is supported by at least two annotators.
        current_max = 0
        candidate = votes[0]
        for i in votes[1:]:
            if i >= candidate:
                current_max = candidate
                candidate = i
        if current_max > 0:
            maximum_with_support[label] = current_max

    return maximum_with_support


def sum_counters(counts: List[dict]):
    total = Counter()
    for d in counts:
        total.update(d)
    return total


def apply_votings(annotations: Dict[str, List[Annotation]]):
    groups = group_spans(annotations)

    exact = []
    exact_relaxed = []
    majority = []
    for group in groups:
        anns_grouped = {annotator: [] for annotator in annotations.keys()}
        for ann in group:
            anns_grouped[ann.annotator].append(ann)

        exact.append(exact_vote(anns_grouped))
        exact_relaxed.append(exact_vote(anns_grouped, expand=True))
        majority.append(span_blind_majority_vote(anns_grouped))

    result = {
        "groups": groups,
        "exact_vote": exact,
        "relaxed_exact_vote": exact_relaxed,
        "span_blind_majority_vote": majority,
        "exact_vote_total": sum_counters(exact),
        "relaxed_exact_vote_total": sum_counters(exact_relaxed),
        "span_blind_majority_vote_total": sum_counters(majority),
    }
    return result


def precision(correct, actual):
    if actual == 0:
        return 0
    return correct / actual


def recall(correct, possible):
    if possible == 0:
        return 0
    return correct / possible


def f1(p, r):
    if p + r == 0:
        return 0
    return 2 * (p * r) / (p + r)


def annotation_set(annotations, label=None):
    if label:
        annotations = [a for a in annotations if a.label == label]
    # drop the annotator ID such that it is not part of hash when we compute set overlap
    annotations = [
        Annotation(
            start=a.start,
            end=a.end,
            label=a.label,
            annotator="",
            document_id=a.document_id,
        )
        for a in annotations
    ]
    return set(annotations)


def annotator_agreement(
    annotations_a: List[Annotation], annotations_b: List[Annotation], label=None
):
    annotations_a = annotation_set(annotations_a, label)
    annotations_b = annotation_set(annotations_b, label)

    correct = len(annotations_a & annotations_b)
    positive = len(annotations_a)
    predicted = len(annotations_b)

    if positive <= 0:
        return np.NaN

    p = precision(correct, predicted)
    r = recall(correct, positive)
    return f1(p, r)


def annotator_agreement_three(
    annotations_a: List[Annotation],
    annotations_b: List[Annotation],
    annotations_c: List[Annotation],
    label=None,
):
    a_b = annotator_agreement(annotations_a, annotations_b, label=label)
    a_c = annotator_agreement(annotations_a, annotations_c, label=label)
    b_c = annotator_agreement(annotations_b, annotations_c, label=label)
    return (a_b + a_c + b_c) / 3

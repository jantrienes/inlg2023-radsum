import numpy as np

from guidedsum.annotation_utils import Annotation as A
from guidedsum.annotation_utils import (
    annotator_agreement,
    apply_votings,
    exact_vote,
    expand_to_borders,
    group_spans,
    label_groups,
    span_blind_majority_vote,
)


def test_group_spans():
    annotations = {
        "a1": [
            A(0, 2, "FOO", "a1"),
            A(3, 5, "BAR", "a1"),
            A(6, 7, "FOO", "a1"),
            A(8, 10, "BAR", "a1"),
            A(12, 14, "BAR", "a1"),
            A(15, 17, "BAR", "a1"),
        ],
        "a2": [
            A(0, 2, "FOO", "a2"),
            A(3, 4, "BAR", "a2"),
            A(6, 7, "BAR", "a2"),
            A(8, 9, "BAR", "a2"),
            A(18, 19, "FOO", "a2"),
        ],
        "a3": [A(0, 2, "FOO", "a3"), A(9, 11, "BAR", "a3"), A(13, 16, "BAR", "a3")],
    }
    expected = [
        [
            A(0, 2, "FOO", "a1"),
            A(0, 2, "FOO", "a2"),
            A(0, 2, "FOO", "a3"),
        ],
        [
            A(3, 5, "BAR", "a1"),
            A(3, 4, "BAR", "a2"),
        ],
        [
            A(6, 7, "FOO", "a1"),
            A(6, 7, "BAR", "a2"),
        ],
        [A(8, 10, "BAR", "a1"), A(8, 9, "BAR", "a2"), A(9, 11, "BAR", "a3")],
        [
            A(12, 14, "BAR", "a1"),
            A(13, 16, "BAR", "a3"),
            A(15, 17, "BAR", "a1"),
        ],
        [
            A(18, 19, "FOO", "a2"),
        ],
    ]
    assert group_spans(annotations) == expected


def test_label_groups():
    groups = [
        [
            A(0, 2, "FOO", "a1"),
            A(0, 2, "FOO", "a2"),
            A(0, 2, "FOO", "a3"),
        ],
        [
            A(3, 5, "BAR", "a1"),
            A(3, 4, "BAR", "a2"),
        ],
        [
            A(6, 7, "FOO", "a1"),
            A(6, 7, "BAR", "a2"),
        ],
        [A(8, 10, "BAR", "a1"), A(8, 9, "BAR", "a2"), A(9, 11, "BAR", "a3")],
        [
            A(12, 14, "BAR", "a1"),
            A(13, 16, "BAR", "a3"),
            A(15, 17, "BAR", "a1"),
        ],
        [
            A(18, 19, "FOO", "a2"),
        ],
    ]
    expected = ["ACCEPT", "EXTEND", "ACCEPT", "EXTEND", "REVIEW", "SINGLE"]
    assert label_groups(groups) == expected


def test_expand_to_borders():
    annotations = [A(8, 10, "BAR", "a1"), A(8, 9, "BAR", "a2"), A(9, 11, "BAR", "a3")]
    expected = [A(8, 11, "BAR", "a1"), A(8, 11, "BAR", "a2"), A(8, 11, "BAR", "a3")]
    assert expand_to_borders(annotations) == expected


def test_exact_vote():
    annotations = {
        "a1": [A(1, 5, "FOO", "a1")],
        "a2": [A(1, 5, "FOO", "a2")],
        "a3": [A(1, 5, "FOO", "a3")],
    }
    assert exact_vote(annotations) == {"FOO": 1}

    annotations = {
        "a1": [],
        "a2": [],
        "a3": [],
    }
    assert exact_vote(annotations) == {}

    annotations = {
        "a1": [A(1, 5, "FOO", "a1")],
        "a2": [],
        "a3": [],
    }
    assert exact_vote(annotations) == {}

    annotations = {"a1": [A(1, 5, "FOO", "a1")], "a2": [A(1, 5, "FOO", "a2")], "a3": []}
    assert exact_vote(annotations) == {}

    annotations = {
        "a1": [A(1, 5, "FOO", "a1")],
        "a2": [A(1, 5, "FOO", "a2")],
        "a3": [A(1, 5, "BAR", "a3")],
    }
    assert exact_vote(annotations) == {}

    annotations = {
        "a1": [A(1, 5, "FOO", "a1")],
        "a2": [A(1, 2, "FOO", "a2"), A(2, 5, "FOO", "a2")],
        "a3": [A(1, 5, "FOO", "a3")],
    }
    assert exact_vote(annotations) == {}

    annotations = {
        "a1": [A(1, 5, "FOO", "a1")],
        "a2": [A(1, 5, "FOO", "a2")],
        "a3": [A(1, 3, "FOO", "a3")],
    }
    assert exact_vote(annotations) == {}
    assert exact_vote(annotations, expand=True) == {"FOO": 1}

    annotations = {
        "a1": [A(12, 14, "BAR", "a1"), A(15, 17, "BAR", "a1")],
        "a2": [],
        "a3": [A(13, 16, "BAR", "a3")],
    }
    assert exact_vote(annotations) == {}
    assert exact_vote(annotations, expand=True) == {}


def test_span_blind_majority_vote():
    annotations = {
        "a1": [A(1, 5, "FOO", "a1")],
        "a2": [A(1, 5, "FOO", "a2")],
        "a3": [A(1, 5, "FOO", "a3")],
    }
    assert span_blind_majority_vote(annotations) == {"FOO": 1}

    annotations = {"a1": [A(1, 5, "FOO", "a1")], "a2": [], "a3": []}
    assert span_blind_majority_vote(annotations) == {}

    annotations = {"a1": [A(1, 5, "FOO", "a1")], "a2": [A(1, 5, "FOO", "a2")], "a3": []}
    assert span_blind_majority_vote(annotations) == {"FOO": 1}

    annotations = {
        "a1": [A(1, 5, "FOO", "a1")],
        "a2": [A(1, 5, "FOO", "a2")],
        "a3": [A(1, 5, "BAR", "a3")],
    }
    assert span_blind_majority_vote(annotations) == {"FOO": 1}

    annotations = {
        "a1": [A(1, 5, "FOO", "a1")],
        "a2": [A(1, 2, "FOO", "a2"), A(2, 5, "FOO", "a2")],
        "a3": [A(1, 5, "FOO", "a3")],
    }
    assert span_blind_majority_vote(annotations) == {"FOO": 1}

    annotations = {"a1": [A(1, 5, "FOO", "a1")], "a2": [], "a3": [A(1, 3, "FOO", "a3")]}
    assert span_blind_majority_vote(annotations) == {"FOO": 1}

    annotations = {
        "a1": [A(1, 5, "FOO", "a1")],
        "a2": [],
        "a3": [A(1, 2, "FOO", "a3"), A(3, 5, "FOO", "a3")],
    }
    assert span_blind_majority_vote(annotations) == {"FOO": 1}

    annotations = {
        "a1": [A(1, 5, "FOO", "a1")],
        "a2": [A(1, 2, "FOO", "a2")],
        "a3": [A(1, 2, "FOO", "a3"), A(3, 5, "FOO", "a3")],
    }
    assert span_blind_majority_vote(annotations) == {"FOO": 1}

    annotations = {
        "a1": [A(1, 5, "FOO", "a1")],
        "a2": [A(1, 2, "FOO", "a2"), A(3, 4, "FOO", "a2")],
        "a3": [A(1, 2, "FOO", "a3"), A(3, 5, "FOO", "a3")],
    }
    assert span_blind_majority_vote(annotations) == {"FOO": 2}

    annotations = {
        "a1": [A(1, 2, "FOO", "a1"), A(3, 5, "FOO", "a1")],
        "a2": [A(1, 2, "FOO", "a2"), A(3, 4, "BAR", "a2")],
        "a3": [A(1, 2, "FOO", "a3"), A(3, 5, "BAR", "a3")],
    }
    assert span_blind_majority_vote(annotations) == {"FOO": 1, "BAR": 1}

    annotations = {
        "a1": [A(1, 5, "FOO", "a1")],
        "a2": [A(1, 2, "FOO", "a2"), A(3, 4, "BAR", "a2")],
        "a3": [A(3, 5, "BAR", "a3")],
    }
    assert span_blind_majority_vote(annotations) == {"FOO": 1, "BAR": 1}

    annotations = {
        "a1": [A(1, 5, "FOO", "a1")],
        "a2": [A(1, 2, "BAR", "a2")],
        "a3": [A(3, 5, "BAR", "a3")],
    }
    assert span_blind_majority_vote(annotations) == {
        "BAR": 1
    }, "Should be blind to the actual spans"

    annotations = {
        "a1": [A(12, 14, "BAR", "a1"), A(15, 17, "BAR", "a1")],
        "a2": [],
        "a3": [A(13, 16, "BAR", "a3")],
    }
    assert span_blind_majority_vote(annotations) == {"BAR": 1}


def test_apply_votings():
    annotations = {
        "a1": [
            A(0, 2, "FOO", "a1"),
            A(3, 5, "BAR", "a1"),
            A(6, 7, "FOO", "a1"),
            A(8, 10, "BAR", "a1"),
            A(12, 14, "BAR", "a1"),
            A(15, 17, "BAR", "a1"),
            A(20, 25, "FOO", "a1"),
        ],
        "a2": [
            A(0, 2, "FOO", "a2"),
            A(3, 4, "BAR", "a2"),
            A(6, 7, "BAR", "a2"),
            A(8, 9, "BAR", "a2"),
            A(18, 19, "FOO", "a2"),
            A(20, 22, "FOO", "a2"),
            A(22, 25, "FOO", "a2"),
        ],
        "a3": [
            A(0, 2, "FOO", "a3"),
            A(9, 11, "BAR", "a3"),
            A(13, 16, "BAR", "a3"),
            A(20, 22, "FOO", "a3"),
            A(22, 23, "FOO", "a3"),
            A(23, 25, "FOO", "a3"),
        ],
    }
    expected_groups = [
        # exact/relaxed/foo: FOO/FOO/FOO
        [
            A(0, 2, "FOO", "a1"),
            A(0, 2, "FOO", "a2"),
            A(0, 2, "FOO", "a3"),
        ],
        # exact/relaxed/foo: -/-/BAR
        [
            A(3, 5, "BAR", "a1"),
            A(3, 4, "BAR", "a2"),
        ],
        # exact/relaxed/foo: -/-/-
        [
            A(6, 7, "FOO", "a1"),
            A(6, 7, "BAR", "a2"),
        ],
        # exact/relaxed/foo: -/-/BAR
        [A(8, 10, "BAR", "a1"), A(8, 9, "BAR", "a2"), A(9, 11, "BAR", "a3")],
        # exact/relaxed/foo: -/-/BAR
        [
            A(12, 14, "BAR", "a1"),
            A(13, 16, "BAR", "a3"),
            A(15, 17, "BAR", "a1"),
        ],
        # exact/relaxed/foo: -/-/-
        [
            A(18, 19, "FOO", "a2"),
        ],
        # exact/relaxed/foo: -/-/2x FOO
        [
            A(20, 25, "FOO", "a1"),
            A(20, 22, "FOO", "a2"),
            A(20, 22, "FOO", "a3"),
            A(22, 25, "FOO", "a2"),
            A(22, 23, "FOO", "a3"),
            A(23, 25, "FOO", "a3"),
        ],
    ]

    result = apply_votings(annotations)
    assert result["groups"] == expected_groups
    assert result["exact_vote"] == [{"FOO": 1}, {}, {}, {}, {}, {}, {}]
    assert result["relaxed_exact_vote"] == [{"FOO": 1}, {}, {}, {"BAR": 1}, {}, {}, {}]
    assert result["span_blind_majority_vote"] == [
        {"FOO": 1},
        {"BAR": 1},
        {},
        {"BAR": 1},
        {"BAR": 1},
        {},
        {"FOO": 2},
    ]
    assert result["exact_vote_total"] == {"FOO": 1}
    assert result["relaxed_exact_vote_total"] == {"FOO": 1, "BAR": 1}
    assert result["span_blind_majority_vote_total"] == {"FOO": 3, "BAR": 3}


def test_annotator_agreement():
    annotator_a = [
        A(0, 5, "ORG", "a1", "doc1"),
        A(6, 8, "ORG", "a1", "doc1"),
        A(10, 15, "COUNTRY", "a1", "doc2"),
    ]
    annotator_b = [A(0, 5, "ORG", "a2", "doc1"), A(6, 8, "LOCATION", "a2", "doc1")]

    r = 1 / 3
    p = 1 / 2
    f1 = 2 * (p * r) / (p + r)
    assert annotator_agreement(annotator_a, annotator_b) == f1

    r = 1 / 2  # 2x ORG
    p = 1  # 1x ORG
    f1 = 2 * (p * r) / (p + r)
    assert annotator_agreement(annotator_a, annotator_b, label="ORG") == f1
    assert annotator_agreement(annotator_a, annotator_b, label="COUNTRY") == 0
    assert np.isnan(annotator_agreement(annotator_a, annotator_b, label="LOCATION"))

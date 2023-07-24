import pandas as pd

from guidedsum.guidance.extractive_guidance import (
    greedy_selection,
    greedy_selection_parallel,
)


def test_greedy_selection():
    src = [
        "this sentence is about an apple .".split(),
        "this is completely unrelated.".split(),
        "and this text about an orange .".split(),
    ]

    tgt = ["this text is about an apple and orange".split()]
    assert greedy_selection(src, tgt, summary_size=2) == [0, 2]

    src[0] = [token.upper() for token in src[0]]
    assert greedy_selection(src, tgt, summary_size=2) == [
        0,
        2,
    ], "Should be insensitive to case."

    df = pd.DataFrame({"src": [src], "tgt": [tgt]})
    result = greedy_selection_parallel(df, summary_size=2)
    assert result[0] == [
        0,
        2,
    ], "Parallel version should give the same output as linear version. "

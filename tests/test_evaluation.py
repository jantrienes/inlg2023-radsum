import pandas as pd
from pandas.testing import assert_frame_equal

from guidedsum.evaluation import chexpert_labels_to_frame


def test_chexpert_labels_to_frame():
    actual = chexpert_labels_to_frame([{"a": 0, "b": 1}, {"a": -1, "c": 0}])
    expected = pd.DataFrame(
        {
            "a": ["Negative", "Uncertain"],
            "b": ["Positive", "N/A"],
            "c": ["N/A", "Negative"],
        }
    )
    assert_frame_equal(actual, expected)

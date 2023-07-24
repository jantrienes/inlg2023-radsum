import nltk
import pytest

from guidedsum.mimic.base import is_abstractive


@pytest.fixture(scope="module", autouse=True)
def my_fixture():
    nltk.download("punkt")


def test_is_abstractive():
    assert is_abstractive("No acute cardiopulmonary process.")
    assert is_abstractive("No evidence of acute cardiopulmonary disease.")
    assert not is_abstractive(
        "No focal consolidation concerning for pneumonia is identified."
    )
    assert not is_abstractive(
        "Mild-to-moderate cardiomegaly. No evidence of acute disease."
    )
    assert not is_abstractive(
        "No acute cardiopulmonary abnormality apart from minimal bibasilar atelectasis."
    )

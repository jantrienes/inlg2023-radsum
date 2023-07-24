from guidedsum.guidance.bertext_clipped import extract_k


def test_extract_k():
    s = "this is a test .<q>this is the second sentence .<q>This is the third sentence"
    assert extract_k(s, k=None) == s
    assert extract_k(s, k=2) == "this is a test .<q>this is the second sentence ."
    assert extract_k(s, k=1) == "this is a test ."

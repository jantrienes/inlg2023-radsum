from guidedsum.mimic.preprocess import combine_background_findings


def test_combine_background_findings():
    report = {
        "indication": "this is the indication",
        "history": "and some history",
        "findings": "findings",
    }
    expected = (
        "Indication:\n"
        "this is the indication\n\n"
        "History:\n"
        "and some history\n\n"
        "Findings:\n"
        "findings"
    )
    assert combine_background_findings(report) == expected

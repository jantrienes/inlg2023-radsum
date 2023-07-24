import re

import spacy
from nltk import sent_tokenize

NLP = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])


def tokenize(text):
    doc = NLP(text)
    return [token.text for token in doc]


def preprocess(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    sents = []
    for sent in sent_tokenize(text):
        sents.append(tokenize(sent))
    return sents


def is_abstractive(impression: str) -> bool:
    match = re.search("with|specifically|aside|and|apart|,", impression, re.IGNORECASE)
    return "acute" in impression and len(sent_tokenize(impression)) < 2 and not match

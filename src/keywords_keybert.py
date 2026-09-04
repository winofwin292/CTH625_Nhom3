"""Phương pháp 2: KeyBERT + vietnamese-bi-encoder.

Model card Hugging Face yêu cầu đầu vào đã tách từ, từ đa tiếng viết với gạch dưới.
Ứng viên là token underthesea (đã là từ tiếng Việt), không ghép bigram hai từ đứng cạnh nhau.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer

from src.config import EMBEDDING_MODEL
from src.preprocess import keep_token, load_stopwords, to_phobert_token, tokenize_words

_keybert_model = None
MAX_UNIGRAM_SYLLABLES = 4
MAX_CANDIDATES = 80


def get_keybert():
    global _keybert_model
    if _keybert_model is None:
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(EMBEDDING_MODEL)
        _keybert_model = KeyBERT(model=encoder)
    return _keybert_model


def _syllable_count(token: str) -> int:
    return len(token.replace("_", " ").split())


def _is_ascii_term(token: str) -> bool:
    core = token.replace("_", "")
    if len(core) < 2:
        return False
    return all(ord(char) < 128 for char in core)


def _keep_unigram(token: str) -> bool:
    if not token or token.replace("_", "").isdigit():
        return False
    bits = token.replace("_", " ").split()
    if len(bits) >= 2 and bits[0].lower() == bits[1].lower():
        return False
    if _is_ascii_term(token):
        return True
    count = _syllable_count(token)
    return 2 <= count <= MAX_UNIGRAM_SYLLABLES


def _analyzer(doc: str) -> list[str]:
    return [token for token in doc.split() if _keep_unigram(token)]


def _score(value: object) -> float:
    item = value.item() if hasattr(value, "item") else value
    return float(item)


def extract_keybert(text: str, top_n: int = 10) -> list[tuple[str, float]]:
    stopwords = load_stopwords()
    tokens = [
        to_phobert_token(token)
        for token in tokenize_words(text)
        if keep_token(token, stopwords)
    ]
    segmented = " ".join(tokens)
    if not segmented:
        return []
    vectorizer = CountVectorizer(
        analyzer=_analyzer,
        lowercase=True,
        max_features=MAX_CANDIDATES,
    )
    pairs = get_keybert().extract_keywords(
        segmented,
        vectorizer=vectorizer,
        top_n=top_n,
        use_mmr=True,
        diversity=0.5,
    )
    return [(term.replace("_", " "), _score(score)) for term, score in pairs]

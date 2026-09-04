"""Phương pháp 2: KeyBERT + vietnamese-bi-encoder.

Model card Hugging Face yêu cầu đầu vào đã tách từ, từ đa tiếng viết với gạch dưới.
"""

from __future__ import annotations

from src.config import EMBEDDING_MODEL
from src.preprocess import load_stopwords, to_phobert_text, tokenize_words

_keybert_model = None


def get_keybert():
    global _keybert_model
    if _keybert_model is None:
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(EMBEDDING_MODEL)
        _keybert_model = KeyBERT(model=encoder)
    return _keybert_model


def extract_keybert(text: str, top_n: int = 10) -> list[tuple[str, float]]:
    segmented = to_phobert_text(tokenize_words(text))
    if not segmented:
        return []
    stopwords = sorted(load_stopwords())
    pairs = get_keybert().extract_keywords(
        segmented,
        keyphrase_ngram_range=(1, 2),
        stop_words=stopwords,
        top_n=top_n,
        use_mmr=True,
        diversity=0.5,
    )
    return [(term.replace("_", " "), float(score)) for term, score in pairs]

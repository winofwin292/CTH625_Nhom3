"""Phương pháp 1: TF-IDF + lọc từ loại N/V/A."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import TFIDF_VECTORIZER_PATH
from src.preprocess import (
    content_tokens,
    keep_keyword_unit,
    to_phobert_token,
    tokens_as_document,
    tokenize_sentences,
    tokenize_words,
)

NGRAM_RANGE = (1, 2)


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=str.split,
        lowercase=True,
        token_pattern=None,
        ngram_range=NGRAM_RANGE,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
    )


def prepare_document(text: str) -> str:
    return tokens_as_document(content_tokens(text, use_pos=True))


def fit_tfidf(corpus_texts: list[str], save_path: Path | None = None) -> TfidfVectorizer:
    documents = []
    for index, text in enumerate(corpus_texts, start=1):
        prepared = prepare_document(text)
        if prepared:
            documents.append(prepared)
        if len(corpus_texts) >= 20 and (index % 25 == 0 or index == len(corpus_texts)):
            print(f"TF-IDF preprocess {index}/{len(corpus_texts)}")
    if not documents:
        raise ValueError("Corpus sau tiền xử lý bị rỗng, không fit được TF-IDF.")
    model = _vectorizer()
    model.fit(documents)
    path = save_path or TFIDF_VECTORIZER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model


def load_tfidf(path: Path | None = None) -> TfidfVectorizer | None:
    file_path = path or TFIDF_VECTORIZER_PATH
    if not file_path.exists():
        return None
    return joblib.load(file_path)


def _adjacent_bigrams(text: str) -> set[str]:
    tokens = [to_phobert_token(token).lower() for token in tokenize_words(text) if token.strip()]
    return {f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)}


def _keep_tfidf_term(feature: str, adjacent: set[str]) -> bool:
    parts = feature.split()
    if len(parts) == 1:
        return keep_keyword_unit(parts[0])
    if len(parts) != 2:
        return False
    if feature.lower() not in adjacent:
        return False
    return keep_keyword_unit(parts[0]) and keep_keyword_unit(parts[1])


def _top_from_vector(model: TfidfVectorizer, text: str, top_n: int) -> list[tuple[str, float]]:
    """Lấy top-k sau khi bỏ unigram quá ngắn và bigram không kề nhau trên văn bản gốc."""
    prepared = prepare_document(text)
    if not prepared:
        return []
    matrix = model.transform([prepared])
    row = matrix.tocoo()
    scored = list(zip(row.col, row.data))
    scored.sort(key=lambda item: item[1], reverse=True)
    names = model.get_feature_names_out()
    adjacent = _adjacent_bigrams(text)
    results: list[tuple[str, float]] = []
    for index, score in scored:
        feature = names[index]
        if not _keep_tfidf_term(feature, adjacent):
            continue
        results.append((feature.replace("_", " "), float(score)))
        if len(results) >= top_n:
            break
    return results


def extract_tfidf(text: str, top_n: int = 10, vectorizer: TfidfVectorizer | None = None) -> list[tuple[str, float]]:
    """Dùng vectorizer đã fit nếu có; nếu chưa train thì lấy câu làm 'văn bản' để tính IDF nội bộ."""
    model = vectorizer or load_tfidf()
    if model is not None:
        return _top_from_vector(model, text, top_n)

    sentences = tokenize_sentences(text)
    documents = [prepare_document(sentence) for sentence in sentences]
    documents = [doc for doc in documents if doc]
    if not documents:
        documents = [prepare_document(text)]
    if not documents or not documents[0]:
        return []
    fallback = _vectorizer()
    fallback.fit(documents)
    return _top_from_vector(fallback, text, top_n)

"""Một điểm gọi duy nhất cho web và notebook."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import KEYWORD_METHODS, METHOD_KEYBERT, METHOD_TFIDF
from src.keywords_tfidf import extract_tfidf
from src.preprocess import content_tokens, tokenize_words
from src.storage import save_run
from src.summarizer import summarize


@dataclass
class PipelineResult:
    text: str
    method: str
    tokens: list[str]
    content_words: list[tuple[str, str]]
    keywords: list[tuple[str, float]] = field(default_factory=list)
    summary: str = ""
    stored_id: str = ""
    warning: str = ""


def extract_keywords(text: str, method: str, top_n: int) -> list[tuple[str, float]]:
    if method == METHOD_TFIDF:
        return extract_tfidf(text, top_n=top_n)
    if method == METHOD_KEYBERT:
        from src.keywords_keybert import extract_keybert

        return extract_keybert(text, top_n=top_n)
    raise ValueError(f"method phải là một trong {KEYWORD_METHODS}, nhận: {method}")


def run(
    text: str,
    method: str = METHOD_TFIDF,
    top_n: int = 10,
    do_summary: bool = True,
    n_sentences: int = 5,
    persist: bool = True,
    source: str = "input",
) -> PipelineResult:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Văn bản đầu vào đang rỗng.")
    if method not in KEYWORD_METHODS:
        raise ValueError(f"method phải là một trong {KEYWORD_METHODS}, nhận: {method}")

    result = PipelineResult(
        text=cleaned,
        method=method,
        tokens=tokenize_words(cleaned),
        content_words=content_tokens(cleaned, use_pos=True),
        keywords=extract_keywords(cleaned, method, top_n),
    )

    if do_summary:
        keyword_terms = [term for term, _ in result.keywords]
        try:
            result.summary = summarize(cleaned, keyword_terms, n_sentences=n_sentences)
        except Exception as exc:
            result.warning = str(exc)

    if persist:
        result.stored_id = save_run(
            cleaned,
            method,
            result.keywords,
            result.summary,
            source=source,
            note=result.warning,
        )
    return result

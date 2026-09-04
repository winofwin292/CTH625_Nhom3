"""Tiền xử lý tiếng Việt: tách từ, từ dừng, lọc từ loại.

Tách từ và POS dùng underthesea vì đề bài yêu cầu lọc Danh từ, Động từ, Tính từ
(phương pháp TF-IDF). pyvi không có gán nhãn từ loại.

Nhãn POS: wiki underthesea (UTS) — N/Nc/Np/Nu/Ny danh từ, V/Vy động từ, A tính từ.
Cách lọc: giữ nhãn bắt đầu bằng N, V hoặc A.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from underthesea import pos_tag, sent_tokenize, word_tokenize

from src.config import POS_KEEP_PREFIXES, STOPWORDS_PATH

_PUNCT_OR_DIGIT = re.compile(r"^[\W\d_]+$", re.UNICODE)


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def tokenize_words(text: str) -> list[str]:
    return word_tokenize(normalize_text(text))


def tokenize_sentences(text: str) -> list[str]:
    sentences = sent_tokenize(normalize_text(text))
    return [s.strip() for s in sentences if s.strip()]


def tag_pos(text: str) -> list[tuple[str, str]]:
    sentences = tokenize_sentences(text)
    if not sentences:
        return pos_tag(normalize_text(text))
    pairs: list[tuple[str, str]] = []
    for sentence in sentences:
        pairs.extend(pos_tag(sentence))
    return pairs


def to_phobert_token(token: str) -> str:
    """PhoBERT / vietnamese-bi-encoder nhận từ đa tiếng nối bằng gạch dưới."""
    return token.strip().replace(" ", "_")


def to_phobert_text(tokens: list[str]) -> str:
    return " ".join(to_phobert_token(t) for t in tokens if t.strip())


def _stopword_variants(word: str) -> set[str]:
    folded = normalize_text(word).lower()
    variants = {folded, folded.replace(" ", "_"), folded.replace("_", " ")}
    return {item for item in variants if item}


@lru_cache(maxsize=1)
def load_stopwords() -> set[str]:
    if not STOPWORDS_PATH.exists():
        raise FileNotFoundError(f"Thiếu danh sách từ dừng: {STOPWORDS_PATH}")
    words: set[str] = set()
    for line in STOPWORDS_PATH.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        words.update(_stopword_variants(raw))
    return words


def is_content_pos(pos: str) -> bool:
    return bool(pos) and pos[0] in POS_KEEP_PREFIXES


def is_noise_token(token: str) -> bool:
    stripped = token.strip()
    if not stripped:
        return True
    return bool(_PUNCT_OR_DIGIT.match(stripped.replace("_", "")))


def keep_token(token: str, stopwords: set[str] | None = None) -> bool:
    words = stopwords if stopwords is not None else load_stopwords()
    folded = normalize_text(token).lower()
    if is_noise_token(folded):
        return False
    return folded not in words and to_phobert_token(folded) not in words


def content_tokens(text: str, use_pos: bool = True) -> list[tuple[str, str]]:
    """Trả về (token, nhãn POS). Bỏ từ dừng và token rác. `use_pos=True` thì chỉ giữ N/V/A."""
    stopwords = load_stopwords()
    kept: list[tuple[str, str]] = []
    for token, pos in tag_pos(text):
        if not keep_token(token, stopwords):
            continue
        if use_pos and not is_content_pos(pos):
            continue
        kept.append((token, pos))
    return kept


def tokens_as_document(pairs: list[tuple[str, str]]) -> str:
    return " ".join(to_phobert_token(token) for token, _ in pairs)

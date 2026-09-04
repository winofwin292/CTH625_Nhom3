"""Cấu hình đường dẫn và biến môi trường. Web và notebook đều đọc từ đây."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

STOPWORDS_PATH = PROJECT_ROOT / "data" / "stopwords" / "vietnamese-stopwords.txt"
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
MODELS_DIR = PROJECT_ROOT / "models"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"

EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
METHOD_TFIDF = "tfidf"
METHOD_KEYBERT = "keybert"
KEYWORD_METHODS = (METHOD_TFIDF, METHOD_KEYBERT)

POS_KEEP_PREFIXES = ("N", "V", "A")


def _from_streamlit_secrets(name: str) -> str | None:
    if "streamlit" not in sys.modules:
        return None
    try:
        import streamlit as st

        value = st.secrets.get(name)
    except Exception:
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def env(name: str, default: str = "") -> str:
    secret = _from_streamlit_secrets(name)
    if secret is not None:
        return secret
    return os.getenv(name, default).strip()


def chroma_path() -> Path:
    raw = env("CHROMA_PATH", str(PROJECT_ROOT / "data" / "chroma"))
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def llm_model() -> str:
    return env("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def hf_token() -> str:
    return env("HF_TOKEN") or env("HUGGINGFACEHUB_API_TOKEN")


def hf_provider() -> str:
    return env("HF_PROVIDER") or "featherless-ai"

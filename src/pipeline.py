"""Điểm gọi của web: học IDF, trích từ khóa, tóm tắt và lưu lịch sử.

Thứ tự: đọc chữ (`text`) → từ khóa (`keywords`) → tóm tắt → ChromaDB.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from src.config import (
    CORPUS_DIR,
    KEYWORD_METHODS,
    METHOD_KEYBERT,
    METHOD_TFIDF,
    TFIDF_VECTORIZER_PATH,
    chroma_path,
    hf_provider,
    hf_token,
    llm_model,
)
from src.keywords import extract_keybert, extract_tfidf, fit_tfidf, load_tfidf
from src.text import content_tokens, list_text_files, read_path, read_upload, tokenize_words

SYSTEM_PROMPT = (
    "Bạn là trợ lý tóm tắt văn bản tiếng Việt. "
    "Chỉ dùng thông tin có trong văn bản. Không bịa thêm sự kiện, số liệu, "
    "tên người, tên tổ chức hay địa danh nếu chúng không xuất hiện trong văn bản."
)


def build_user_prompt(text: str, keywords: list[str], n_sentences: int) -> str:
    joined = ", ".join(keywords) if keywords else "(không có từ khóa)"
    return (
        f"Từ khóa cốt lõi (neo ngữ nghĩa): {joined}\n\n"
        f"Văn bản:\n{text}\n\n"
        "Yêu cầu:\n"
        f"- Viết đúng khoảng {n_sentences} câu tiếng Việt, mạch lạc, thành một đoạn.\n"
        "- Mỗi câu phải gắn với ít nhất một từ khóa trong danh sách neo "
        "(dùng đúng ý của từ khóa, không bắt buộc lặp nguyên chữ).\n"
        "- Không thêm số liệu, tên riêng hay sự kiện không có trong văn bản.\n"
        "- Không liệt kê lại từ khóa. Không dùng gạch đầu dòng."
    )


def _chat_huggingface(messages: list[dict], max_tokens: int) -> str:
    from huggingface_hub import InferenceClient

    token = hf_token()
    if not token:
        raise RuntimeError(
            "Chưa có HF_TOKEN. Thêm token vào .env hoặc Streamlit Secrets để gọi LLM."
        )
    client_kwargs = {"token": token}
    provider = hf_provider()
    if provider:
        client_kwargs["provider"] = provider
    client = InferenceClient(**client_kwargs)
    try:
        response = client.chat_completion(
            messages=messages,
            model=llm_model(),
            max_tokens=max_tokens,
            temperature=0.3,
        )
    except Exception as exc:
        text = str(exc)
        if "model_not_supported" in text or "not supported by any provider" in text:
            raise RuntimeError(
                "Hugging Face không gọi được Qwen2.5-7B-Instruct vì chưa bật provider. "
                "Vào https://huggingface.co/settings/inference-providers, bật Featherless AI "
                "(Routed by HF, không cần tài khoản Featherless). "
                "Trong .env đặt HF_PROVIDER=featherless-ai rồi chạy lại Streamlit."
            ) from exc
        raise
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Hugging Face API trả về nội dung rỗng.")
    return content.strip()


def summarize(text: str, keywords: list[str], n_sentences: int = 5) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(text, keywords, n_sentences)},
    ]
    max_tokens = min(80 * max(n_sentences, 1), 800)
    return _chat_huggingface(messages, max_tokens)

COLLECTION_NAME = "keyword_summaries_local"
VN_TZ = timezone(timedelta(hours=7))
EMBED_DIM = 32
_client = None
_collection = None


class _HashEmbeddingFunction:
    """Embedding nội bộ: không tải MiniLM/ONNX. Đủ để Chroma lưu bản ghi lịch sử."""

    @staticmethod
    def name() -> str:
        return "local_hash_32"

    def is_legacy(self) -> bool:
        return False

    def get_config(self) -> dict:
        return {"dim": EMBED_DIM}

    @staticmethod
    def build_from_config(config: dict) -> _HashEmbeddingFunction:
        return _HashEmbeddingFunction()

    def __call__(self, input: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in input:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([(byte / 255.0) - 0.5 for byte in digest[:EMBED_DIM]])
        return vectors


def _patch_sqlite() -> None:
    import sqlite3

    if sqlite3.sqlite_version_info >= (3, 35, 0):
        return
    try:
        import pysqlite3 as sqlite3_new
    except ImportError:
        return
    sys.modules["sqlite3"] = sqlite3_new


def _clear_chroma_cache() -> None:
    try:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:
        pass


def _reset_chroma_files(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _open_persistent(path: Path):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def _open_ephemeral():
    import chromadb
    from chromadb.config import Settings

    return chromadb.EphemeralClient(
        settings=Settings(anonymized_telemetry=False),
    )


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    _patch_sqlite()
    path = chroma_path()
    last_error: Exception | None = None
    for attempt, reset in enumerate((False, True)):
        try:
            _clear_chroma_cache()
            if reset:
                _reset_chroma_files(path)
            _client = _open_persistent(path)
            _collection = _client.get_or_create_collection(
                COLLECTION_NAME,
                embedding_function=_HashEmbeddingFunction(),
            )
            return _collection
        except Exception as exc:
            last_error = exc
            _client = None
            _collection = None

    try:
        _clear_chroma_cache()
        _client = _open_ephemeral()
        _collection = _client.get_or_create_collection(
            COLLECTION_NAME,
            embedding_function=_HashEmbeddingFunction(),
        )
        return _collection
    except Exception:
        if last_error is not None:
            raise last_error
        raise


def format_vn_time(iso_text: str) -> str:
    if not iso_text:
        return "—"
    try:
        moment = datetime.fromisoformat(iso_text)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(VN_TZ).strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return iso_text


def save_run(
    text: str,
    method: str,
    keywords: list[tuple[str, float]],
    summary: str,
    source: str = "input",
    note: str = "",
) -> str:
    run_id = str(uuid4())
    keyword_text = "; ".join(f"{term}:{score:.4f}" for term, score in keywords)
    metadata = {
        "method": method,
        "keywords": keyword_text[:4000],
        "summary": (summary or " ")[:4000],
        "source": source[:200],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if note:
        metadata["note"] = note[:400]
    _get_collection().add(
        ids=[run_id],
        documents=[text],
        metadatas=[metadata],
    )
    return run_id


def list_recent(limit: int = 8) -> list[dict]:
    collection = _get_collection()
    if collection.count() == 0:
        return []
    data = collection.get(include=["documents", "metadatas"])
    rows = []
    for index, run_id in enumerate(data.get("ids", [])):
        meta = data["metadatas"][index] if data.get("metadatas") else {}
        document = data["documents"][index] if data.get("documents") else ""
        rows.append(
            {
                "id": run_id,
                "method": meta.get("method", ""),
                "keywords": meta.get("keywords", ""),
                "summary": meta.get("summary", "") or "",
                "note": meta.get("note", "") or "",
                "source": meta.get("source", ""),
                "created_at": meta.get("created_at", ""),
                "created_at_label": format_vn_time(meta.get("created_at", "")),
                "preview": document[:180],
            }
        )
    rows.sort(key=lambda row: row.get("created_at", ""), reverse=True)
    return rows[:limit]

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


@dataclass
class FitResult:
    n_docs: int
    n_terms: int
    saved: bool
    path: Path
    message: str


def _only_samples(corpus_dir: Path, files: list[Path]) -> bool:
    samples_dir = (corpus_dir / "samples").resolve()
    if corpus_dir.name == "samples":
        return True
    return samples_dir.is_dir() and all(path.resolve().is_relative_to(samples_dir) for path in files)


def fit_corpus(corpus_dir: str | Path | None = None, save_path: Path | None = None) -> FitResult:
    """Học IDF trên các file .txt. Chỉ có mẫu trong samples/ thì giữ joblib đã có."""

    root = Path(corpus_dir).expanduser() if corpus_dir else CORPUS_DIR
    root = root.resolve()
    target = save_path or TFIDF_VECTORIZER_PATH
    if not root.is_dir():
        raise FileNotFoundError(f"Không thấy thư mục dữ liệu: {root}")
    files = list_text_files(root)
    if not files:
        raise FileNotFoundError(f"Không có file .txt trong {root}")

    if _only_samples(root, files):
        model = load_tfidf(target)
        if model is None:
            raise FileNotFoundError(
                "Chưa có file joblib. Thêm văn bản .txt vào thư mục dữ liệu rồi chạy lại."
            )
        return FitResult(
            n_docs=len(files),
            n_terms=len(model.get_feature_names_out()),
            saved=False,
            path=target,
            message=f"Corpus chỉ có {len(files)} file mẫu. Giữ joblib đã fit, không ghi đè.",
        )

    model = fit_tfidf([read_path(path) for path in files], save_path=target)
    return FitResult(
        n_docs=len(files),
        n_terms=len(model.get_feature_names_out()),
        saved=True,
        path=target,
        message=f"Đã lưu {target}",
    )


def load_input(
    text: str = "",
    file_path: str = "",
    file_name: str = "",
    file_bytes: bytes | None = None,
) -> tuple[str, str]:
    """Đọc chữ dán hoặc file .txt/.pdf. Trả về văn bản và nhãn nguồn."""

    if file_bytes is not None:
        name = file_name.strip()
        if not name:
            raise ValueError("Thiếu tên file.")
        loaded = read_upload(name, file_bytes).strip()
        if not loaded:
            raise ValueError("File không có nội dung chữ.")
        return loaded, name

    path = file_path.strip()
    if path:
        loaded = read_path(path).strip()
        if not loaded:
            raise ValueError("File không có nội dung chữ.")
        return loaded, Path(path).name

    pasted = text.strip()
    if pasted:
        return pasted, "văn bản dán"
    raise ValueError("Điền đường dẫn file hoặc dán văn bản.")


def extract_keywords(text: str, method: str, top_n: int) -> list[tuple[str, float]]:
    if method == METHOD_TFIDF:
        return extract_tfidf(text, top_n=top_n)
    if method == METHOD_KEYBERT:

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


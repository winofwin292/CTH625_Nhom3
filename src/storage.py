"""Lưu văn bản, từ khóa và tóm tắt vào ChromaDB."""

from __future__ import annotations

import hashlib
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from src.config import chroma_path

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

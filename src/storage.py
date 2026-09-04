"""Lưu văn bản, từ khóa và tóm tắt vào ChromaDB."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
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


def _get_collection():
    global _client, _collection
    if _collection is None:
        import chromadb

        _client = chromadb.PersistentClient(path=str(chroma_path()))
        _collection = _client.get_or_create_collection(
            COLLECTION_NAME,
            embedding_function=_HashEmbeddingFunction(),
        )
    return _collection


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
        "keywords": keyword_text[:1000],
        "summary": (summary or " ")[:1000],
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

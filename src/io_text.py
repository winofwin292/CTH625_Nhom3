"""Đọc văn bản từ ô nhập, file .txt hoặc .pdf."""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".txt", ".pdf"}


class InputError(ValueError):
    """Lỗi đọc file đầu vào."""


def read_txt_bytes(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise InputError("Không đọc được file .txt. Hãy lưu file ở encoding UTF-8.")


def repair_pdf_extracted_text(text: str) -> str:
    """Nối mảnh chữ PDF (mỗi glyph một dòng) thành từ tiếng Việt liền.

    Dòng trống (cột / đoạn) được giữ làm ranh giới, không dính hai cụm khác nhau.
    """
    pieces: list[str] = []
    buf = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if buf:
                pieces.append(buf)
                buf = ""
            continue
        if not buf:
            buf = line
            continue
        last = buf.split()[-1]
        if len(line) <= 3 or len(last) <= 2:
            buf += line
        else:
            buf += " " + line
    if buf:
        pieces.append(buf)
    return "\n\n".join(pieces).strip()


def _extract_pdf_page(page) -> str:
    try:
        layout = page.extract_text(extraction_mode="layout") or ""
    except Exception:
        layout = ""
    if layout.strip():
        return re.sub(r"[ \t]{2,}", " ", layout)
    raw = page.extract_text() or ""
    return repair_pdf_extracted_text(raw)


def read_pdf_bytes(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    pages = [_extract_pdf_page(page) for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        raise InputError("File PDF không có lớp văn bản (có thể là bản scan).")
    return text


def read_upload(name: str, data: bytes) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".txt":
        return read_txt_bytes(data)
    if suffix == ".pdf":
        return read_pdf_bytes(data)
    raise InputError("Chỉ hỗ trợ .txt và .pdf.")


def read_path(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        raise InputError(f"Không tìm thấy file: {file_path}")
    return read_upload(file_path.name, file_path.read_bytes())


def list_text_files(root: str | Path) -> list[Path]:
    return sorted(path for path in Path(root).rglob("*.txt") if path.is_file())

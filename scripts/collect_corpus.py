"""Thu thập bài báo khoa học tiếng Việt từ tạp chí OA trên VJOL (OAI-PMH)."""

from __future__ import annotations

import json
import logging
import random
import re
import time
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logging.getLogger("pypdf").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "corpus" / "baibao_khoahoc"
MANIFEST_PATH = PROJECT_ROOT / "data" / "corpus" / "manifest.jsonl"
VIETNEWS_DIR = PROJECT_ROOT / "data" / "corpus" / "vietnews"

N_DOCS = 80
MIN_CHARS = 8000
SEED = 42
SLEEP_SEC = 0.35
USER_AGENT = "CTH625-Nhom3-corpus-collector/1.0 (academic coursework)"

JOURNALS = [
    {
        "id": "khcn",
        "name": "Tạp chí Khoa học và Công nghệ Việt Nam - A",
        "oai": "https://vjol.info.vn/khcn/oai",
        "from": "2018-01-01",
        "portal": "https://vjol.info.vn/khcn",
    },
    {
        "id": "jte",
        "name": "Tạp chí Khoa học Giáo dục Kỹ thuật",
        "oai": "https://vjol.info.vn/jte/oai",
        "from": "2018-01-01",
        "portal": "https://vjol.info.vn/jte",
    },
]

VI_MARKERS = (
    "nghiên cứu",
    "phương pháp",
    "kết quả",
    "đặt vấn đề",
    "tóm tắt",
    "kết luận",
    "từ khóa",
)

_VIEW_RE = re.compile(r"/article/view/(\d+)/(\d+)/?")


def http_get(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def harvest_journal(journal: dict) -> list[dict]:
    rows: list[dict] = []
    params = {"verb": "ListRecords", "metadataPrefix": "oai_dc", "from": journal["from"]}
    url = journal["oai"] + "?" + urllib.parse.urlencode(params)
    pages = 0
    while url and pages < 8:
        pages += 1
        time.sleep(SLEEP_SEC)
        xml = http_get(url).decode("utf-8", "replace")
        xml = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", xml)
        for block in re.findall(r"<record>(.*?)</record>", xml, flags=re.S):
            if 'status="deleted"' in block:
                continue
            language = ""
            lang_match = re.search(r"<dc:language[^>]*>(.*?)</dc:language>", block)
            if lang_match:
                language = lang_match.group(1).strip().lower()
            if language and not language.startswith("vi"):
                continue
            title = ""
            for title_match in re.finditer(r"<dc:title[^>]*>(.*?)</dc:title>", block):
                candidate = " ".join(title_match.group(1).split())
                if candidate:
                    title = candidate
                    break
            date_match = re.search(r"<dc:date>(.*?)</dc:date>", block)
            date = date_match.group(1).strip() if date_match else ""
            view_url = ""
            file_id = ""
            for rel in re.findall(r"<dc:relation>(.*?)</dc:relation>", block):
                match = _VIEW_RE.search(rel)
                if match:
                    view_url = rel.strip()
                    file_id = match.group(2)
                    break
            if not view_url:
                continue
            rows.append(
                {
                    "journal_id": journal["id"],
                    "journal": journal["name"],
                    "portal": journal["portal"],
                    "title": title,
                    "date": date,
                    "view_url": view_url,
                    "download_url": view_url.replace("/article/view/", "/article/download/"),
                    "file_id": file_id,
                }
            )
        token_match = re.search(r"<resumptionToken[^>]*>(.*?)</resumptionToken>", xml)
        token = token_match.group(1).strip() if token_match else ""
        url = (
            f"{journal['oai']}?verb=ListRecords&resumptionToken={urllib.parse.quote(token)}"
            if token
            else ""
        )
    return rows


def extract_pdf_text(data: bytes) -> tuple[str, int]:
    reader = PdfReader(BytesIO(data))
    pages = [(page.extract_text() or "") for page in reader.pages]
    text = "\n".join(pages)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, len(reader.pages)


def vietnamese_letter_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 400:
        return 0.0
    viet = sum(1 for char in letters if "\u0100" <= char <= "\u1ef9")
    return viet / len(letters)


def is_vietnamese_paper(text: str) -> bool:
    lowered = text.lower()
    marker_hits = sum(1 for marker in VI_MARKERS if marker in lowered)
    return marker_hits >= 2 and vietnamese_letter_ratio(text) >= 0.12


def write_corpus(selected: list[dict]) -> None:
    if OUT_DIR.exists():
        for old in OUT_DIR.glob("*.txt"):
            old.unlink()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as manifest:
        for index, item in enumerate(selected, start=1):
            name = f"{index:03d}_{item['journal_id']}_{item['file_id']}.txt"
            body = f"{item['title']}\n\n{item['text']}\n"
            (OUT_DIR / name).write_text(body, encoding="utf-8")
            record = {
                "file": f"baibao_khoahoc/{name}",
                "title": item["title"],
                "date": item["date"],
                "journal": item["journal"],
                "view_url": item["view_url"],
                "download_url": item["download_url"],
                "n_chars": item["n_chars"],
                "n_pages": item["n_pages"],
                "license_note": "PDF toàn văn trên VJOL (tạp chí OA). Dùng cho đồ án học phần.",
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")


def remove_vietnews_subset() -> None:
    if not VIETNEWS_DIR.exists():
        return
    for path in VIETNEWS_DIR.glob("*.txt"):
        path.unlink()


def main() -> None:
    candidates: list[dict] = []
    for journal in JOURNALS:
        harvested = harvest_journal(journal)
        print(f"{journal['id']}: {len(harvested)} OAI records (vi)")
        candidates.extend(harvested)
    random.Random(SEED).shuffle(candidates)

    selected: list[dict] = []
    for item in candidates:
        if len(selected) >= N_DOCS:
            break
        time.sleep(SLEEP_SEC)
        try:
            data = http_get(item["download_url"])
        except Exception:
            continue
        if not data.startswith(b"%PDF"):
            continue
        try:
            text, n_pages = extract_pdf_text(data)
        except (PdfReadError, Exception):
            continue
        if len(text) < MIN_CHARS or not is_vietnamese_paper(text):
            continue
        item["text"] = text
        item["n_chars"] = len(text)
        item["n_pages"] = n_pages
        selected.append(item)
        print(f"keep {len(selected)}/{N_DOCS} chars={len(text)} pages={n_pages}")

    if len(selected) < 20:
        raise RuntimeError(f"Chi thu duoc {len(selected)} bai dat nguong, can it nhat 20.")
    write_corpus(selected)
    remove_vietnews_subset()
    print(f"Wrote {len(selected)} files to {OUT_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

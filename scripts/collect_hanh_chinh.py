"""Lấy mẫu văn bản hành chính / QPPL tiếng Việt từ dataset công khai trên Hugging Face.

Nguồn: th1nhng0/vietnamese-legal-documents (CC BY 4.0), gốc vbpl.vn.
Không cào cổng vbpl trực tiếp; chỉ lấy bản đã công bố trên Hugging Face.
"""

from __future__ import annotations

import html
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "corpus" / "hanh_chinh"
MANIFEST_PATH = OUT_DIR / "manifest.jsonl"

N_DOCS = 40
MIN_CHARS = 2000
MAX_CHARS = 22000
SEED = 43
SLEEP_SEC = 8.0
USER_AGENT = "CTH625-Nhom3-corpus-collector/1.0 (academic coursework)"
HF_ROWS = "https://datasets-server.huggingface.co/rows"
CONTENT_ROWS = 170824

WANTED_TYPES = (
    "QUYẾT ĐỊNH",
    "THÔNG TƯ",
    "NGHỊ QUYẾT",
    "CHỈ THỊ",
    "CÔNG VĂN",
    "THÔNG BÁO",
    "NGHỊ ĐỊNH",
)
MAX_PER_TYPE = 10
SKIP_PREFIXES = ("LUẬT", "BỘ LUẬT", "HIẾN PHÁP")


def http_get_json(url: str, timeout: int = 90) -> dict:
    last_error: Exception | None = None
    for attempt in range(6):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429:
                raise
            time.sleep(15 * (attempt + 1))
    raise last_error or RuntimeError("Khong tai duoc JSON tu Hugging Face.")


def html_to_text(raw_html: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw_html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p>", "\n", text)
    text = re.sub(r"(?is)</div>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_type(text: str) -> str | None:
    head = text[:800].upper()
    for prefix in SKIP_PREFIXES:
        if re.search(rf"\b{re.escape(prefix)}\b", head):
            return None
    for label in WANTED_TYPES:
        if label in head:
            return label
    return None


def vietnamese_letter_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    viet = sum(1 for char in letters if ord(char) > 127)
    return viet / len(letters)


def fetch_content_page(offset: int, length: int = 20) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "dataset": "th1nhng0/vietnamese-legal-documents",
            "config": "content",
            "split": "data",
            "offset": str(offset),
            "length": str(length),
        }
    )
    payload = http_get_json(f"{HF_ROWS}?{query}")
    return [item["row"] for item in payload.get("rows") or []]


def load_existing() -> tuple[set[str], list[dict], int]:
    known: set[str] = set()
    records: list[dict] = []
    if MANIFEST_PATH.exists():
        for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(record)
            source_id = str(record.get("source_id") or "")
            if source_id:
                known.add(source_id)
    next_index = 1
    for path in OUT_DIR.glob("*.txt"):
        match = re.match(r"^(\d+)_", path.name)
        if match:
            next_index = max(next_index, int(match.group(1)) + 1)
    return known, records, next_index


def append_corpus(selected: list[dict], start_index: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("a", encoding="utf-8") as manifest:
        for offset, item in enumerate(selected):
            index = start_index + offset
            name = f"{index:03d}_{item['doc_type_slug']}_{item['id']}.txt"
            body = f"{item['doc_type']}\n\n{item['text']}\n"
            (OUT_DIR / name).write_text(body, encoding="utf-8")
            record = {
                "file": f"hanh_chinh/{name}",
                "source_id": item["id"],
                "doc_type": item["doc_type"],
                "n_chars": item["n_chars"],
                "dataset": "th1nhng0/vietnamese-legal-documents",
                "portal": "https://vbpl.vn",
                "license_note": (
                    "Văn bản QPPL thuộc phạm vi công khai. "
                    "Bản tập hợp trên Hugging Face: CC BY 4.0."
                ),
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")


TYPE_SLUG = {
    "QUYẾT ĐỊNH": "quyet_dinh",
    "THÔNG TƯ": "thong_tu",
    "NGHỊ QUYẾT": "nghi_quyet",
    "CHỈ THỊ": "chi_thi",
    "CÔNG VĂN": "cong_van",
    "THÔNG BÁO": "thong_bao",
    "NGHỊ ĐỊNH": "nghi_dinh",
}


def main() -> None:
    rng = random.Random(SEED)
    known_ids, existing, start_index = load_existing()
    remaining = max(N_DOCS - len(existing), 0)
    if remaining == 0:
        print(f"Da co {len(existing)} van ban (>= {N_DOCS}). Khong tai them.")
        return
    page_starts = list(range(0, CONTENT_ROWS - 20, 20))
    rng.shuffle(page_starts)
    selected: list[dict] = []
    counts: Counter[str] = Counter()
    for record in existing:
        doc_type = record.get("doc_type")
        if doc_type:
            counts[doc_type] += 1

    for offset in page_starts[:28]:
        if len(selected) >= remaining:
            break
        time.sleep(SLEEP_SEC)
        try:
            rows = fetch_content_page(offset)
        except Exception as exc:
            print(f"skip page offset={offset}: {exc}")
            continue
        print(f"page offset={offset} rows={len(rows)} new={len(selected)}")
        rng.shuffle(rows)
        for row in rows:
            if len(selected) >= remaining:
                break
            doc_id = str(row.get("id") or "")
            if not doc_id or doc_id in known_ids:
                continue
            if not row.get("content_html"):
                continue
            text = html_to_text(row["content_html"])
            if not (MIN_CHARS <= len(text) <= MAX_CHARS):
                continue
            if vietnamese_letter_ratio(text) < 0.08:
                continue
            doc_type = detect_type(text)
            if doc_type is None or counts[doc_type] >= MAX_PER_TYPE:
                continue
            slug = TYPE_SLUG[doc_type]
            selected.append(
                {
                    "id": doc_id,
                    "text": text,
                    "n_chars": len(text),
                    "doc_type": doc_type,
                    "doc_type_slug": slug,
                }
            )
            known_ids.add(doc_id)
            counts[doc_type] += 1
            print(f"keep {len(existing)+len(selected)}/{N_DOCS} {slug} id={doc_id} chars={len(text)}")

    if not selected:
        raise RuntimeError("Khong them duoc van ban moi.")
    append_corpus(selected, start_index)
    print(f"Added {len(selected)} files to {OUT_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")
    print("Types:", {TYPE_SLUG.get(key, key): value for key, value in counts.items()})


if __name__ == "__main__":
    main()

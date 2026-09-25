"""Tách chữ PDF tiểu luận/luận văn thành .txt để fit TF-IDF.

PDF nằm trong data/corpus/tieu_luan/ (không commit Git). Mỗi PDF ra một .txt cùng tên.
Bỏ file dưới 8.000 ký tự hoặc tỷ lệ chữ có dấu quá thấp — thường là bản scan.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io_text import InputError, read_path

OUT_DIR = PROJECT_ROOT / "data" / "corpus" / "tieu_luan"
MIN_CHARS = 8000
MIN_VI_RATIO = 0.08


def vietnamese_letter_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 200:
        return 0.0
    viet = sum(1 for char in letters if "\u0100" <= char <= "\u1ef9" or char in "đĐ")
    return viet / len(letters)


def main() -> None:
    warnings.filterwarnings("ignore", module="pypdf")
    pdfs = sorted(OUT_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"Không có PDF trong {OUT_DIR}")
    kept = 0
    for path in pdfs:
        try:
            text = read_path(path).strip()
        except InputError as exc:
            print(f"BỎ  {path.name}: {exc}")
            continue
        ratio = vietnamese_letter_ratio(text)
        if len(text) < MIN_CHARS or ratio < MIN_VI_RATIO:
            print(f"BỎ  {path.name}: {len(text)} ký tự, tỷ lệ dấu {ratio:.3f}")
            continue
        target = path.with_suffix(".txt")
        target.write_text(text + "\n", encoding="utf-8")
        kept += 1
        print(f"OK   {target.name}: {len(text)} ký tự, tỷ lệ dấu {ratio:.3f}")
    print(f"Giữ {kept}/{len(pdfs)} file .txt")


if __name__ == "__main__":
    main()

# Corpus

Đề bài 7 nêu ba **kiểu đầu vào web**: bài báo, tiểu luận, văn bản hành chính. Đó không phải yêu cầu corpus train phải cân ba thể loại. TF-IDF vẫn nên thấy được từ vùng hành chính / tiểu luận, nếu không từ vựng IDF sẽ lệch về bài báo khoa học.

## Git

Full-text train **không** đưa lên Git (bản quyền). Nhóm gửi zip riêng, giải nén đúng cây thư mục repo:

- `data/corpus/baibao_khoahoc/*.txt`
- `data/corpus/hanh_chinh/*.txt` (+ `manifest.jsonl` nếu có)
- `data/corpus/tieu_luan/*.txt` (nếu có)
- `data/corpus/manifest.jsonl`

`models/tfidf_vectorizer.joblib` **commit Git** (không chứa full-text, để Cloud nạp TF-IDF đã fit). Fit lại trên máy rồi commit file này khi corpus đổi.

Trên Git còn README, `samples/` (văn bản nhóm soạn), stopwords, và `.gitkeep`.

Nén corpus (từ gốc repo; dùng `tar` để giữ `data/…` trong zip — `Compress-Archive` làm dẹt thư mục):

```powershell
tar -a -cf corpus-train.zip data/corpus/baibao_khoahoc data/corpus/hanh_chinh data/corpus/manifest.jsonl
```

Giải nén vào gốc repo:

```powershell
tar -xf corpus-train.zip
```

## Bài báo khoa học

Full-text PDF từ tạp chí open access trên [VJOL](https://vjol.info.vn) (Cục Thông tin, Thống kê — Bộ KH&CN). Giao thức: **OAI-PMH** + URL `/article/download/`. Không cào HTML trang báo điện tử. Vietnews đã bỏ (quá ngắn, thể loại báo chí).

## Nguồn tạp chí

| Tạp chí | OAI |
| --- | --- |
| Tạp chí Khoa học và Công nghệ Việt Nam - A | `https://vjol.info.vn/khcn/oai` |
| Tạp chí Khoa học Giáo dục Kỹ thuật | `https://vjol.info.vn/jte/oai` |

Script: `python scripts/collect_corpus.py`

- 80 bài, `random.seed(42)`
- Giữ bài ≥ 8.000 ký tự, tiếng Việt (từ khóa học thuật + tỷ lệ dấu thanh)
- File: `data/corpus/baibao_khoahoc/`
- Metadata (URL xem/tải, số trang): `data/corpus/manifest.jsonl`

## Văn bản hành chính

`data/corpus/hanh_chinh/` — 40 văn bản (mỗi loại 10: Quyết định, Thông tư, Chỉ thị, Nghị quyết) lấy từ [th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents) (CC BY 4.0; gốc [vbpl.vn](https://vbpl.vn)). Đây là văn bản QPPL công khai, gần thể loại hành chính hơn bài báo, nhưng **không** phải thông báo nội bộ trường.

```bash
python scripts/collect_hanh_chinh.py
```

## Tiểu luận

`data/corpus/tieu_luan/` — **chưa có file train**. Đã quét DSpace ĐHQGHN qua OAI-PMH: metadata mở, PDF toàn văn **đòi đăng nhập**. Không tải được từ máy này. Hướng dẫn sau khi bạn login: `data/corpus/tieu_luan/README.md`.

## File minh họa (demo web)

`data/corpus/samples/` — 3 văn bản nhóm soạn (bài ngắn / tiểu luận / hành chính). Dùng thử kiểu đầu vào, không thay corpus train.

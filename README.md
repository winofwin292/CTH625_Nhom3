# CTH625 — Đề tài 7: Tóm tắt văn bản dựa trên trích xuất từ khóa

Nhóm 3, học phần CTH625 Xử lý ngôn ngữ tự nhiên. Bốn thành viên (tên sẽ bổ sung khi nhóm chốt).

Ứng dụng web nhận **một văn bản tiếng Việt** (nhập trực tiếp, `.txt` hoặc `.pdf`), trả về **danh sách từ khóa** và **đoạn tóm tắt**.

## Tài liệu

| File | Đọc khi |
| --- | --- |
| [docs/04-huong-dan-web-va-deploy.md](docs/04-huong-dan-web-va-deploy.md) | **Cấu hình, chạy lần đầu, hạn mức Hugging Face, cập nhật corpus, train lại, deploy** |
| [docs/03-huong-dan-train.md](docs/03-huong-dan-train.md) | Chi tiết notebook local / Google Colab |
| [docs/00-yeu-cau-de-tai.md](docs/00-yeu-cau-de-tai.md) | Yêu cầu nộp bài (rút từ `docs/DoAnMonHoc_2026.docx`) |
| [docs/01-ly-thuyet.md](docs/01-ly-thuyet.md) | TF-IDF, KeyBERT, PhoBERT, prompt LLM |
| [docs/02-kien-truc-he-thong.md](docs/02-kien-truc-he-thong.md) | Luồng xử lý, thư mục, lựa chọn kỹ thuật |

Tóm tắt LLM chỉ qua Hugging Face. Mỗi người clone repo tự tạo token (không commit `.env`).

Corpus train (full-text) **không** nằm trên Git — giải nén zip riêng (xem `data/corpus/README.md`). File `models/tfidf_vectorizer.joblib` **có** trên Git để Streamlit Cloud dùng TF-IDF đã fit.

## Chạy nhanh

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run streamlit_app.py
```

Trên Linux/macOS: `source .venv/bin/activate` và `cp .env.example .env`. Điền `HF_TOKEN` theo `docs/04`.

## Việc chưa làm ở bước này

- Báo cáo Word (nộp sau).
- Tên thành viên, GitHub để deploy Streamlit, hạn nộp / GVHD.

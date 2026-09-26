# Tóm tắt văn bản tiếng Việt theo từ khóa

Web nhận một văn bản tiếng Việt (nhập tay, `.txt` hoặc `.pdf`) và trả về danh sách từ khóa cùng đoạn tóm tắt.

| Việc | Tài liệu |
| --- | --- |
| Chạy web, token, deploy | [docs/04-huong-dan-web-va-deploy.md](docs/04-huong-dan-web-va-deploy.md) |
| Fit TF-IDF | [docs/03-huong-dan-train.md](docs/03-huong-dan-train.md) |
| Thư mục và luồng xử lý | [docs/02-kien-truc-he-thong.md](docs/02-kien-truc-he-thong.md) |
| Corpus | [data/corpus/README.md](data/corpus/README.md) |

## Chạy

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run streamlit_app.py
```

Linux/macOS: `source .venv/bin/activate` và `cp .env.example .env`. Điền `HF_TOKEN` theo `docs/04`. Mở http://localhost:8501.

Toàn văn dùng để học IDF không có trên Git. File `models/tfidf_vectorizer.joblib` có trên Git để web dùng TF-IDF đã fit.

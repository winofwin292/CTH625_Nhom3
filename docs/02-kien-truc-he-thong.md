# Kiến trúc hệ thống

## Lựa chọn kỹ thuật (đều nằm trong danh sách đề bài)

| Hạng mục | Chọn | Lý do nằm trong đề bài |
| --- | --- | --- |
| Web | Streamlit | Đề bài nêu Streamlit; deploy demo lên [share.streamlit.io](https://share.streamlit.io) |
| Tách từ + POS | underthesea | Cần POS để lọc N/V/A; pyvi không có POS |
| Từ dừng | vietnamese-stopwords (GitHub) + bước chuẩn hóa | Đề bài yêu cầu stop words cải tiến, không yêu cầu tự soạn từ |
| Baseline | TF-IDF + POS | Đúng phương pháp 1 |
| Deep learning | KeyBERT + `vietnamese-bi-encoder` | Đúng phương pháp 2 |
| LLM | Qwen2.5-7B-Instruct | Đúng danh sách đề bài; gọi Hugging Face Inference Providers, không load 7B local |
| CSDL | ChromaDB | Đúng nhóm vector database được nêu |

Không dùng RAG trong pipeline này. RAG xuất hiện ở mục lý thuyết chung của file Word, không nằm trong mô tả kỹ thuật đề tài 7.

## Luồng xử lý

```text
.txt / .pdf / ô nhập
        │
        ▼
   tiền xử lý (tách câu, tách từ, stop words, POS N/V/A)
        │
        ├─ phương pháp 1: TF-IDF
        └─ phương pháp 2: KeyBERT + vietnamese-bi-encoder
        │
        ▼
   từ khóa = neo ngữ nghĩa
        │
        ▼
   prompt → LLM (Hugging Face Inference Providers)
        │
        ▼
   giao diện Streamlit + lưu ChromaDB
```

Web và notebook **không** tự cài thuật toán song song. Cả hai gọi `src.pipeline.run`.

## Thư mục

```text
streamlit_app.py          # entry Streamlit Cloud (để ở thư mục gốc)
src/
  config.py               # đường dẫn, HF_TOKEN, HF_PROVIDER
  io_text.py              # đọc txt/pdf
  preprocess.py           # tách từ, stop words, POS
  keywords_tfidf.py       # fit + extract TF-IDF
  keywords_keybert.py     # extract KeyBERT (lazy load model)
  summarizer.py           # prompt + gọi LLM
  storage.py              # ChromaDB
  pipeline.py             # run()
notebooks/train_keyword_extraction.ipynb
data/stopwords/
data/corpus/              # baibao_khoahoc + hanh_chinh + tieu_luan (nếu có) + samples; fit TF-IDF
models/                   # tfidf_vectorizer.joblib sau khi train
```

## LLM

Nhóm chốt **chỉ Hugging Face Inference Providers** (`HF_TOKEN`, `HF_PROVIDER=featherless-ai`, model `Qwen/Qwen2.5-7B-Instruct`). Không nhúng model 7B vào process Streamlit: RAM Community Cloud không đủ, và đó không phải yêu cầu đề bài. Hạn mức credit: `docs/04-huong-dan-web-va-deploy.md`.

ChromaDB lưu lịch sử bằng embedding hash nội bộ (`src/storage.py`), không tải `all-MiniLM-L6-v2`. Bật «Lưu vào ChromaDB» không kéo model ONNX.

## KeyBERT trên Cloud

Lần đầu chọn phương pháp 2, app tải `vietnamese-bi-encoder` (~0.1B tham số). TF-IDF không tải encoder. Nếu Cloud hết RAM, dùng TF-IDF để demo và chạy KeyBERT trên Colab/local.

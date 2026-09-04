# Hướng dẫn train — local và Google Colab

**Khi nào train lại** (sau khi thêm corpus, sửa tiền xử lý): xem [docs/04-huong-dan-web-va-deploy.md](04-huong-dan-web-va-deploy.md) mục 4–5. File này chỉ chi tiết notebook.

File train: `notebooks/train_keyword_extraction.ipynb`.

Đề bài yêu cầu **một tập tin mã nguồn huấn luyện/fine-tune** (khuyến khích Colab). Với đề tài 7:

- **Huấn luyện có trọng số thật:** fit `TfidfVectorizer` trên corpus, lưu `models/tfidf_vectorizer.joblib`.
- **Phương pháp 2:** KeyBERT dùng pretrained `vietnamese-bi-encoder`. Đề bài không giao dataset từ khóa vàng, nên notebook **không bịa bước fine-tune**. Nếu nhóm có tập gán nhãn sau này, mới bổ sung fine-tune encoder.

## Chuẩn bị corpus

Đề tài 7 không chỉ định dataset. Corpus fit TF-IDF: **80 bài báo khoa học** VJOL + mẫu hành chính/QPPL từ dataset vbpl trên Hugging Face + 3 file minh họa. Full-text này **không commit Git** (zip riêng). Tiểu luận dài **chưa có** trong train; thêm thủ công vào `data/corpus/tieu_luan/` rồi chạy lại notebook. Chi tiết: `data/corpus/README.md`.

Không dùng tin tức Vietnews (quá ngắn, không phải bài báo khoa học). Không fine-tune encoder vì không có tập từ khóa vàng.

## Cách 1 — Local

```bash
pip install -r requirements-train.txt
```

Mở Jupyter từ **thư mục gốc repo**:

```bash
jupyter notebook notebooks/train_keyword_extraction.ipynb
```

Chạy lần lượt các cell. Artifact ghi vào `models/tfidf_vectorizer.joblib`. Web sẽ tự nạp file này.

## Cách 2 — Google Colab

1. Upload repo lên Google Drive, **hoặc** push Git rồi clone.
2. Mở notebook trên Colab.
3. Cell cấu hình:

```python
COLAB_SOURCE = "drive"   # hoặc "clone"
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/CTH625_Nhom3"
REPO_URL = ""            # điền nếu COLAB_SOURCE = "clone"
```

4. Chạy cell cài gói (chỉ khi `IN_COLAB` là True).
5. Nếu dùng Drive: cell `drive.mount('/content/drive')` rồi `cd` vào `DRIVE_PROJECT_PATH`.
6. Fit TF-IDF, thử KeyBERT, lưu `models/` (trên Drive thì file còn sau khi tắt máy).

GPU Colab hữu ích khi load `vietnamese-bi-encoder`, không bắt buộc cho TF-IDF.

## Kiểm tra sau train

`models/tfidf_vectorizer.joblib` tồn tại. Chạy web, chọn TF-IDF: từ khóa phải lấy IDF từ corpus chứ không chỉ từ các câu trong một file.

## Không làm trong notebook này

- Fine-tune Qwen/Vistral/PhoGPT (đề bài dùng LLM để sinh tóm tắt, không yêu cầu train LLM).

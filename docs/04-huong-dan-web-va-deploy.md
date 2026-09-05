# Cấu hình, chạy lần đầu, cập nhật dữ liệu, train lại, deploy

Một file cho thao tác hàng ngày. Lý thuyết: `docs/01`. Kiến trúc: `docs/02`. Chi tiết cell notebook/Colab: `docs/03`.

Nhóm chốt **LLM chỉ qua Hugging Face Inference Providers**. Không chạy Qwen 7B trên máy (Ollama / load model trong Streamlit).

## 1. Cấu hình Hugging Face (làm một lần sau khi clone)

Mỗi thành viên dùng **token của chính mình**. Không commit `.env`, không dán `HF_TOKEN` lên Git/chat.

1. Tài khoản [huggingface.co](https://huggingface.co), xác nhận email.
2. Token **fine-grained** với quyền **Make calls to Inference Providers**: [tạo token sẵn quyền](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained). Token chỉ *Read* sẽ bị `403`. Sửa token cũ: tick Inference **của user** (không phải org) rồi **Save token**.
3. Bật provider host Qwen 7B: [settings/inference-providers](https://huggingface.co/settings/inference-providers) → **Featherless AI** → *Routed by Hugging Face* (không cần tài khoản Featherless). Bỏ bước này thì API trả `model_not_supported`.
4. Copy `.env.example` thành `.env`, dán token:

```env
HF_TOKEN=hf_...
HF_PROVIDER=featherless-ai
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

Không có token: web vẫn trích từ khóa; ô tóm tắt báo thiếu cấu hình.

## 2. Chạy lần đầu (máy local)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Trên Linux/macOS: `source .venv/bin/activate` và `cp .env.example .env`. Điền `HF_TOKEN` như mục 1.

```bash
streamlit run streamlit_app.py
```

Đổi `.env` khi app đang chạy thì token cũ vẫn còn trong RAM: **tắt hẳn** process (Ctrl+C) rồi chạy lại lệnh Streamlit. File watcher đã tắt (`fileWatcherType = "none"` trong `.streamlit/config.toml`) để Streamlit không quét gói `transformers` (traceback `torchvision` giả). Sửa code Python thì bấm **Rerun** trên web hoặc restart. Không nhân đôi khối `[theme]` trong file cấu hình — TOML lỗi thì Streamlit bỏ cả file và watcher bật lại.

`models/tfidf_vectorizer.joblib` nằm trên Git (Cloud nạp được). Full-text corpus **không** commit — zip riêng. Chưa có file joblib thì TF-IDF tính IDF tạm trên từng văn bản đang nhập.

## 3. Hạn mức Hugging Face (Inference Providers)

Nguồn: [Pricing and Billing — Inference Providers](https://huggingface.co/docs/inference-providers/pricing) (Hugging Face có thể đổi số). App gọi *Routed by Hugging Face* (`HF_PROVIDER=featherless-ai`) nên **credit tháng của tài khoản HF được trừ**, không cần tài khoản Featherless.

| Loại tài khoản | Credit mỗi tháng | Dùng cho | Dùng thêm sau khi hết credit |
| --- | --- | --- | --- |
| Free | **$0.10** (ghi *subject to change*) | Inference Providers | Có, **phải mua credit** trước |
| PRO | $2.00 | Toàn bộ dịch vụ compute HF | Có |
| Team / Enterprise | $2.00 / ghế | Toàn bộ dịch vụ compute HF | Có |

Hết $0.10 của tài khoản free thì request tóm tắt sẽ lỗi thanh toán / hết hạn mức — không còn “gọi miễn phí không giới hạn”. Hugging Face tính đúng giá provider, không cộng markup. Số lượt tóm tắt cụ thể phụ thuộc độ dài văn bản (token vào + token ra); xem mức đã dùng tại [Inference Providers settings](https://huggingface.co/settings/inference-providers) và [billing](https://huggingface.co/settings/billing).

Tải encoder KeyBERT (`vietnamese-bi-encoder`) từ Hub **không** trừ credit Inference Providers.

## 4. Khi cập nhật dữ liệu (corpus)

Làm theo thứ tự: **thu thập → train lại TF-IDF → restart web**. Không cần train lại LLM hay encoder KeyBERT.

| Việc | Lệnh / thao tác |
| --- | --- |
| Thêm / làm mới bài báo VJOL | `python scripts/collect_corpus.py` |
| Thêm / làm mới văn bản QPPL | `python scripts/collect_hanh_chinh.py` (bỏ qua ID đã có) |
| Thêm tiểu luận | Bỏ file `.txt` vào `data/corpus/tieu_luan/` (xem README thư mục đó) |
| Fit lại TF-IDF | Mở `notebooks/train_keyword_extraction.ipynb`, chạy các cell fit, ghi `models/tfidf_vectorizer.joblib` |
| Web nhận vectorizer mới | Restart Streamlit |

ChromaDB **không** tự cập nhật khi corpus đổi. Lịch sử trên web là các lần người dùng đã bấm tóm tắt và bật «Lưu vào ChromaDB». Không xóa `data/chroma/` trừ khi muốn làm trống lịch sử demo.

Chi tiết nguồn corpus: `data/corpus/README.md`. Chi tiết cell notebook / Colab: `docs/03-huong-dan-train.md`.

## 5. Train lại — khi nào và làm gì

| Thay đổi | Có train lại TF-IDF? | Ghi chú |
| --- | --- | --- |
| Thêm/xóa/sửa file trong `data/corpus/` | **Có** | IDF phải học lại toàn corpus |
| Chỉ sửa code tiền xử lý / lọc từ khóa | Nên fit lại | Vectorizer cũ có thể lệch token |
| Chỉ sửa prompt LLM / Streamlit | Không | Restart web là đủ |
| KeyBERT / `vietnamese-bi-encoder` | Không | Pretrained; không có tập từ khóa vàng nên không fine-tune |

Train = fit `TfidfVectorizer`, không phải fine-tune Qwen. Local: `pip install -r requirements-train.txt` rồi mở notebook từ **thư mục gốc repo**.

## 6. Lỗi LLM thường gặp

| Thông báo | Việc cần làm |
| --- | --- |
| `403` + *sufficient permissions* | Token thiếu quyền Inference Providers; Save quyền hoặc tạo token mới. |
| `model_not_supported` | Chưa bật Featherless; `HF_PROVIDER=featherless-ai`. |
| Chưa có `HF_TOKEN` | Từ khóa vẫn chạy; ô tóm tắt báo thiếu cấu hình. |
| Hết credit / billing | Xem mục 3; mua credit hoặc đợi credit tháng sau. |

Bật **Lưu vào ChromaDB** không tải MiniLM: app dùng embedding hash nội bộ. Lịch sử nằm collection `keyword_summaries_local`.

## 7. Streamlit Community Cloud

1. Đưa repo lên GitHub.
2. [share.streamlit.io](https://share.streamlit.io) → New app → repo, branch, **Main file path:** `streamlit_app.py`.
3. Python **3.11 hoặc 3.12**.
4. Secrets:

```toml
HF_TOKEN = "hf_..."
HF_PROVIDER = "featherless-ai"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
```

5. Deploy. Lần đầu TF-IDF nhẹ. KeyBERT tải encoder; nếu app bị kill vì RAM, demo phương pháp 1 trên Cloud và quay video phương pháp 2 trên máy/Colab.

Cloud đọc: `requirements.txt`, `streamlit_app.py`, `src/`, `data/stopwords/`, `data/corpus/samples/`, `models/tfidf_vectorizer.joblib`. Full-text `baibao_khoahoc/` và `hanh_chinh/` không có trên Git. Không commit `.env` hay `.streamlit/secrets.toml`.

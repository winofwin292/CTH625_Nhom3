# Yêu cầu đồ án — đề tài 7

Nguồn: `docs/DoAnMonHoc_2026.docx` (Đồ án học phần Xử lý ngôn ngữ tự nhiên). Các mục dưới đây là nội dung đã trích từ file đó, không bổ sung quy định ngoài tài liệu.

## File cần nộp (áp dụng mọi đề tài)

1. **01 file Word báo cáo:** kiến trúc hệ thống, lý thuyết nền tảng (Transformer, LLMs, RAG, BERT, v.v.), kỹ thuật tiền xử lý tiếng Việt, đánh giá thực nghiệm.
2. **02 tập tin mã nguồn:**
   - 01 file huấn luyện / fine-tune mô hình (khuyến khích Jupyter Notebook / Google Colab).
   - 01 file mã nguồn ứng dụng Web (zip hoặc liên kết Git).

## Yêu cầu kỹ thuật hệ thống (chung)

- Web: giao diện trực quan. Công cụ được nêu: Streamlit, Gradio, hoặc FastAPI + React/Vue.
- Dữ liệu cấu trúc / phi cấu trúc lưu bằng CSDL phù hợp. Hệ được nêu: MongoDB, PostgreSQL, hoặc vector database (ChromaDB, Milvus, Pinecone).

## Đề tài 7 — Tóm tắt văn bản dựa trên trích xuất từ khóa

**Yêu cầu hệ thống:** Web Application, xử lý **một** văn bản đầu vào tiếng Việt (bài báo, tiểu luận, văn bản hành chính). Người dùng nắm ý chính qua **danh sách từ khóa** và **đoạn văn tóm tắt**.

### Bước 1 — Tiền xử lý

- Đọc văn bản từ `.txt`, `.pdf`, hoặc nhập trên giao diện.
- Tách từ tiếng Việt: underthesea hoặc pyvi.
- Xây dựng và áp dụng danh sách từ dừng (stop words) cải tiến cho tiếng Việt.

### Bước 2 — Trích xuất từ khóa

- **Phương pháp 1 (Baseline):** TF-IDF kết hợp từ loại; chỉ giữ Danh từ, Động từ, Tính từ; lấy từ có trọng số cao nhất.
- **Phương pháp 2 (Advanced Deep Learning):** KeyBERT kết hợp pretrained language model phù hợp tiếng Việt (PhoBERT-base hoặc vietnamese-bi-encoder).

### Bước 3 — Sinh tóm tắt bằng LLM

- Dùng từ khóa ở bước 2 làm **chỉ dẫn ngữ nghĩa (semantic anchors)**.
- Thiết kế prompt điều khiển LLM mã nguồn mở (Vistral-7B-Chat, PhoGPT, hoặc Qwen-2.5-7B-Instruct) để tạo văn bản tóm tắt.

## Điểm đề bài không nêu (cần nhóm chốt, không tự bịa)

- Tên 4 thành viên.
- Bộ ngữ liệu huấn luyện TF-IDF (đề tài 7 không gắn URL dataset như một số đề khác).
- Token / máy chủ LLM cụ thể để demo.
- Ngày hết hạn nộp và tên GVHD (không có trong file Word này).

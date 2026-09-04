# Lý thuyết — đề tài 7

Tài liệu này chỉ trình bày những thành phần **đề bài yêu cầu dùng**. Công thức viết theo nghĩa vận hành trong code, không thay bằng biến thể tự nghĩ ra.

## 1. Tiền xử lý tiếng Việt

Tiếng Việt không tách từ theo khoảng trắng. `underthesea.word_tokenize` trả về danh sách token (một token có thể gồm nhiều tiếng, ví dụ `học sinh`).

Gán nhãn từ loại: `underthesea.pos_tag`. Nhãn dùng trong đồ án lấy theo cột UTS trên wiki underthesea ([Mô tả dữ liệu bài toán POS Tag](https://github.com/undertheseanlp/underthesea/wiki/M%C3%B4-t%E1%BA%A3-d%E1%BB%AF-li%E1%BB%87u-b%C3%A0i-to%C3%A1n-POS-Tag)):

| Nhóm đề bài yêu cầu giữ | Nhãn UTS điển hình |
| --- | --- |
| Danh từ | `N`, `Nc`, `Np`, `Nu`, `Ny` |
| Động từ | `V`, `Vy` |
| Tính từ | `A` |

Trong code: giữ token có nhãn bắt đầu bằng `N`, `V` hoặc `A`.

Từ dừng: file `data/stopwords/vietnamese-stopwords.txt` copy từ [stopwords/vietnamese-stopwords](https://github.com/stopwords/vietnamese-stopwords). Phần cải tiến không thêm từ tự bịa, mà:

- chuẩn hóa Unicode NFC;
- so khớp cả dạng cách và dạng gạch dưới (`vui tính` / `vui_tính`) sau khi tách từ;
- bỏ token chỉ gồm dấu câu hoặc chữ số.

## 2. TF-IDF (phương pháp 1)

Gọi \(t\) là term, \(d\) là văn bản, \(N\) là số văn bản trong corpus.

sklearn `TfidfVectorizer` (mặc định `smooth_idf=True`, đúng với code) dùng:

\[
\mathrm{tfidf}(t, d) = \mathrm{tf}(t, d) \times \big(\ln \frac{1+N}{1+\mathrm{df}(t)} + 1\big)
\]

rồi chuẩn hóa L2 từng văn bản.

Luồng đề bài: tách từ → bỏ stop words → **chỉ giữ N/V/A** → tính TF-IDF → lấy top từ khóa.

Khi đã chạy notebook, hệ thống nạp `models/tfidf_vectorizer.joblib` (IDF học từ corpus). Khi chưa train, IDF được tính tạm bằng cách coi **từng câu** của văn bản hiện tại là một “văn bản” — vì đề bài 7 xử lý đúng một file đầu vào, không có IDF nền nếu chưa fit corpus.

## 3. KeyBERT (phương pháp 2)

KeyBERT (Maarten Grootendorst) không sinh từ mới. Các bước:

1. Tạo ứng viên n-gram từ văn bản đã tách từ.
2. Nhúng toàn văn và nhúng từng ứng viên bằng mô hình BERT/sentence embedding.
3. Xếp hạng theo cosine:

\[
\cos(a, b) = \frac{a \cdot b}{\|a\|\,\|b\|}
\]

Ứng viên gần vector văn bản nhất được coi là từ khóa.

Code dùng thư viện `keybert` với backbone [bkai-foundation-models/vietnamese-bi-encoder](https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder) (đúng tên đề bài nêu). Model card ghi **input phải đã word-segment**; từ đa tiếng viết `học_sinh`. Backbone của encoder này là PhoBERT-base-v2.

`use_mmr=True` là Maximal Marginal Relevance có sẵn trong KeyBERT: giảm trùng cụm gần giống nhau. Đây là tham số thư viện, không phải bước đề bài bắt buộc.

## 4. BERT / PhoBERT (để đọc báo cáo)

BERT (Devlin et al., 2019) là Transformer encoder, học hai nhiệm vụ: masked language model và next sentence prediction.

PhoBERT (Nguyen & Nguyen, 2020) là mô hình kiểu RoBERTa huấn luyện trên tiếng Việt đã tách từ. `vietnamese-bi-encoder` fine-tune tiếp PhoBERT-base-v2 cho bài toán nhúng câu (bài báo BKAI, arXiv:2403.01616).

## 5. LLM và neo ngữ nghĩa

Đề bài: từ khóa là **semantic anchors**, prompt điều khiển LLM mã nguồn mở (Vistral-7B-Chat, PhoGPT, hoặc Qwen-2.5-7B-Instruct).

Hệ thống dùng **Qwen/Qwen2.5-7B-Instruct** vì:

- nằm đúng danh sách đề bài;
- gọi qua Hugging Face Inference Providers (phù hợp Streamlit Cloud; không load model 7B trong process Streamlit).

Prompt hệ thống bắt buộc không bịa thông tin ngoài văn bản. Prompt người dùng nhét danh sách từ khóa rồi mới nhét văn bản gốc.

Transformer decoder (GPT-style) sinh token theo \(P(x_t \mid x_{<t})\). Đồ án không train lại LLM; chỉ thiết kế prompt.

## 6. ChromaDB

Đề bài chung bắt lưu CSDL (MongoDB, PostgreSQL, hoặc vector DB). ChromaDB lưu văn bản, từ khóa, tóm tắt để xem lại trên web. Việc trích xuất từ khóa **không** phụ thuộc truy vấn Chroma. App dùng embedding hash nội bộ (không tải `all-MiniLM-L6-v2`).

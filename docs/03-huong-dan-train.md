# Fit TF-IDF

Notebook `notebooks/train_keyword_extraction.ipynb` tự chứa code xử lý. Không cần import `src/`.

Việc notebook huấn luyện là fit TF-IDF trên các file `.txt`, rồi ghi `models/tfidf_vectorizer.joblib`. KeyBERT dùng `bkai-foundation-models/vietnamese-bi-encoder` có sẵn, không fine-tune. Qwen không được train trong notebook.

Nếu `data/corpus/` chỉ còn file trong `samples/`, ô fit giữ joblib đã có. Có thêm `.txt` ngoài `samples/` thì ô fit ghi đè joblib.

## Local

```bash
pip install -r requirements-train.txt
jupyter notebook notebooks/train_keyword_extraction.ipynb
```

1. Ở ô cấu hình, để `DATA_DIR = "data/corpus"` khi dữ liệu nằm trong project. Điền đường dẫn khác nếu file `.txt` ở chỗ khác.
2. Chạy lần lượt các ô.
3. Ô cuối mặc định đọc `data/corpus/samples/01_bai_bao.txt`. Đổi `FILE_PATH`, dán `TEXT`, hoặc đặt `METHOD = "keybert"`.
4. `DO_SUMMARY = True` in thêm tóm tắt khi môi trường có `HF_TOKEN`. Không có token thì vẫn in từ khóa.

Web nạp joblib mới sau khi tắt và chạy lại Streamlit.

## Google Colab

Notebook không có ô tiêu đề «Cài đặt». Ô code thứ hai, ngay dưới ô `DATA_DIR`, gắn Drive và cài thư viện.

1. Đưa dữ liệu lên Drive theo cây sau. File `.txt` nằm trong `corpus`. File từ dừng và joblib nằm cạnh thư mục `data`.

   `My Drive/CTH625_Nhom3/data/corpus/`

   `My Drive/CTH625_Nhom3/data/stopwords/vietnamese-stopwords.txt`

   `My Drive/CTH625_Nhom3/models/tfidf_vectorizer.joblib`

2. Mở notebook trên Colab. Ô code đầu tiên có `DATA_DIR = "data/corpus"`. Để nguyên khi Drive đúng cây trên. Thư mục khác thì sửa `DATA_DIR` thành đường dẫn đầy đủ, và điền `STOPWORDS_PATH`, `MODEL_PATH` nếu hai file kia không nằm đúng chỗ.
3. Chọn **Runtime → Run all**. Khi hiện cửa sổ Google Drive, bấm Allow.
4. Đợi ô fit in số văn bản. Ô cuối trên Colab hiện hộp chọn file `.txt` hoặc `.pdf`, rồi in từ khóa.

Muốn dùng KeyBERT: ở ô cuối sửa `METHOD = "tfidf"` thành `METHOD = "keybert"`. Chọn **Runtime → Change runtime type → T4 GPU → Save**. Colab khởi động lại và xóa kết quả đã chạy, nên chọn **Runtime → Run all** lần nữa, rồi Allow Drive nếu được hỏi. TF-IDF không cần GPU.

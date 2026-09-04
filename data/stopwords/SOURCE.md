# Nguồn danh sách từ dừng

File `vietnamese-stopwords.txt` lấy nguyên văn từ:

- Kho: https://github.com/stopwords/vietnamese-stopwords
- File: https://github.com/stopwords/vietnamese-stopwords/blob/master/vietnamese-stopwords.txt

Đây là danh sách công khai, không phải do nhóm tự soạn.

Phần "cải tiến" nằm ở `src/preprocess.py`: chuẩn hóa Unicode, thêm biến thể gạch dưới (phù hợp văn bản đã tách từ kiểu PhoBERT), và loại token chỉ gồm dấu câu/số.

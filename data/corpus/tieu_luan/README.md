# Tiểu luận / luận văn

Đặt PDF vào thư mục này. Git bỏ qua `*.pdf` và `*.txt`. Không đưa các file này lên internet.

Tách chữ:

```bash
python scripts/extract_tieu_luan.py
```

Script ghi một `.txt` cùng tên cạnh mỗi PDF. File dưới 8.000 ký tự hoặc gần như không có dấu tiếng Việt bị bỏ. Sau đó chạy lại notebook fit TF-IDF.

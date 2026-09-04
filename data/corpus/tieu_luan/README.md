# Tiểu luận — thêm thủ công (DSpace)

Script **không** tải được PDF tiểu luận/khóa luận từ DSpace khi chưa đăng nhập. OAI-PMH chỉ trả metadata.

## Đã quét

| Kho | OAI | Kết quả |
| --- | --- | --- |
| [repository.vnu.edu.vn](https://repository.vnu.edu.vn) (DSpace 7, ĐHQGHN) | `https://repository.vnu.edu.vn/server/oai/request` | Identify / ListSets / GetRecord **OK**. Bitstream `/bitstreams/{uuid}/download` trả trang **Đăng nhập**, không phải PDF. |
| [dspace.ctu.edu.vn](https://dspace.ctu.edu.vn) (ĐH Cần Thơ) | `https://dspace.ctu.edu.vn/oai/request` | OAI sống. Toàn văn luận văn trên LRC thường **cần tài khoản trường**. |
| `elib.vnuf.edu.vn`, `tailieuso.udn.vn`, HUST/HCMUS DSpace | — | Timeout hoặc DNS không ra. |

Bộ sưu tập gần **tiểu luận** nhất trên VNU (FYP / ESSAY / khóa luận):

- `col_VNU_123_33317` HUS — Student Reports (FYP/ESSAY)
- `col_VNU_123_33144` USSH — Student Reports (FYP/ESSAY)
- `col_VNU_123_33323` UET — Student Reports (FYP/ESSAY)
- `col_VNU_123_33728` IS — Student Final Year Project
- `col_VNU_123_169914` IS — Student Reports (SCIENTIFIC RESEARCH / ESSAY)

Luận văn thạc sĩ (dài hơn tiểu luận, vẫn dùng được cho TF-IDF): `HUS - Master Theses` (`col_VNU_123_33316`), `USSH - Master Theses` (`col_VNU_123_33313`), …

## Cách tải sau khi đăng nhập

1. Vào [repository.vnu.edu.vn](https://repository.vnu.edu.vn), đăng nhập tài khoản ĐHQGHN (hoặc DSpace trường bạn).
2. Mở một collection FYP/ESSAY ở bảng trên (hoặc tìm *Student Reports*).
3. Chỉ lấy mục có PDF toàn văn **sau login**, tiếng Việt, ≥ ~3.000 ký tự khi convert.
4. PDF → `.txt` UTF-8, bỏ bìa / mục lục / tài liệu tham khảo nếu quá nhiễu.
5. Đặt vào thư mục này: `001_ten-ngan.txt`. 10–20 file là đủ.
6. Ghi URL handle (ví dụ `https://repository.vnu.edu.vn/handle/VNU_123/...`) vào `manifest.jsonl` hoặc báo cáo.
7. Fit lại TF-IDF trong `notebooks/train_keyword_extraction.ipynb`.

Không cào LMS, Google Classroom, thư viện trả phí. Không copy luận văn HUST gom sẵn trên Hugging Face vào repo.

Nguồn thay thế nếu không có tài khoản DSpace: tiểu luận/đồ án của nhóm, hoặc PDF tác giả tự đăng kèm giấy phép mở.

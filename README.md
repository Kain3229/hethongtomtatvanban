# Hệ Thống Tóm Tắt Văn Bản

## Link sử dụng nhanh

- Streamlit app: https://hethongtomtat.streamlit.app/

## Lưu ý quan trọng

- Hệ thống được thiết kế để tóm tắt văn bản tiếng Anh.
- Kết quả ổn định nhất khi đầu vào là nội dung có cấu trúc tương đối rõ như bài giải thích, tin tức, hướng dẫn, FAQ, ghi chú theo ý, hoặc hội thoại ngắn.
- Nếu văn bản quá cảm tính, quá rời rạc, trộn nhiều ngôn ngữ, hoặc thiếu ngữ cảnh, chất lượng tóm tắt có thể giảm.

## Giới thiệu

Ứng dụng sử dụng mô hình Transformer để tóm tắt văn bản tiếng Anh và tự động chia nhỏ văn bản dài thành nhiều chunk trước khi sinh bản tóm tắt cuối cùng.

Hệ thống hiện giữ lại 2 mô hình thực dụng nhất:

- `t5-small`: nhẹ, nhanh, phù hợp cho demo và văn bản ngắn đến vừa
- `facebook/bart-large-cnn`: chậm hơn nhưng thường cho kết quả tốt hơn với văn bản dài, nhiều chi tiết, FAQ hoặc nội dung có cấu trúc rõ

## Hai mô hình đang hỗ trợ

| Mô hình | Vai trò | Dữ liệu / đặc điểm | Khi nên dùng |
|---|---|---|---|
| `t5-small` | Mặc định | T5 tiền huấn luyện trên C4 | Khi cần tốc độ, chạy ổn định, văn bản ngắn đến vừa |
| `facebook/bart-large-cnn` | Chất lượng cao | Fine-tune trên CNN/DailyMail cho tác vụ tóm tắt | Khi văn bản dài hơn, có nhiều chi tiết, FAQ, hướng dẫn hoặc bố cục rõ |

## Tính năng chính

- Tóm tắt văn bản tiếng Anh bằng giao diện Streamlit tiếng Việt
- Tự động đếm token đầu vào
- Tự động chia chunk cho văn bản dài
- Tăng khả năng xử lý nhiều kiểu nội dung như danh sách, FAQ, ghi chú có cấu trúc và một số dạng hội thoại ngắn
- Cho phép điều chỉnh độ dài tóm tắt
- Hiển thị thời gian xử lý và chi tiết từng chunk

## Cách dùng nhanh

1. Mở app tại: https://hethongtomtat.streamlit.app/
2. Chọn mô hình
3. Nhấn `Tải Mô Hình`
4. Dán văn bản tiếng Anh vào ô nhập liệu
5. Nhấn `Tóm Tắt Văn Bản`
6. Xem kết quả và thông tin chi tiết

## Khuyến nghị sử dụng

- Dùng `t5-small` nếu bạn cần tốc độ và chạy ổn định.
- Dùng `facebook/bart-large-cnn` nếu bạn ưu tiên chất lượng tóm tắt.
- Với đầu vào là bài báo, bản tin, nội dung phân tích, hướng dẫn, FAQ hoặc ghi chú có cấu trúc, `facebook/bart-large-cnn` thường cho kết quả tốt hơn.
- Với đầu vào ngắn, rõ ý, `t5-small` thường đã đủ tốt.

## Chạy local

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

App local mặc định mở tại `http://localhost:8501`.

## Cấu trúc dự án

```text
text_summarization_system/
├── app.py
├── summarizer.py
├── requirements.txt
└── README.md
```

## Ghi chú kỹ thuật

- `t5-small` là mô hình T5 tiền huấn luyện tổng quát trên dữ liệu tiếng Anh quy mô lớn.
- `facebook/bart-large-cnn` là mô hình BART đã được fine-tune cho tác vụ tóm tắt trên bộ dữ liệu CNN/DailyMail.
- Thuật toán hiện bổ sung bước nhận diện phong cách nội dung, tách câu theo dòng/bullet và chấm điểm câu theo hồ sơ văn bản để thích ứng tốt hơn với nhiều thể loại đầu vào.

## Trạng thái dự án

- Phiên bản: `v1.2`
- Giao diện: tiếng Việt
- Ngôn ngữ đầu vào tối ưu: tiếng Anh
- Mục tiêu: demo hệ thống tóm tắt văn bản dùng mô hình học sâu

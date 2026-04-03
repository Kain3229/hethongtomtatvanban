# Hệ Thống Tóm Tắt Văn Bản

Thông tin sinh viên

Nguyễn Hữu Thắng
DH23TIN09
236690
nguyenhuuthangys@gmail.com

## Link sử dụng nhanh

- Streamlit app: https://hethongtomtat.streamlit.app/

## Lưu ý quan trọng
- Vui lòng reboot app tại manager app ở góc dưới bên phải màn hình để sử dụng phiên bản mới nhất của ứng dụng
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

## Dùng model local thay vì cache

Hệ thống hiện ưu tiên tải model theo thứ tự sau:

1. Thư mục `models/` trong project
2. Cache Hugging Face sẵn có
3. Tải mới từ Hugging Face

Bạn có thể chép model đã tải sẵn vào project theo cấu trúc này:

```text
text_summarization_system/
├── models/
│   ├── t5-small/
│   │   ├── config.json
│   │   ├── tokenizer_config.json
│   │   ├── spiece.model
│   │   └── ...
│   └── facebook/
│       └── bart-large-cnn/
│           ├── config.json
│           ├── tokenizer_config.json
│           ├── merges.txt
│           ├── vocab.json
│           └── ...
├── app.py
├── summarizer.py
├── requirements.txt
└── README.md
```

Nếu bạn đã từng tải model bằng Hugging Face, trên Windows thường có thể lấy từ cache tại:

- `C:\Users\<ten-ban>\.cache\huggingface\hub\models--t5-small\snapshots\<hash>\`
- `C:\Users\<ten-ban>\.cache\huggingface\hub\models--facebook--bart-large-cnn\snapshots\<hash>\`

Chỉ cần copy toàn bộ file trong thư mục snapshot tương ứng vào:

- `models/t5-small/`
- `models/facebook/bart-large-cnn/`

Sau đó ứng dụng sẽ tự ưu tiên dùng bản local này và không cần tải lại bản khác nếu thư mục model local đã đầy đủ.

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

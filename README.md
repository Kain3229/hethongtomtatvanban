# Hệ Thống Tóm Tắt Văn Bản (Text Summarization System)
- hệ thống được triển khai tại https://hethongtomtat.streamlit.app/
  
🎯 **Ứng dụng AI tóm tắt văn bản tiếng Anh** sử dụng các mô hình Transformer hiện đại với hỗ trợ chia nhỏ tự động cho văn bản dài.

## ✨ Tính Năng Chính

- **Chia Nhỏ Thông Minh**: Tự động chia văn bản vượt 1024 tokens thành các chunk nhỏ
- **3 Mô Hình Lựa Chọn**: T5-small, BART-base, BART-large-CNN
- **Đếm Token Thời Gian Thực**: Biết chính xác số token trong văn bản
- **Giao Diện Web**: Ứng dụng Streamlit thân thiện, toàn tiếng Việt
- **Cấu Hình Linh Hoạt**: Điều chỉnh độ dài tóm tắt và các tham số khác
- **Xử Lý Thông Minh**: Tóm tắt từng chunk rồi ghép lại một cách thông minh

## 🚀 Quick Start (Bắt Đầu Nhanh)

### Cách 1: Một Dòng Lệnh (Linux/Mac)
```bash
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && streamlit run app.py
```

### Cách 2: Từng Bước (Khuyến Khích - Windows/Mac/Linux)

```bash
# 1. Tạo virtual environment
python -m venv venv

# 2. Kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Cài đặt dependencies
pip install -r requirements.txt

# 4. Tải dữ liệu NLTK (chạy 1 lần)
python -c "import nltk; nltk.download('punkt')"

# 5. Chạy ứng dụng
streamlit run app.py
```

Ứng dụng sẽ mở tại: **http://localhost:8501**

⚠️ **Lần Chạy Đầu Tiên**: Ứng dụng sẽ tự động download mô hình AI (~800MB - 5GB tùy mô hình), nên cần kết nối internet

## ☁️ Deploy Lên Streamlit Cloud

### Cấu hình đã chuẩn bị

- Ứng dụng đã dùng `st.cache_resource` để tránh tải lại mô hình không cần thiết
- `requirements.txt` đã được làm gọn để build ổn định hơn trên Linux của Streamlit Cloud
- `torch` đã được nâng lên bản tương thích với Python mặc định hiện tại của Streamlit Community Cloud
- Mô hình mặc định nên dùng là `t5-small` để giảm nguy cơ thiếu RAM khi deploy

### Cách deploy

1. Đẩy source code lên GitHub, **không** đẩy thư mục `venv/`
2. Vào Streamlit Community Cloud: `https://share.streamlit.io/`
3. Chọn repo này
4. Điền file chính là `app.py`
5. Deploy

### Khuyến nghị khi chạy trên cloud

- Nên dùng `t5-small` làm lựa chọn chính
- `facebook/bart-base` có thể chạy nhưng chậm hơn và tốn RAM hơn
- `facebook/bart-large-cnn` dễ vượt giới hạn tài nguyên của gói miễn phí
- Lần chạy đầu sẽ mất thời gian vì phải tải model từ Hugging Face

### Nếu deploy bị lỗi

- Kiểm tra lại `requirements.txt`
- Nếu log báo không tìm thấy `torch==2.2.2`, nguyên nhân thường là app đang build với Python mặc định mới hơn, trong khi bản `torch` đó không có wheel tương ứng
- Trong màn hình deploy của Streamlit, vào `Advanced settings` và chọn `Python 3.11` nếu bạn muốn giữ dependency cũ; với repo hiện tại thì có thể để Python mặc định vì `torch` đã được cập nhật
- Xác nhận repo không chứa `venv/`, `__pycache__/`, hoặc file cache lớn
- Nếu app bị crash khi tải model, chuyển về `t5-small`
- Nếu build timeout, redeploy lại sau khi app cache xong model nhẹ hơn

## 📖 Hướng Dẫn Sử Dụng

### 🌐 Sử Dụng Giao Diện Web (Khuyến Khích)

1. **Mở ứng dụng**: Đã chạy `streamlit run app.py` ở trên
2. **Trang Chủ**: Thông tin giới thiệu và điều hướng
3. **Công Cụ**: Thực hiện tóm tắt
   - Chọn mô hình phù hợp
   - Dán văn bản cần tóm tắt
   - Ấn nút "Tóm Tắt"
   - Xem kết quả chi tiết
4. **Hướng Dẫn**: Cách sử dụng chi tiết
5. **Về Ứng Dụng**: Thông tin kỹ thuật

### 🐍 Sử Dụng Trực Tiếp Trong Python

```python
from summarizer import TextSummarizer

# Khởi tạo với mô hình
summarizer = TextSummarizer(
    model_name="t5-small",  # hoặc facebook/bart-base, facebook/bart-large-cnn
    max_tokens=1024
)

# Tóm tắt văn bản
text = "Your long English text here..."
result = summarizer.summarize(
    text,
    max_length=150,
    min_length=50
)

# Hiển thị kết quả
print("Tóm tắt:", result['final_summary'])
print("Token gốc:", result['token_count'])
print("Cần chia chunk?", result['needs_chunking'])
```

### 📊 3 Mô Hình Được Hỗ Trợ

| Mô Hình | Dung Lượng | Tốc Độ | Chất Lượng | Khuyến Khích |
|---------|-----------|--------|----------|--------------|
| `t5-small` | ~800MB | ⚡⚡⚡ Rất nhanh | ⭐⭐⭐ Tốt | Văn bản ≤ 1K tokens |
| `facebook/bart-base` | ~2GB | ⚡⚡ Nhanh | ⭐⭐⭐⭐ Rất tốt | Văn bản 1K-3K tokens |
| `facebook/bart-large-cnn` | ~5GB | ⚡ Vừa | ⭐⭐⭐⭐⭐ Xuất sắc | Văn bản > 3K tokens |

## 💡 Mẹo Tối Ưu Hóa Hiệu Suất

### ⚙️ Cấu Hình Trong Ứng Dụng Web

- **Chọn Mô Hình**: Tùy theo độ dài văn bản (được gợi ý tự động)
- **Max Token Chunks**: 512-2048 (mặc định 1024)
- **Max Summary Length**: 50-500 (mặc định 150)
- **Min Summary Length**: 10-200 (mặc định 50)

### 🚀 Cách Tăng Tốc Độ

1. **Dùng GPU** (nếu có NVIDIA/CUDA)
   - Tự động phát hiện và sử dụng
   - **Nhanh hơn 5-10 lần** so với CPU

2. **Chọn Mô Hình Cập Nhật**
   - `t5-small`: Nhanh nhất (chỉ 800MB)
   - Tốc độ: small > base > large

3. **Tóm Tắt Ngắn**
   - Độ dài ngắn (50-100) → xử lý nhanh
   - Độ dài dài (200+) → chậm hơn

4. **Văn Bản Tối Ưu**
   - Tính năng chia chunk tối ưu cho: 2000-5000 tokens
   - Văn bản > 10000 tokens có thể mất vài phút

### 🩹 Khắc Phục Sự Cố

#### ❌ Lỗi: "Out of Memory" / "CUDA out of memory"
**Giải Pháp:**
```bash
# Dùng mô hình nhỏ hơn
# Thay vì: facebook/bart-large-cnn
# Sử dụng: t5-small
```

Hoặc giảm Max Token per Chunk từ 1024 xuống 512

#### ❌ Lỗi: Model download thất bại
**Giải Pháp:**
- Kiểm tra kết nối internet
- Cache lưu ở: `~/.cache/huggingface/`
- Thử tải thủ công:
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
```

#### ❌ Streamlit không mở trình duyệt (Windows)
**Giải Pháp:**
- Mở thủ công: http://localhost:8501
- Hoặc chạy: `streamlit run app.py --logger.level=debug`

#### ⚠️ Tóm tắt chất lượng kém
**Giải Pháp:**
- Dùng mô hình lớn hơn (t5-small → facebook/bart-base → facebook/bart-large-cnn)
- Kiểm tra văn bản đầu vào (phải tiếng Anh)

## 📁 Cấu Trúc Dự Án

```
text_summarization_system/
├── app.py              # 🎨 Ứng dụng Streamlit (tiếng Việt)
├── summarizer.py       # 🧠 Logic tóm tắt chính
├── requirements.txt    # 📦 Các thư viện cần thiết
├── README.md          # 📖 Hướng dẫn này
└── __pycache__/       # (Tạo động, không chỉnh sửa)
```

## 📦 Thư Viện Sử Dụng

| Thư Viện | Phiên Bản | Mục Đích |
|----------|-----------|---------|
| **transformers** | 4.35.2 | Pre-trained models từ Hugging Face |
| **torch** | 2.10.0 | Deep learning framework |
| **streamlit** | 1.28.1 | Web UI framework |
| **nltk** | 3.8.1 | Xử lý ngôn ngữ (sentence tokenize) |
| **sentencepiece** | 0.1.99 | Sub-word tokenization |
| **requests** | 2.31.0 | HTTP requests |

## 🔬 Cách Sử Dụng Nâng Cao

### Đếm Token Trước Tóm Tắt
```python
from summarizer import TextSummarizer

summarizer = TextSummarizer("t5-small")
text = "Your English text..."

token_count = summarizer.count_tokens(text)
print(f"Văn bản có {token_count} tokens")

# Chọn mô hình phù hợp
if token_count <= 1000:
    model = "t5-small"
elif token_count <= 3000:
    model = "facebook/bart-base"
else:
    model = "facebook/bart-large-cnn"
```

### Tóm Tắt Nhiều Văn Bản
```python
summarizer = TextSummarizer("facebook/bart-base")
texts = ["Text 1...", "Text 2...", "Text 3..."]

for i, text in enumerate(texts, 1):
    result = summarizer.summarize(text)
    print(f"Văn bản {i}: {result['final_summary']}")
```

### Xem Chi Tiết Chunks
```python
result = summarizer.summarize(long_text)

if result['needs_chunking']:
    for i, (chunk, summary) in enumerate(
        zip(result['chunks'], result['chunk_summaries']), 1
    ):
        print(f"\n--- Chunk {i} ---")
        print(f"Gốc: {chunk[:100]}...")
        print(f"Tóm tắt: {summary}")
```

## 📊 Hiệu Suất

### Thời Gian Xử Lý (CPU vs GPU)

```
1000 tokens:  CPU 0.5-1s  │  GPU 0.1-0.2s  (5-10x nhanh)
5000 tokens:  CPU 2-5s    │  GPU 0.3-0.5s  (5-10x nhanh)
10000 tokens: CPU 5-10s   │  GPU 1-2s      (5-10x nhanh)
```

### Chất Lượng Tóm Tắt

- **Giảm chiều dài**: 60-80% (mất 20-40% text)
- **Giữ thông tin chính**: 70-90% nội dung quan trọng
- **Độ chính xác**: Tùy mô hình (BART-large tốt nhất)

## ℹ️ Thông Tin Dự Án

**Phiên Bản:** v1.0 (Beta)  
**Trạng Thái:** Đang phát triển  
**Ngôn Ngữ UI:** Tiếng Việt 100%  
**Ngôn Ngữ Hỗ Trợ:** Tiếng Anh  
**License:** Open Source (Tự do sử dụng)

## 🤝 Hỗ Trợ

Nếu gặp vấn đề:

1. **Kiểm tra lại yêu cầu hệ thống**: Python 3.8+, 4GB RAM
2. **Đọc phần Troubleshooting** phía trên
3. **Xóa cache mô hình**: `rm -rf ~/.cache/huggingface/` (Linux/Mac) hoặc `rmdir %USERPROFILE%\.cache\huggingface\` (Windows)
4. **Cài đặt lại thư viện**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

## 📝 Lịch Sử Phiên Bản

### v1.0 (Beta) - Hiện Tại
- ✅ 3 mô hình chính: T5-small, BART-base, BART-large-CNN
- ✅ Giao diện Streamlit tiếng Việt
- ✅ Tính năng chia chunk tự động
- ✅ Đếm token, đề xuất mô hình
- ✅ Hỗ trợ GPU/CPU

---

**Phát triển với ❤️ bằng [Streamlit](https://streamlit.io/) & [Hugging Face](https://huggingface.co/)**
- Automatic chunking for long texts
- Streamlit web interface
- Support for multiple models
- Real-time token counting

---

**Last Updated**: March 2026

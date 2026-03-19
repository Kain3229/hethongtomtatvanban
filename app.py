"""
Streamlit Web App for Text Summarization
Provides a user-friendly interface for text summarization with chunking support
"""

import streamlit as st
from summarizer import TextSummarizer
import time


MODEL_OPTIONS = {
    "t5-small": {
        "label": "t5-small",
        "description": "Nhẹ nhất, tiền huấn luyện trên C4 và phù hợp cho văn bản tiếng Anh ngắn đến vừa"
    },
    "facebook/bart-large-cnn": {
        "label": "facebook/bart-large-cnn",
        "description": "Đã fine-tune trên CNN/DailyMail nên thường cho kết quả tốt nhất với bài viết tiếng Anh kiểu tin tức"
    }
}


@st.cache_resource(show_spinner=False)
def load_summarizer(model_name: str, max_tokens: int) -> TextSummarizer:
    return TextSummarizer(model_name=model_name, max_tokens=max_tokens)


def clear_text_input() -> None:
    st.session_state.input_text = ""
    st.session_state.text_input = ""
    st.session_state.result = None

# Page configuration
st.set_page_config(
    page_title="Hệ Thống Tóm Tắt Văn Bản",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .home-container {
        text-align: center;
        padding: 4rem 2rem;
    }
    .home-title {
        font-size: 3.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .home-subtitle {
        font-size: 1.3rem;
        color: #666;
        margin-bottom: 3rem;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .model-status-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.35rem;
        color: #4a4a4a;
        font-size: 0.9rem;
    }
    .model-status-dot {
        width: 0.65rem;
        height: 0.65rem;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }
    .model-status-dot.active {
        background: #22c55e;
        box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.18);
    }
    .model-status-dot.inactive {
        background: #b8c2cc;
    }
    /* Remove disabled cursor from textarea */
    textarea:disabled {
        cursor: text;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = None
if 'result' not in st.session_state:
    st.session_state.result = None
if 'current_page_index' not in st.session_state:
    st.session_state.current_page_index = 0
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""

# Sidebar navigation
pages = ["🏠 Trang Chủ", "🔧 Công Cụ", "📖 Hướng Dẫn Sử Dụng", "ℹ️ Về Ứng Dụng"]

with st.sidebar:
    st.header("📚 Menu")
    selected_page_index = st.radio(
        "Chọn trang",
        range(len(pages)),
        format_func=lambda i: pages[i],
        key="page_radio",
        index=st.session_state.current_page_index
    )
    st.session_state.current_page_index = selected_page_index
    page = pages[selected_page_index]

# ============== HOME PAGE ==============
if page == "🏠 Trang Chủ":
    st.markdown("""
    <div class="home-container">
        <div class="home-title">📝 Tóm Tắt Văn Bản</div>
        <div class="home-subtitle">Công cụ AI mạnh mẽ để tóm tắt bất kỳ văn bản nào</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
        <h3>⚡ Nhanh & Hiệu Quả</h3>
        <p>Xử lý văn bản dài ngay lập tức với AI tiên tiến</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
        <h3>🎯 Chính Xác</h3>
        <p>Bản tóm tắt giữ lại các thông tin quan trọng nhất</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
        <h3>🎨 Dễ Sử Dụng</h3>
        <p>Giao diện đơn giản, dễ dàng cho bất kỳ ai</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Start button - full width
    if st.button("🚀 Bắt Đầu Tóm Tắt", use_container_width=True, key="start_home"):
        st.session_state.current_page_index = 1  # Index của "🔧 Công Cụ"
        st.rerun()
    
    st.markdown("---")
    
    # Navigation buttons - split 50-50
    col_nav1, col_nav2 = st.columns(2)
    
    with col_nav1:
        if st.button("📖 Hướng Dẫn Sử Dụng", use_container_width=True):
            st.session_state.current_page_index = 2  # Index của "📖 Hướng Dẫn Sử Dụng"
            st.rerun()
    
    with col_nav2:
        if st.button("ℹ️ Về Ứng Dụng", use_container_width=True):
            st.session_state.current_page_index = 3  # Index của "ℹ️ Về Ứng Dụng"
            st.rerun()

# ============== SETTINGS PAGE ==============
elif page == "🔧 Công Cụ":
    st.title("🔧 Công Cụ Tóm Tắt")
    st.caption(
        "Ứng dụng hỗ trợ tốt nhất cho văn bản tiếng Anh, đặc biệt là nội dung mang phong cách báo chí hoặc giải thích. "
        "t5-small là mô hình T5 tiền huấn luyện trên C4, còn facebook/bart-large-cnn được fine-tune trên CNN/DailyMail nên thường chính xác hơn với bài viết tiếng Anh dạng tin tức."
    )
    
    # Settings in columns for better layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🛠️ Tùy Chỉnh Mô Hình")
        
        model_option = st.selectbox(
            "Chọn Mô Hình",
            list(MODEL_OPTIONS.keys()),
            index=0
        )

        st.caption(MODEL_OPTIONS[model_option]["description"])
        
        st.markdown("---")
        
        max_tokens = st.slider(
            "Số Token Tối Đa Trên Mỗi Chunk",
            min_value=512,
            max_value=2048,
            value=1024,
            step=256
        )
        
        max_summary_length = st.slider(
            "Độ Dài Tối Đa Của Tóm Tắt",
            min_value=50,
            max_value=500,
            value=150,
            step=10
        )

        max_allowed_min_length = min(200, max_summary_length)
        
        min_summary_length = st.slider(
            "Độ Dài Tối Thiểu Của Tóm Tắt",
            min_value=10,
            max_value=max_allowed_min_length,
            value=min(50, max_allowed_min_length),
            step=10
        )
        
        st.markdown("---")
        
        # Model Loading Progress Display
        st.markdown("**📦 Trạng Thái Các Mô Hình:**")
        
        models_status = {
            "t5-small": False,
            "facebook/bart-large-cnn": False
        }
        
        # Check which model is currently loaded
        if st.session_state.summarizer:
            current_model = st.session_state.summarizer.model_name
            
            for model_name in models_status.keys():
                if current_model and model_name in str(current_model):
                    models_status[model_name] = True
        
        # Display status
        for model, loaded in models_status.items():
            status_class = "active" if loaded else "inactive"
            st.markdown(
                f"""
                <div class="model-status-item">
                    <span class="model-status-dot {status_class}"></span>
                    <span>{model}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        if st.button("🔄 Tải Mô Hình", use_container_width=True, key="load_model"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text(f"⏳ Đang tải {model_option}...")
                progress_bar.progress(30)
                
                st.session_state.summarizer = load_summarizer(model_option, max_tokens)
                
                progress_bar.progress(80)
                st.session_state.model_loaded = True
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                st.success(f"✅ {model_option} đã tải thành công!")
                if st.session_state.summarizer.max_tokens != max_tokens:
                    st.info(
                        f"Mô hình này chỉ hỗ trợ tối đa {st.session_state.summarizer.max_tokens} token mỗi chunk. "
                        "Giá trị đã được tự động điều chỉnh để tránh lỗi khi deploy."
                    )
            except Exception as e:
                progress_bar.progress(0)
                status_text.empty()
                st.error(f"❌ Lỗi: {str(e)}")
        
        # Current model status
        st.markdown("---")
        if st.session_state.summarizer:
            st.success(f"✅ Mô hình đã sẵn sàng")
        else:
            st.warning("⚠️ Chưa tải mô hình")
    
    with col2:
        st.subheader("📝 Nhập Văn Bản")
        input_text = st.text_area(
            "Dán văn bản của bạn ở đây:",
            height=400,
            placeholder="Nhập hoặc dán văn bản mà bạn muốn tóm tắt...",
            label_visibility="collapsed",
            key="text_input"
        )
        st.session_state.input_text = input_text
        
        if input_text and st.session_state.summarizer:
            token_count = st.session_state.summarizer.count_tokens(input_text)
            st.info(f"📊 Số token: **{token_count}** tokens")
            
            # Model recommendation based on token count
            st.markdown("**💡 Khuyến Nghị Mô Hình:**")
            if token_count <= 1200:
                st.success("✅ t5-small: Nhanh nhất và phù hợp cho văn bản tiếng Anh ngắn đến vừa")
            else:
                st.info("ℹ️ facebook/bart-large-cnn: Phù hợp hơn khi cần chất lượng cao với văn bản dài hoặc phong cách báo chí")
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("✨ Tóm Tắt Văn Bản", use_container_width=True, key="summarize"):
                    if not input_text.strip():
                        st.error("❌ Vui lòng nhập một số văn bản để tóm tắt")
                    elif min_summary_length > max_summary_length:
                        st.error("❌ Độ dài tối thiểu không được lớn hơn độ dài tối đa")
                    else:
                        with st.spinner("Đang tóm tắt..."):
                            try:
                                start_time = time.time()
                                st.session_state.result = st.session_state.summarizer.summarize(
                                    input_text,
                                    max_length=max_summary_length,
                                    min_length=min_summary_length
                                )
                                elapsed_time = time.time() - start_time
                                st.session_state.result['elapsed_time'] = elapsed_time
                                st.success("✅ Tóm tắt hoàn thành!")
                            except Exception as e:
                                st.error(f"❌ Lỗi khi tóm tắt: {str(e)}")
            
            with col_btn2:
                if st.button(
                    "🗑️ Xóa Văn Bản",
                    use_container_width=True,
                    key="clear",
                    on_click=clear_text_input
                ):
                    st.rerun()
        
        elif not st.session_state.summarizer:
            st.warning("⚠️ Vui lòng tải mô hình trước")
    
    # Results section
    if st.session_state.result:
        st.markdown("---")
        st.subheader("📤 Kết Quả Tóm Tắt")
        
        result = st.session_state.result
        summary_text = result['final_summary']
        
        st.text_area(
            "Bản tóm tắt được tạo:",
            value=summary_text,
            height=150,
            disabled=True,
            label_visibility="collapsed"
        )
        
        # Detailed results
        st.markdown("---")
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.metric("Token Gốc", result['token_count'])
        
        with col_info2:
            st.metric("Thời Gian", f"{result['elapsed_time']:.2f}s")
        
        with col_info3:
            status = "Có" if result['needs_chunking'] else "Không"
            st.metric("Chia Chunk", status)
        
        # Show chunks if needed
        if result['needs_chunking']:
            st.markdown("#### 📄 Chi Tiết Các Chunk")
            st.info(f"Văn bản được chia thành **{result['num_chunks']}** chunk")
            
            for i, (chunk, summary) in enumerate(zip(result['chunks'], result['chunk_summaries']), 1):
                with st.expander(f"📄 Chunk {i}"):
                    st.markdown("**Chunk Gốc:**")
                    st.text_area(
                        "Chunk gốc:",
                        value=chunk,
                        height=120,
                        disabled=True,
                        label_visibility="collapsed"
                    )
                    st.markdown("**Bản Tóm Tắt:**")
                    st.success(summary)

# ============== GUIDE PAGE ==============
elif page == "📖 Hướng Dẫn Sử Dụng":
    st.title("📖 Hướng Dẫn Sử Dụng")
    
    col_guide1, col_guide2 = st.columns([1, 1])
    
    with col_guide1:
        st.markdown("""
        ## 🚀 Cách Sử Dụng
        
        ### Bước 1: Khởi Động
        1. Vào trang **"Công Cụ"**
        2. Chọn mô hình (mặc định: t5-small)
        3. Nhấp **"Tải Mô Hình"**
        
        ### Bước 2: Nhập Văn Bản
        1. Dán/gõ văn bản vào hộp
        2. Xem **đề xuất mô hình** tự động
        3. Điều chỉnh độ dài nếu cần
        
        ### Bước 3: Tóm Tắt
        1. Nhấp **"✨ Tóm Tắt"**
        2. Chờ xử lý
        3. Xem kết quả
        
        ### Bước 4: Xem Chi Tiết
        - Văn bản > 1024 token → chia **chunk**
        - Kéo dọc để xem chunk riêng
        - Kiểm tra **thời gian** & **token count**
        """)
    
    with col_guide2:
        st.markdown("""
        ## 📊 Chọn Mô Hình

        Lưu ý: hệ thống cho kết quả ổn định nhất với **văn bản tiếng Anh** có cách viết rõ ràng, mạch lạc, đặc biệt là nội dung theo **phong cách báo chí** hoặc giải thích.
        
        | Kích Thước | Khuyến Nghị |
        |-----------|------------|
        | **≤ 1.2K** | 🟢 t5-small |
        | **> 1.2K** | 🔴 facebook/bart-large-cnn |
        
        ### Chi Tiết Mô Hình
        
        **t5-small** (800MB)
        - Nhanh nhất
        - Phù hợp cho văn bản tiếng Anh ngắn đến vừa
        
        **facebook/bart-large-cnn** (5GB)
        - Chất lượng cao hơn
        - Tốt nhất cho bài viết tiếng Anh kiểu tin tức, báo chí hoặc nội dung dài
        """)
    

# ============== ABOUT PAGE ==============
elif page == "ℹ️ Về Ứng Dụng":
    st.title("ℹ️ Về Ứng Dụng")
    
    st.markdown("""
    ## 🎯 Giới Thiệu
    
    **Hệ Thống Tóm Tắt Văn Bản** là ứng dụng web sử dụng trí tuệ nhân tạo (AI) để tóm tắt và rút gọn nội dung văn bản dài. 
    Công cụ này được xây dựng đặc biệt để xử lý **văn bản tiếng Anh**, giúp người dùng tiết kiệm thời gian đọc và nắm bắt 
    thông tin chính yếu một cách nhanh chóng.
    
    Đây là một **phiên bản thử nghiệm (Beta)** nhằm kiểm chứng hiệu suất của các mô hình học sâu hiện đại trong lĩnh vực 
    xử lý ngôn ngữ tự nhiên (NLP). Ứng dụng tích hợp hai mô hình từ Hugging Face Hub, cho phép người dùng 
    lựa chọn giữa tốc độ xử lý và chất lượng tóm tắt trong hai kiến trúc **T5** và **BART**.
    
    ---
    
    ## ✨ Tính Năng
    
    ### ⚡ Xử Lý Nhanh & Hiệu Quả
    - Tóm tắt văn bản trong vài giây
    - Hỗ trợ GPU để tăng tốc độ lên tới 5-10 lần
    - Tự động chia nhỏ văn bản dài thành các chunk có thể xử lý
    
    ### 🧠 2 Mô Hình AI Lựa Chọn
    - **T5-small** (800MB): Nhẹ và nhanh, phù hợp văn bản tiếng Anh ngắn đến vừa
    - **BART-large-CNN** (5GB): Chất lượng cao hơn, phù hợp bài viết dài và nội dung mang phong cách báo chí
    - Tự động đề xuất mô hình phù hợp dựa trên độ dài văn bản
    
    ### 📊 Thông Tin Chi Tiết
    - Đếm token thời gian thực
    - Hiển thị thời gian xử lý
    - Xem chi tiết từng phần (chunk) riêng biệt
    - Xem tóm tắt từng phần một cách rõ ràng
    
    ### 🎨 Giao Diện Thân Thiện
    - Thiết kế hiện đại, dễ nhìn
    - Hỗ trợ 100% tiếng Việt
    - Dễ sử dụng cho mọi trình độ
    - Tương thích với tất cả thiết bị (desktop/mobile)
    
    ---
    
    ## 💻 Công Nghệ Sử Dụng
    
    | Thành Phần | Chi Tiết |
    |-----------|---------|
    | **AI Framework** | Transformers (Hugging Face) |
    | **Deep Learning** | PyTorch |
    | **Web Framework** | Streamlit |
    | **Mô Hình** | T5-small, BART-large-CNN |
    | **Xử Lý Ngôn Ngữ** | NLTK |
    
    **Chi Tiết Kỹ Thuật:** Ứng dụng sử dụng các mô hình từ Hugging Face. `t5-small` là mô hình T5 tiền huấn luyện tổng quát trên dữ liệu tiếng Anh quy mô lớn, còn `facebook/bart-large-cnn` được fine-tune cho bài toán tóm tắt trên bộ dữ liệu CNN/DailyMail nên phù hợp hơn với bài viết tiếng Anh kiểu tin tức.
    
    ---
    
    ## 📋 Yêu Cầu Hệ Thống
    
    | Yêu Cầu | Chi Tiết |
    |--------|---------|
    | **Python** | 3.8 trở lên |
    | **RAM** | 4GB tối thiểu |
    | **GPU** | Tùy chọn (khuyến khích) |
    | **Hệ Điều Hành** | Windows / Mac / Linux |
    | **Trình Duyệt** | Chrome / Firefox / Safari |
    
    **Dung Lượng Mô Hình:**
    - T5-small: ~800MB
    - BART-large-CNN: ~5GB
    
    ---
    
    ## 📈 Hiệu Suất & Tốc Độ
    
    ### Thời Gian Xử Lý (Ước Tính)
    
    **Văn bản 1000 tokens:**
    - CPU: ~0.5-1 giây
    - GPU: ~0.2 giây
    
    **Văn bản 5000 tokens:**
    - CPU: ~2-5 giây
    - GPU: ~0.5 giây
    
    **Văn bản 10000+ tokens:**
    - CPU: ~5-10 giây
    - GPU: ~1-2 giây
    
    ### Chất Lượng Tóm Tắt
    - Giảm chiều dài văn bản: 60-80%
    - Giữ lại thông tin chính: 20-40%
    - Mất mát thông tin: Tối thiểu
    
    ---
    
    ## 👥 Dành Cho Ai?
    
    ✅ **Sinh viên** - Tóm tắt tài liệu học tập  
    ✅ **Nhà Báo/Biên Tập** - Xử lý tin tức và bài viết  
    ✅ **Nhân Viên Nghiên Cứu** - Phân tích tài liệu khoa học  
    ✅ **Chuyên Gia Công Nghệ** - Tích hợp vào ứng dụng khác  
    ✅ **Mọi Người** - Tiết kiệm thời gian đọc hiệu quả  
    
    ---
    
    ## ℹ️ Thông Tin Thêm
    
    **Phiên Bản:** v1.2 (Beta)  
    **Trạng Thái:** Còn đang phát triển  
    **Nguồn Mô Hình:** Hugging Face Hub  
    **Framework:** Streamlit  
    
    Ứng dụng này sử dụng mã nguồn mở và mô hình AI miễn phí từ cộng đồng Hugging Face. 
    Nó được phát triển liên tục và có thể cập nhật thêm tính năng mới trong tương lai.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px;'>
    Được tạo với ❤️ bằng Streamlit | Hệ Thống Tóm Tắt Văn Bản v1.2
</div>
""", unsafe_allow_html=True)

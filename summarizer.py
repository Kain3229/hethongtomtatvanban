"""
Module Tóm Tắt Văn Bản
Xử lý tóm tắt văn bản với hỗ trợ chia nhỏ
"""

import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tải dữ liệu NLTK yêu cầu
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class TextSummarizer:
    """
    Lớp xử lý tóm tắt văn bản với chia nhỏ tự động cho văn bản dài.
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn", max_tokens: int = 1024):
        """
        Khởi tạo bộ tóm tắt với mô hình được huấn luyện trước.
        
        Args:
            model_name: Mô hình HuggingFace để sử dụng tóm tắt
            max_tokens: Số token tối đa mỗi chunk (mặc định: 1024)
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Đang tải mô hình: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = min(max_tokens, self._get_model_token_limit())
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Mô hình đã tải trên thiết bị: {self.device}")

    def _get_model_token_limit(self) -> int:
        model_limit = getattr(self.tokenizer, 'model_max_length', self.max_tokens)
        if not isinstance(model_limit, int) or model_limit <= 0 or model_limit > 100000:
            return self.max_tokens
        return model_limit
    
    def count_tokens(self, text: str) -> int:
        """
        Đếm số lượng token trong văn bản.
        
        Args:
            text: Văn bản đầu vào
            
        Returns:
            Số lượng token
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Chia văn bản thành các câu sử dụng NLTK.
        
        Args:
            text: Văn bản đầu vào
            
        Returns:
            Danh sách các câu
        """
        sentences = nltk.sent_tokenize(text)
        return sentences
    
    def chunk_text_by_tokens(self, text: str, max_tokens: int = None) -> List[str]:
        """
        Chia văn bản thành các chunk dựa trên số lượng token.
        
        Args:
            text: Văn bản đầu vào
            max_tokens: Số token tối đa trên mỗi chunk (mặc định: self.max_tokens)
            
        Returns:
            Danh sách các chunk văn bản
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # Nếu một câu vượt quá max_tokens, nó sẽ là chunk riêng
            if sentence_tokens > max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_token_count = 0
                chunks.append(sentence)
            # Nếu thêm câu này không vượt giới hạn, thêm nó vào
            elif current_token_count + sentence_tokens <= max_tokens:
                current_chunk.append(sentence)
                current_token_count += sentence_tokens
            # Ngược lại, bắt đầu chunk mới
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_token_count = sentence_tokens
        
        # Thêm chunk cuối cùng
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    @torch.inference_mode()
    def summarize_single(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """
        Tóm tắt một chunk văn bản đơn.
        
        Args:
            text: Văn bản đầu vào (nên <= max_tokens)
            max_length: Độ dài tối đa của bản tóm tắt
            min_length: Độ dài tối thiểu của bản tóm tắt
            
        Returns:
            Văn bản được tóm tắt
        """
        inputs = self.tokenizer.encode(
            text,
            return_tensors="pt",
            max_length=self.max_tokens,
            truncation=True
        )
        inputs = inputs.to(self.device)
        
        summary_ids = self.model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            early_stopping=True
        )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 50) -> Dict:
        """
        Tóm tắt văn bản với chia nhỏ tự động nếu cần.
        
        Args:
            text: Văn bản đầu vào
            max_length: Độ dài tối đa của mỗi chunk tóm tắt
            min_length: Độ dài tối thiểu của mỗi chunk tóm tắt
            
        Returns:
            Dictionary chứa:
            - 'original_text': Văn bản đầu vào gốc
            - 'token_count': Số token trong văn bản gốc
            - 'chunks': Danh sách các chunk văn bản
            - 'chunk_summaries': Danh sách tóm tắt cho mỗi chunk
            - 'final_summary': Bản tóm tắt kết hợp của tất cả chunk
        """
        # Đếm token trong văn bản gốc
        token_count = self.count_tokens(text)
        logger.info(f"Số token của văn bản gốc: {token_count}")
        
        # Nếu văn bản nằm trong giới hạn kích thước, tóm tắt trực tiếp
        if token_count <= self.max_tokens:
            logger.info("Văn bản nằm trong giới hạn token, tóm tắt trực tiếp")
            summary = self.summarize_single(text, max_length, min_length)
            return {
                'original_text': text,
                'token_count': token_count,
                'chunks': [text],
                'chunk_summaries': [summary],
                'final_summary': summary,
                'needs_chunking': False
            }
        
        # Nếu văn bản vượt quá giới hạn, chia nó thành các chunk
        logger.info(f"Văn bản vượt quá giới hạn token ({self.max_tokens}), đang chia văn bản")
        chunks = self.chunk_text_by_tokens(text, self.max_tokens)
        logger.info(f"Văn bản được chia thành {len(chunks)} chunk")
        
        # Tóm tắt mỗi chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Đang tóm tắt chunk {i+1}/{len(chunks)}")
            summary = self.summarize_single(chunk, max_length, min_length)
            chunk_summaries.append(summary)
        
        # Ghép tất cả các bản tóm tắt
        combined_summary = " ".join(chunk_summaries)
        
        # Nếu bản tóm tắt kết hợp vẫn quá dài, tóm tắt lại
        if self.count_tokens(combined_summary) > self.max_tokens:
            logger.info("Bản tóm tắt kết hợp vượt quá giới hạn token, đang tóm tắt lại")
            final_summary = self.summarize_single(combined_summary, max_length * 2, min_length)
        else:
            final_summary = combined_summary
        
        return {
            'original_text': text,
            'token_count': token_count,
            'chunks': chunks,
            'chunk_summaries': chunk_summaries,
            'final_summary': final_summary,
            'needs_chunking': True,
            'num_chunks': len(chunks)
        }

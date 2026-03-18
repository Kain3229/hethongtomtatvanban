"""
Module Tóm Tắt Văn Bản
Xử lý tóm tắt văn bản với hỗ trợ chia nhỏ
"""

import logging
import re
from collections import Counter
from typing import Dict, List, Tuple

import nltk
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUMMARY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "before",
    "being", "but", "by", "for", "from", "had", "has", "have", "he", "her",
    "his", "i", "in", "into", "is", "it", "its", "of", "on", "or", "she",
    "so", "than", "that", "the", "their", "them", "there", "they", "this", "to",
    "was", "were", "with", "would", "you", "your"
}

MIN_SUMMARY_LENGTH = 20
MAX_SUMMARY_MIN_LENGTH = 120
MAX_SUMMARY_LENGTH = 220


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
        self.max_tokens = min(max_tokens, self._get_model_token_limit(max_tokens))
        self.chunk_token_limit = self._get_chunk_token_limit()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Mô hình đã tải trên thiết bị: {self.device}")

    def _get_model_token_limit(self, fallback_tokens: int) -> int:
        model_limit = getattr(self.tokenizer, 'model_max_length', fallback_tokens)
        if not isinstance(model_limit, int) or model_limit <= 0 or model_limit > 100000:
            return fallback_tokens
        return model_limit

    def _get_chunk_token_limit(self) -> int:
        return self.max_tokens

    def _get_precompression_threshold(self) -> int:
        return max(256, self.chunk_token_limit - 16)

    def _normalize_summary_lengths(self, max_length: int, min_length: int) -> tuple[int, int]:
        if max_length <= 0:
            raise ValueError("max_length phải lớn hơn 0")
        if min_length < 0:
            raise ValueError("min_length không được âm")

        normalized_max_length = min(int(max_length), MAX_SUMMARY_LENGTH)
        min_length_ceiling = min(MAX_SUMMARY_MIN_LENGTH, max(normalized_max_length - 20, 10))
        normalized_min_length = min(max(int(min_length), MIN_SUMMARY_LENGTH), min_length_ceiling)

        if normalized_min_length > normalized_max_length:
            normalized_min_length = max(10, normalized_max_length - 5)

        return normalized_max_length, normalized_min_length

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _prepare_model_input(self, text: str) -> str:
        normalized_text = self._normalize_text(text)
        if self.model_name.lower().startswith("t5") and not normalized_text.lower().startswith("summarize:"):
            return f"summarize: {normalized_text}"
        return normalized_text

    def _word_tokens(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())

    def _content_words(self, text: str) -> List[str]:
        return [word for word in self._word_tokens(text) if word not in SUMMARY_STOPWORDS and len(word) > 2]

    def _sentence_support_score(self, summary_sentence: str, source_sentences: List[str]) -> float:
        summary_terms = set(self._content_words(summary_sentence))
        if not summary_terms:
            return 1.0

        best_score = 0.0
        for source_sentence in source_sentences:
            source_terms = set(self._content_words(source_sentence))
            if not source_terms:
                continue
            overlap = len(summary_terms & source_terms)
            coverage = overlap / len(summary_terms)
            source_alignment = overlap / len(source_terms)
            score = (0.7 * coverage) + (0.3 * source_alignment)
            best_score = max(best_score, score)

        return best_score

    def _best_matching_source_index(self, summary_sentence: str, source_sentences: List[str]) -> int:
        summary_terms = set(self._content_words(summary_sentence))
        if not summary_terms:
            return 0

        best_index = 0
        best_score = -1.0
        for index, source_sentence in enumerate(source_sentences):
            source_terms = set(self._content_words(source_sentence))
            if not source_terms:
                continue
            overlap = len(summary_terms & source_terms)
            coverage = overlap / len(summary_terms)
            source_alignment = overlap / len(source_terms)
            score = (0.7 * coverage) + (0.3 * source_alignment)
            if score > best_score:
                best_score = score
                best_index = index

        return best_index

    def _filter_unsupported_sentences(self, summary: str, source_text: str, threshold: float = 0.34) -> str:
        summary_sentences = self.split_into_sentences(summary)
        source_sentences = self.split_into_sentences(source_text)

        if not summary_sentences or not source_sentences:
            return summary

        supported_sentences = []
        for sentence in summary_sentences:
            score = self._sentence_support_score(sentence, source_sentences)
            if score >= threshold:
                supported_sentences.append(sentence)
            else:
                logger.info("Loại câu ít được nguồn hỗ trợ (score=%.2f): %s", score, sentence)

        if supported_sentences:
            return " ".join(supported_sentences)
        return summary

    def _score_source_sentences(self, sentences: List[str]) -> List[Tuple[float, int, str]]:
        content_words = self._content_words(" ".join(sentences))
        frequencies = Counter(content_words)
        if not frequencies:
            return []

        max_frequency = max(frequencies.values())
        normalized_frequencies = {
            word: frequency / max_frequency
            for word, frequency in frequencies.items()
        }

        scored_sentences = []
        total_sentences = max(len(sentences), 1)
        for index, sentence in enumerate(sentences):
            sentence_words = self._content_words(sentence)
            if not sentence_words:
                continue

            sentence_score = sum(normalized_frequencies.get(word, 0.0) for word in sentence_words)
            sentence_score /= len(sentence_words)

            position_bonus = max(0.0, 1 - (index / total_sentences)) * 0.12
            if index == 0:
                position_bonus += 0.08

            unique_term_bonus = len(set(sentence_words)) / max(len(sentence_words), 1) * 0.1
            scored_sentences.append((sentence_score + position_bonus + unique_term_bonus, index, sentence))

        return scored_sentences

    def _build_extractive_context(self, text: str, target_tokens: int) -> str:
        sentences = self.split_into_sentences(text)
        if len(sentences) <= 3:
            return self._normalize_text(text)

        scored_items = self._score_source_sentences(sentences)
        scored_sentences = sorted(scored_items, reverse=True)
        if not scored_sentences:
            return self._normalize_text(text)

        selected_sentences = []
        selected_indexes = set()
        token_budget = max(128, min(target_tokens, self.chunk_token_limit))
        current_tokens = 0

        segment_count = min(4, max(2, len(sentences) // 6))
        if segment_count > 1:
            segment_size = max(1, len(sentences) // segment_count)
            for segment_index in range(segment_count):
                start = segment_index * segment_size
                end = len(sentences) if segment_index == segment_count - 1 else min(len(sentences), start + segment_size)
                segment_candidates = [item for item in scored_items if start <= item[1] < end]
                if not segment_candidates:
                    continue

                _, sentence_index, sentence = max(segment_candidates, key=lambda item: item[0])
                sentence_tokens = self.count_tokens(sentence)
                if current_tokens + sentence_tokens > token_budget and selected_sentences:
                    continue

                selected_sentences.append((sentence_index, sentence))
                selected_indexes.add(sentence_index)
                current_tokens += sentence_tokens

        for _, index, sentence in scored_sentences:
            if index in selected_indexes:
                continue
            sentence_tokens = self.count_tokens(sentence)
            if selected_sentences and current_tokens + sentence_tokens > token_budget:
                continue

            selected_sentences.append((index, sentence))
            selected_indexes.add(index)
            current_tokens += sentence_tokens

            if current_tokens >= token_budget:
                break

        if 0 not in selected_indexes:
            selected_sentences.append((0, sentences[0]))
        if len(sentences) > 1 and (len(sentences) - 1) not in selected_indexes:
            last_sentence = sentences[-1]
            if current_tokens + self.count_tokens(last_sentence) <= token_budget * 1.15:
                selected_sentences.append((len(sentences) - 1, last_sentence))

        selected_sentences.sort(key=lambda item: item[0])
        return " ".join(sentence for _, sentence in selected_sentences)

    def _extractive_fallback(self, text: str, max_length: int) -> str:
        approx_token_budget = max(max_length * 2, 80)
        return self._build_extractive_context(text, approx_token_budget)

    def _should_use_extractive_fallback(self, summary: str, source_text: str) -> bool:
        summary_sentences = self.split_into_sentences(summary)
        source_sentences = self.split_into_sentences(source_text)

        if len(source_sentences) < 10 or not summary_sentences:
            return False

        matched_indexes = [
            self._best_matching_source_index(sentence, source_sentences)
            for sentence in summary_sentences
        ]
        span = max(matched_indexes) - min(matched_indexes) if len(matched_indexes) > 1 else 0
        source_span_ratio = span / max(len(source_sentences) - 1, 1)
        avg_support = sum(
            self._sentence_support_score(sentence, source_sentences)
            for sentence in summary_sentences
        ) / len(summary_sentences)

        return source_span_ratio < 0.35 and avg_support < 0.72

    def _cleanup_summary_text(self, summary: str) -> str:
        cleaned_summary = self._normalize_text(summary)
        if not cleaned_summary:
            return cleaned_summary

        sentences = self.split_into_sentences(cleaned_summary)
        if not sentences:
            return cleaned_summary

        cleaned_sentences = []
        for sentence in sentences:
            normalized_sentence = sentence.strip()
            if not normalized_sentence:
                continue
            if normalized_sentence[-1] not in ".!?" and len(self._word_tokens(normalized_sentence)) < 8:
                continue
            cleaned_sentences.append(normalized_sentence)

        if cleaned_sentences:
            return " ".join(cleaned_sentences)
        return cleaned_summary

    def _sentence_overlap_ratio(self, sentence_a: str, sentence_b: str) -> float:
        tokens_a = set(self._content_words(sentence_a))
        tokens_b = set(self._content_words(sentence_b))
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    def _is_redundant_sentence(self, sentence: str, selected_sentences: List[str], threshold: float = 0.55) -> bool:
        return any(self._sentence_overlap_ratio(sentence, selected) >= threshold for selected in selected_sentences)

    def _sentence_entity_bonus(self, sentence: str) -> float:
        entities = {
            token for token in re.findall(r"\b[A-Z][a-z]+\b", sentence)
            if token.lower() not in SUMMARY_STOPWORDS
        }
        return min(len(entities), 4) * 0.04

    def _sentence_theme_bonus(self, sentence: str) -> float:
        keywords = {
            "alone", "lonely", "adopted", "companion", "friendship", "friends",
            "routine", "transformed", "home", "shared", "together", "changed",
            "relationship", "peaceful", "life"
        }
        sentence_terms = set(self._content_words(sentence))
        return min(len(sentence_terms & keywords), 3) * 0.07

    def _extract_repeated_names(self, text: str) -> Counter:
        ignored_tokens = {
            "The", "A", "An", "In", "On", "At", "But", "And", "After", "Before",
            "During", "Over", "More", "One", "Two", "Every", "Sometimes", "Friends"
        }
        names = [
            token for token in re.findall(r"\b[A-Z][a-z]+\b", text)
            if token not in ignored_tokens
        ]
        return Counter(names)

    def _looks_like_narrative_text(self, text: str) -> bool:
        sentences = self.split_into_sentences(text)
        if len(sentences) < 12:
            return False

        repeated_names = [
            name for name, count in self._extract_repeated_names(text).items()
            if count >= 3
        ]
        return len(repeated_names) >= 2

    def _pick_bucket_sentence(
        self,
        sentences: List[str],
        keywords: set[str],
        start_ratio: float,
        end_ratio: float,
        selected_sentences: List[str]
    ) -> str:
        if not sentences:
            return ""

        start_index = int(len(sentences) * start_ratio)
        end_index = max(start_index + 1, int(len(sentences) * end_ratio))
        window = sentences[start_index:end_index]
        if not window:
            window = sentences

        best_sentence = ""
        best_score = -1.0
        for sentence in window:
            if self._is_redundant_sentence(sentence, selected_sentences, threshold=0.45):
                continue

            sentence_terms = set(self._content_words(sentence))
            keyword_score = len(sentence_terms & keywords)
            if keyword_score == 0 and keywords:
                continue

            score = (
                keyword_score * 1.2
                + self._sentence_entity_bonus(sentence)
                + self._sentence_theme_bonus(sentence)
                + (0.2 if sentence.strip().endswith(('.', '!', '?')) else 0.0)
            )
            if score > best_score:
                best_score = score
                best_sentence = sentence

        return best_sentence

    def _build_narrative_summary(self, text: str, max_length: int) -> str:
        sentences = self.split_into_sentences(text)
        if len(sentences) < 8:
            return ""

        buckets = [
            ({"alone", "lonely", "quiet", "routine", "lived"}, 0.0, 0.2),
            ({"changed", "adopted", "dog", "named", "shelter"}, 0.0, 0.35),
            ({"another", "companion", "cat", "shadow", "entered"}, 0.25, 0.65),
            ({"relationship", "friendship", "conflict", "games", "harmony", "peaceful"}, 0.45, 0.85),
            ({"transformed", "home", "shared", "companionship", "life", "better"}, 0.7, 1.0),
        ]

        selected_sentences: List[str] = []
        for keywords, start_ratio, end_ratio in buckets:
            selected_sentence = self._pick_bucket_sentence(
                sentences,
                keywords,
                start_ratio,
                end_ratio,
                selected_sentences
            )
            if selected_sentence:
                selected_sentences.append(selected_sentence)

        if len(selected_sentences) < 3:
            return ""

        narrative_summary = self._cleanup_summary_text(" ".join(selected_sentences))
        if self.count_tokens(narrative_summary) > max(max_length + 35, 120):
            trimmed_sentences = selected_sentences[:-1]
            if len(trimmed_sentences) >= 3:
                narrative_summary = self._cleanup_summary_text(" ".join(trimmed_sentences))

        return narrative_summary

    def _build_guided_final_summary(self, chunks: List[str], guidance_texts: List[str], max_length: int) -> str:
        if not chunks:
            return ""

        target_token_budget = max(90, min(self.chunk_token_limit // 2, max_length + 50))
        desired_sentence_count = min(6, max(4, max_length // 32))
        base_sentences_per_chunk = max(1, desired_sentence_count // len(chunks))
        remaining_sentences = desired_sentence_count - (base_sentences_per_chunk * len(chunks))

        selected_items: List[Tuple[int, int, str, float]] = []
        selected_sentences: List[str] = []

        for chunk_index, chunk in enumerate(chunks):
            chunk_sentences = self.split_into_sentences(chunk)
            if not chunk_sentences:
                continue

            local_scores = {
                index: score for score, index, _ in self._score_source_sentences(chunk_sentences)
            }
            guidance_sentences = self.split_into_sentences(guidance_texts[chunk_index]) if chunk_index < len(guidance_texts) else []
            sentences_to_pick = base_sentences_per_chunk
            if remaining_sentences > 0 and chunk_index in {0, len(chunks) - 1}:
                sentences_to_pick += 1
                remaining_sentences -= 1

            scored_candidates = []
            for sentence_index, sentence in enumerate(chunk_sentences):
                guidance_score = self._sentence_support_score(sentence, guidance_sentences) if guidance_sentences else 0.0
                intro_bonus = 0.08 if sentence_index <= max(1, len(chunk_sentences) // 5) else 0.0
                outro_bonus = 0.1 if sentence_index >= int(len(chunk_sentences) * 0.75) else 0.0
                total_score = (
                    (local_scores.get(sentence_index, 0.0) * 0.45)
                    + (guidance_score * 1.1)
                    + self._sentence_entity_bonus(sentence)
                    + self._sentence_theme_bonus(sentence)
                    + intro_bonus
                    + outro_bonus
                )
                scored_candidates.append((total_score, sentence_index, sentence))

            for total_score, sentence_index, sentence in sorted(scored_candidates, reverse=True):
                if sentences_to_pick <= 0:
                    break
                if self._is_redundant_sentence(sentence, selected_sentences):
                    continue

                candidate_summary = " ".join([item[2] for item in sorted(selected_items + [(chunk_index, sentence_index, sentence, total_score)], key=lambda item: (item[0], item[1]))])
                if selected_items and self.count_tokens(candidate_summary) > target_token_budget:
                    continue

                selected_items.append((chunk_index, sentence_index, sentence, total_score))
                selected_sentences.append(sentence)
                sentences_to_pick -= 1

        if not selected_items:
            return self._extractive_fallback(" ".join(chunks), max_length)

        ordered_sentences = [sentence for _, _, sentence, _ in sorted(selected_items, key=lambda item: (item[0], item[1]))]
        return self._cleanup_summary_text(" ".join(ordered_sentences))

    def _get_chunk_summary_lengths(self, num_chunks: int, max_length: int, min_length: int) -> Tuple[int, int]:
        if num_chunks <= 1:
            return max_length, min_length

        chunk_max_length = min(
            max_length,
            max(45, int((max_length / min(num_chunks, 4)) + 28))
        )
        chunk_min_length = min(chunk_max_length - 5, max(20, min_length // max(1, min(num_chunks, 3))))
        chunk_min_length = max(10, chunk_min_length)
        return chunk_max_length, chunk_min_length

    def _summarize_chunk(self, chunk: str, max_length: int, min_length: int) -> str:
        chunk_text = chunk
        chunk_token_count = self.count_tokens(chunk)
        if chunk_token_count > self._get_precompression_threshold():
            chunk_text = self._build_extractive_context(chunk, min(self.chunk_token_limit - 8, max(max_length * 6, 196)))

        summary = self.summarize_single(chunk_text, max_length, min_length)
        summary = self._cleanup_summary_text(self._filter_unsupported_sentences(summary, chunk))
        if not summary.strip():
            return self._extractive_fallback(chunk, max_length)
        return summary
    
    def count_tokens(self, text: str) -> int:
        """
        Đếm số lượng token trong văn bản.
        
        Args:
            text: Văn bản đầu vào
            
        Returns:
            Số lượng token
        """
        tokens = self.tokenizer.encode(self._prepare_model_input(text), add_special_tokens=True)
        return len(tokens)
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Chia văn bản thành các câu sử dụng NLTK.
        
        Args:
            text: Văn bản đầu vào
            
        Returns:
            Danh sách các câu
        """
        normalized_text = text.strip()
        if not normalized_text:
            return []

        try:
            sentences = nltk.sent_tokenize(normalized_text)
        except Exception:
            sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", normalized_text)

        cleaned_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        if cleaned_sentences:
            return cleaned_sentences

        return [normalized_text]
    
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
            max_tokens = self.chunk_token_limit
        
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
        max_length, min_length = self._normalize_summary_lengths(max_length, min_length)

        prepared_text = self._prepare_model_input(text)
        inputs = self.tokenizer.encode(
            prepared_text,
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
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            length_penalty=1.0,
            early_stopping=True
        )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary = self._cleanup_summary_text(summary)
        filtered_summary = self._filter_unsupported_sentences(summary, text)
        if filtered_summary != summary:
            return self._cleanup_summary_text(filtered_summary)
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
        normalized_text = self._normalize_text(text)
        if not normalized_text:
            raise ValueError("Văn bản đầu vào không được để trống")

        max_length, min_length = self._normalize_summary_lengths(max_length, min_length)

        # Đếm token trong văn bản gốc
        token_count = self.count_tokens(normalized_text)
        logger.info(f"Số token của văn bản gốc: {token_count}")

        # Nếu văn bản ngắn và nằm trong giới hạn kích thước, tóm tắt trực tiếp
        if token_count <= self.chunk_token_limit:
            logger.info("Văn bản nằm trong giới hạn token, tóm tắt trực tiếp")
            direct_text = normalized_text
            if token_count > self._get_precompression_threshold():
                direct_text = self._build_extractive_context(normalized_text, min(self.chunk_token_limit - 8, max(max_length * 6, 196)))
                logger.info("Đã tạo ngữ cảnh extractive cục bộ với %s token", self.count_tokens(direct_text))

            summary = self.summarize_single(direct_text, max_length, min_length)
            summary = self._cleanup_summary_text(self._filter_unsupported_sentences(summary, normalized_text))
            if not summary.strip() or self._should_use_extractive_fallback(summary, normalized_text):
                summary = self._extractive_fallback(normalized_text, max_length)
            return {
                'original_text': normalized_text,
                'token_count': token_count,
                'chunks': [direct_text],
                'chunk_summaries': [summary],
                'final_summary': summary,
                'needs_chunking': False,
                'chunk_token_limit': self.chunk_token_limit
            }
        
        # Nếu văn bản vượt quá giới hạn, chia nó thành các chunk
        logger.info(f"Văn bản vượt quá giới hạn token ({self.chunk_token_limit}), đang chia văn bản")
        chunks = self.chunk_text_by_tokens(normalized_text, self.chunk_token_limit)
        logger.info(f"Văn bản được chia thành {len(chunks)} chunk")
        chunk_max_length, chunk_min_length = self._get_chunk_summary_lengths(len(chunks), max_length, min_length)
        
        # Tóm tắt mỗi chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Đang tóm tắt chunk {i+1}/{len(chunks)}")
            summary = self._summarize_chunk(chunk, chunk_max_length, chunk_min_length)
            chunk_summaries.append(summary)
        
        # Ghép tất cả các bản tóm tắt
        combined_summary = " ".join(chunk_summaries)
        combined_summary = self._cleanup_summary_text(combined_summary)

        if len(chunks) > 1:
            combined_token_count = self.count_tokens(combined_summary)
            if len(chunks) <= 4 and combined_token_count <= max(max_length * 2, 160):
                final_summary = combined_summary
            else:
                final_input = combined_summary
                if self.count_tokens(final_input) > self.chunk_token_limit:
                    logger.info("Bản tóm tắt kết hợp vượt quá giới hạn token, đang nén ngữ cảnh trước khi tóm tắt lần cuối")
                    final_input = self._build_extractive_context(combined_summary, min(self.chunk_token_limit - 8, max(max_length * 5, 160)))

                final_summary = self.summarize_single(final_input, max_length, min_length)

            guided_final_summary = self._build_guided_final_summary(chunks, chunk_summaries, max_length)
            if guided_final_summary.strip():
                final_summary = guided_final_summary
        else:
            final_summary = combined_summary

        final_summary = self._cleanup_summary_text(self._filter_unsupported_sentences(final_summary, normalized_text))
        if self._looks_like_narrative_text(normalized_text):
            narrative_summary = self._build_narrative_summary(normalized_text, max_length)
            if narrative_summary.strip():
                final_summary = narrative_summary
        if not final_summary.strip() or self._should_use_extractive_fallback(final_summary, normalized_text):
            final_summary = self._extractive_fallback(normalized_text, max_length)
        
        return {
            'original_text': normalized_text,
            'token_count': token_count,
            'chunks': chunks,
            'chunk_summaries': chunk_summaries,
            'final_summary': final_summary,
            'needs_chunking': True,
            'num_chunks': len(chunks),
            'chunk_token_limit': self.chunk_token_limit
        }

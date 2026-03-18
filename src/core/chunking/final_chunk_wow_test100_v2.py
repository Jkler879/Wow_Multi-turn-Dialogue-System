# langchain1.0_chunk_wow_test100_optimized_final.py
from config.paths import paths
import json
import logging
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

# LangChain 1.0正确导包语句
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wow_langchain_chunking_final.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WoWTextSplitter(TextSplitter):
    """WoW对话分割器 - 优化版"""

    def __init__(self, max_chunk_size: int = 1600, min_chunk_size: int = 200, **kwargs):
        super().__init__(**kwargs)
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def split_text(self, text: str) -> List[str]:
        """分割文本"""
        return [text]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档 - 优化版"""
        all_chunks = []
        for doc in documents:
            # 从metadata获取对话数据
            dialog_item = doc.metadata.get('original_data', {})
            dialog_turns = dialog_item.get('dialog', [])
            dialog_index = doc.metadata.get('dialog_index', 0)

            if not dialog_turns:
                chunks = super().split_documents([doc])
                all_chunks.extend(chunks)
                continue

            # 构建完整内容并检查大小
            full_content = self._build_content(dialog_item, dialog_turns)
            content_length = len(full_content)

            logger.debug(f"对话 {dialog_index}: {content_length}字符, {len(dialog_turns)}轮次")

            # 智能分割决策
            if content_length <= self.max_chunk_size:
                # 保持完整
                chunk_doc = self._create_document(dialog_item, dialog_turns, dialog_index, 0, "complete")
                all_chunks.append(chunk_doc)
            else:
                # 需要分割 - 使用语义感知分割
                chunks = self._semantic_aware_split(dialog_item, dialog_turns, dialog_index)
                all_chunks.extend(chunks)

        # 修复：构建分块链接关系 - 确保在返回前正确构建链接
        all_chunks = self._build_chunk_links(all_chunks)
        logger.info(f"分块链接构建完成，共处理 {len(all_chunks)} 个分块")

        return all_chunks

    def _semantic_aware_split(self, dialog_item: Dict, dialog_turns: List[Dict], dialog_index: int) -> List[Document]:
        """语义感知的对话分割"""
        if len(dialog_turns) <= 3:
            # 对话太短，保持完整
            return [self._create_document(dialog_item, dialog_turns, dialog_index, 0, "complete_short")]

        # 寻找最佳语义分割点
        best_split_point = self._find_semantic_split_point(dialog_turns)

        if best_split_point == -1:
            # 找不到语义分割点，使用安全分割
            logger.warning(f"对话 {dialog_index} 无法找到语义分割点，使用安全分割")
            return self._safe_split_dialog(dialog_item, dialog_turns, dialog_index)

        # 执行分割
        first_half = dialog_turns[:best_split_point]
        second_half = dialog_turns[best_split_point:]

        # 检查分割后的大小是否合理
        content1 = self._build_content(dialog_item, first_half)
        content2 = self._build_content(dialog_item, second_half)

        if len(content1) < self.min_chunk_size or len(content2) < self.min_chunk_size:
            # 分割后块太小，保持完整
            logger.warning(f"对话 {dialog_index} 分割后块太小，保持完整")
            return [self._create_document(dialog_item, dialog_turns, dialog_index, 0, "complete_oversized")]

        chunk1 = self._create_document(dialog_item, first_half, dialog_index, 0, "semantic_split")
        chunk2 = self._create_document(dialog_item, second_half, dialog_index, 1, "semantic_split")

        logger.info(f"对话 {dialog_index} 语义分割: {len(first_half)} + {len(second_half)} 轮次")

        return [chunk1, chunk2]

    def _find_semantic_split_point(self, dialog_turns: List[Dict]) -> int:
        """寻找最佳语义分割点"""
        if len(dialog_turns) <= 4:
            return -1

        # 策略1: 寻找话题转换点
        topic_change_point = self._find_topic_change_point(dialog_turns)
        if topic_change_point != -1:
            return topic_change_point

        # 策略2: 寻找长的沉默间隙（基于轮次长度变化）
        gap_point = self._find_conversation_gap(dialog_turns)
        if gap_point != -1:
            return gap_point

        # 策略3: 在中间点附近寻找安全分割
        return self._find_safe_midpoint(dialog_turns)

    def _find_topic_change_point(self, dialog_turns: List[Dict]) -> int:
        """寻找话题转换点"""
        for i in range(2, len(dialog_turns) - 2):
            if dialog_turns[i]['speaker'] == "0_Wizard":
                current_text = dialog_turns[i]['text']
                if (len(current_text) > 150 and
                        any(keyword in current_text.lower() for keyword in
                            ['so', 'therefore', 'in conclusion', 'now', 'next', 'another'])):
                    if self._is_absolutely_safe_split(dialog_turns, i + 1):
                        return i + 1
        return -1

    def _find_conversation_gap(self, dialog_turns: List[Dict]) -> int:
        """寻找对话间隙"""
        turn_lengths = [len(turn['text']) for turn in dialog_turns]

        changes = []
        for i in range(1, len(turn_lengths)):
            if turn_lengths[i - 1] > 0:
                change = abs(turn_lengths[i] - turn_lengths[i - 1]) / turn_lengths[i - 1]
                changes.append((i, change))

        if changes:
            max_change_point = max(changes, key=lambda x: x[1])[0]
            if (max_change_point > 2 and max_change_point < len(dialog_turns) - 2 and
                    self._is_absolutely_safe_split(dialog_turns, max_change_point)):
                return max_change_point

        return -1

    def _find_safe_midpoint(self, dialog_turns: List[Dict]) -> int:
        """寻找安全的中点分割"""
        mid_point = len(dialog_turns) // 2

        for offset in range(len(dialog_turns) // 4 + 1):
            if mid_point + offset < len(dialog_turns) - 1:
                if self._is_absolutely_safe_split(dialog_turns, mid_point + offset):
                    return mid_point + offset
            if mid_point - offset > 1:
                if self._is_absolutely_safe_split(dialog_turns, mid_point - offset):
                    return mid_point - offset

        return -1

    def _safe_split_dialog(self, dialog_item: Dict, dialog_turns: List[Dict], dialog_index: int) -> List[Document]:
        """安全分割超长对话"""
        best_split_point = self._find_safe_midpoint(dialog_turns)

        if best_split_point == -1:
            logger.warning(f"对话 {dialog_index} 无法安全分割，保持完整")
            return [self._create_document(dialog_item, dialog_turns, dialog_index, 0, "complete_oversized")]

        first_half = dialog_turns[:best_split_point]
        second_half = dialog_turns[best_split_point:]

        chunk1 = self._create_document(dialog_item, first_half, dialog_index, 0, "safe_split")
        chunk2 = self._create_document(dialog_item, second_half, dialog_index, 1, "safe_split")

        logger.info(f"对话 {dialog_index} 安全分割: {len(first_half)} + {len(second_half)} 轮次")

        return [chunk1, chunk2]

    def _is_absolutely_safe_split(self, dialog_turns: List[Dict], split_index: int) -> bool:
        """绝对安全的分割检查"""
        if split_index <= 0 or split_index >= len(dialog_turns):
            return False

        if not self._absolute_qa_protection(dialog_turns, split_index):
            return False

        if not self._protect_conversation_flow(dialog_turns, split_index):
            return False

        return True

    def _absolute_qa_protection(self, dialog_turns: List[Dict], split_index: int) -> bool:
        """绝对问答对保护"""
        for i in range(max(0, split_index - 2), min(len(dialog_turns) - 1, split_index + 2)):
            current_turn = dialog_turns[i]
            next_turn = dialog_turns[i + 1]

            is_qa_pair = (
                    current_turn['speaker'] == "0_Wizard" and
                    next_turn['speaker'] == "1_Apprentice"
            )

            if is_qa_pair and (i < split_index <= i + 1):
                return False

        return True

    def _protect_conversation_flow(self, dialog_turns: List[Dict], split_index: int) -> bool:
        """保护对话流连续性"""
        prev_speaker = dialog_turns[split_index - 1]['speaker']
        curr_speaker = dialog_turns[split_index]['speaker']

        if prev_speaker == curr_speaker:
            return False

        min_turns = 2
        if split_index < min_turns or (len(dialog_turns) - split_index) < min_turns:
            return False

        return True

    def _build_content(self, dialog_item: Dict, turns: List[Dict]) -> str:
        """构建对话内容"""
        content_parts = []
        content_parts.append(f"Topic: {dialog_item.get('chosen_topic', 'Unknown')}")
        content_parts.append("")

        for turn in turns:
            speaker = "Wizard" if turn['speaker'] == "0_Wizard" else "Apprentice"
            content_parts.append(f"{speaker}: {turn['text']}")

        return "\n".join(content_parts)

    def _generate_content_hash(self, content: str) -> str:
        """生成内容哈希"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _calculate_evidence_metrics(self, turns: List[Dict], dialog_item: Dict) -> Dict[str, Any]:
        """计算证据相关指标"""
        stats = {
            "total_retrieved_passages": 0,
            "total_retrieved_topics": 0,
            "has_checked_evidence": False,
            "wizard_eval_score": dialog_item.get('wizard_eval', 3),
            "evidence_density": 0.0
        }

        # 统计检索段落和主题
        for turn in turns:
            # 检索段落计数
            retrieved = turn.get('retrieved_passages', [])
            if retrieved:
                stats["total_retrieved_passages"] += len(retrieved)

            # 检索主题计数
            topics = turn.get('retrieved_topics', [])
            if topics:
                stats["total_retrieved_topics"] += len(topics)

            # 检查是否有验证证据
            if turn.get('checked_sentence') or turn.get('checked_passage'):
                stats["has_checked_evidence"] = True

        # 计算证据密度
        total_turns = len(turns)
        if total_turns > 0:
            stats["evidence_density"] = (
                                                stats["total_retrieved_passages"] +
                                                stats["total_retrieved_topics"]
                                        ) / total_turns

        return stats

    def _detect_content_type(self, turns: List[Dict]) -> str:
        """改进版内容类型检测"""
        if not turns:
            return "unknown"

        # 分析文本特征
        all_text = " ".join(turn['text'] for turn in turns)
        text_lower = all_text.lower()

        # 1. 问答检测
        question_indicators = ['?', 'what', 'how', 'why', 'when', 'where', 'which']
        has_questions = any(indicator in text_lower for indicator in question_indicators)

        # 2. 解释性内容检测
        explanation_indicators = ['because', 'therefore', 'thus', 'means that', 'in other words']
        has_explanation = any(indicator in text_lower for indicator in explanation_indicators)

        # 3. 事实陈述检测
        fact_indicators = ['fact', 'according to', 'research shows', 'studies indicate']
        has_facts = any(indicator in text_lower for indicator in fact_indicators)

        # 4. 统计信息
        avg_turn_length = sum(len(turn['text']) for turn in turns) / len(turns)
        wizard_turns = sum(1 for turn in turns if "Wizard" in turn['speaker'])
        apprentice_turns = sum(1 for turn in turns if "Apprentice" in turn['speaker'])

        # 5. 基于多重条件的分类
        if has_questions and apprentice_turns > wizard_turns:
            return "qa_pair"
        elif has_explanation and avg_turn_length > 150:
            return "explanation"
        elif has_facts and avg_turn_length < 100:
            return "fact_statement"
        elif len(turns) <= 2:
            return "brief_exchange"
        elif wizard_turns == 0:  # 只有学徒在说话
            return "user_query"
        elif apprentice_turns == 0:  # 只有巫师在说话
            return "expert_response"
        else:
            return "dialogue"

    def _detect_speaker_role(self, turns: List[Dict]) -> str:
        """改进版：说话者角色检测"""
        if not turns:
            return "unknown"

        # 统计各角色类型的轮次
        speaker_types = {}
        for turn in turns:
            speaker = turn['speaker']
            # 解析说话者类型 (0_Wizard -> Wizard, 1_Apprentice -> Apprentice)
            if '_' in speaker:
                role_type = speaker.split('_')[1]  # 取后半部分
            else:
                role_type = speaker

            speaker_types[role_type] = speaker_types.get(role_type, 0) + 1

        # 确定主导角色
        if not speaker_types:
            return "unknown"

        main_role = max(speaker_types, key=speaker_types.get)
        main_count = speaker_types[main_role]
        total_turns = len(turns)

        # 如果主导角色占比超过70%，则认为是该角色主导
        if main_count / total_turns >= 0.7:
            return main_role.lower()
        elif main_count / total_turns >= 0.5:
            return f"{main_role.lower()}_dominant"
        else:
            # 分析对话模式
            return self._analyze_conversation_pattern(turns)

    def _analyze_conversation_pattern(self, turns: List[Dict]) -> str:
        """分析对话模式"""
        if len(turns) < 2:
            return "single_turn"

        # 检查问答模式
        qa_patterns = 0
        for i in range(len(turns) - 1):
            current = turns[i]['speaker']
            next_speaker = turns[i + 1]['speaker']

            # Wizard回答Apprentice问题
            if "Wizard" in current and "Apprentice" in next_speaker:
                qa_patterns += 1
            # Apprentice回答Wizard问题
            elif "Apprentice" in current and "Wizard" in next_speaker:
                qa_patterns += 1

        qa_ratio = qa_patterns / (len(turns) - 1) if len(turns) > 1 else 0

        if qa_ratio > 0.6:
            return "qa_exchange"
        else:
            return "balanced_dialogue"

    def _build_chunk_links(self, all_chunks: List[Document]) -> List[Document]:
        """构建分块间的链接关系 - 完全修复版本"""
        logger.info("开始构建分块链接关系...")

        # 按对话分组
        chunks_by_dialog = {}
        for chunk in all_chunks:
            dialog_id = chunk.metadata.get('dialog_id')
            if not dialog_id:
                logger.warning(f"分块缺少dialog_id: {chunk.metadata.get('chunk_id', 'unknown')}")
                continue

            if dialog_id not in chunks_by_dialog:
                chunks_by_dialog[dialog_id] = []
            chunks_by_dialog[dialog_id].append(chunk)

        logger.info(f"按对话分组完成，共 {len(chunks_by_dialog)} 个对话组")

        # 对每个对话内的分块按顺序排序并建立链接
        total_links_created = 0

        for dialog_id, chunks in chunks_by_dialog.items():
            # 按chunk_seq排序
            try:
                chunks.sort(key=lambda x: x.metadata['chunk_seq'])
            except KeyError as e:
                logger.warning(f"对话 {dialog_id} 的分块缺少chunk_seq: {e}")
                continue

            logger.debug(
                f"对话 {dialog_id} 有 {len(chunks)} 个分块，序列: {[chunk.metadata['chunk_seq'] for chunk in chunks]}")

            # 建立前后链接
            for i, chunk in enumerate(chunks):
                # 前一个分块
                if i > 0:
                    prev_chunk = chunks[i - 1]
                    prev_chunk_id = prev_chunk.metadata.get('chunk_id')
                    if prev_chunk_id:
                        # 关键修复：直接修改metadata字典
                        chunk.metadata['prev_chunk_id'] = prev_chunk_id
                        total_links_created += 1
                        logger.debug(f"分块 {chunk.metadata['chunk_id']} 的前分块: {prev_chunk_id}")
                    else:
                        chunk.metadata['prev_chunk_id'] = ""
                        logger.warning(f"前分块缺少chunk_id: {prev_chunk.metadata}")
                else:
                    chunk.metadata['prev_chunk_id'] = ""  # 第一个分块没有前分块
                    logger.debug(f"分块 {chunk.metadata['chunk_id']} 是第一个分块")

                # 后一个分块
                if i < len(chunks) - 1:
                    next_chunk = chunks[i + 1]
                    next_chunk_id = next_chunk.metadata.get('chunk_id')
                    if next_chunk_id:
                        # 关键修复：直接修改metadata字典
                        chunk.metadata['next_chunk_id'] = next_chunk_id
                        total_links_created += 1
                        logger.debug(f"分块 {chunk.metadata['chunk_id']} 的后分块: {next_chunk_id}")
                    else:
                        chunk.metadata['next_chunk_id'] = ""
                        logger.warning(f"后分块缺少chunk_id: {next_chunk.metadata}")
                else:
                    chunk.metadata['next_chunk_id'] = ""  # 最后一个分块没有后分块
                    logger.debug(f"分块 {chunk.metadata['chunk_id']} 是最后一个分块")

        # 验证链接构建结果
        non_empty_prev = sum(1 for chunk in all_chunks if chunk.metadata.get('prev_chunk_id'))
        non_empty_next = sum(1 for chunk in all_chunks if chunk.metadata.get('next_chunk_id'))

        logger.info(f"分块链接构建完成: {non_empty_prev} 个非空prev_chunk_id, {non_empty_next} 个非空next_chunk_id")
        logger.info(f"总共建立了 {total_links_created} 个链接关系")

        # 关键修复：确保返回修改后的对象
        return all_chunks

    def _create_document(self, dialog_item: Dict, turns: List[Dict],
                         dialog_index: int, chunk_index: int, split_type: str) -> Document:
        """创建LangChain文档 - 增强元数据版本"""
        content = self._build_content(dialog_item, turns)

        # 生成文档ID和分块ID
        document_id = f"doc_{dialog_index}"
        chunk_id = f"{document_id}_chunk_{chunk_index:03d}"

        # 计算证据相关指标
        evidence_metrics = self._calculate_evidence_metrics(turns, dialog_item)

        # 生成新元数据
        metadata = {
            # 基础标识
            "source": "wow_dataset",
            "document_id": document_id,
            "chunk_id": chunk_id,
            "chunk_seq": chunk_index,

            # 内容特征
            "content_hash": self._generate_content_hash(content),
            "content_type": self._detect_content_type(turns),
            "content_length": len(content),

            # 对话上下文
            "dialog_id": f"dialog_{dialog_index}",
            "dialog_index": dialog_index,
            "topic": dialog_item.get('chosen_topic', 'unknown'),
            "turn_count": len(turns),
            "speaker_role": self._detect_speaker_role(turns),

            # 证据和质量评估 - 新增5个元数据字段
            "wizard_eval_score": evidence_metrics["wizard_eval_score"],
            "has_checked_evidence": evidence_metrics["has_checked_evidence"],
            "total_retrieved_passages": evidence_metrics["total_retrieved_passages"],
            "total_retrieved_topics": evidence_metrics["total_retrieved_topics"],
            "evidence_density": evidence_metrics["evidence_density"],

            # 连贯性支持 - 初始化为空，将在_build_chunk_links中设置
            "prev_chunk_id": "",
            "next_chunk_id": "",

            # 分割信息
            "split_type": split_type,
            "original_turns": turns,
            "original_data": dialog_item
        }

        return Document(page_content=content, metadata=metadata)


class BatchEmbeddingProcessor:
    """批量嵌入处理器 - 优化版"""

    def __init__(self, embeddings, batch_size: int = 32):
        self.embeddings = embeddings
        self.batch_size = batch_size

    def batch_embed(self, texts: List[str]) -> np.ndarray:
        """批量嵌入文本 - 优化版"""
        if not texts:
            return np.array([])

        # 确保所有文本都是字符串
        cleaned_texts = []
        for text in texts:
            if isinstance(text, str):
                cleaned_texts.append(text)
            elif isinstance(text, dict):
                try:
                    if 'text' in text:
                        cleaned_texts.append(str(text['text']))
                    elif 'content' in text:
                        cleaned_texts.append(str(text['content']))
                    else:
                        cleaned_texts.append(json.dumps(text, ensure_ascii=False))
                except Exception:
                    cleaned_texts.append(str(text))
            else:
                cleaned_texts.append(str(text))

        all_embeddings = []
        for i in range(0, len(cleaned_texts), self.batch_size):
            batch_texts = cleaned_texts[i:i + self.batch_size]
            try:
                batch_embeddings = self.embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.warning(f"批量嵌入失败: {e}")
                embedding_dim = 768
                zero_embeddings = [np.zeros(embedding_dim) for _ in batch_texts]
                all_embeddings.extend(zero_embeddings)

        return np.array(all_embeddings)

    def batch_cosine_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """批量计算余弦相似度"""
        if len(embeddings1) == 0 or len(embeddings2) == 0:
            return np.array([])

        try:
            embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
            embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
            return np.dot(embeddings1_norm, embeddings2_norm.T)
        except Exception as e:
            logger.error(f"余弦相似度计算失败: {e}")
            return np.array([])


class WoWChunkQualityEvaluator:
    """WoW数据集分块质量综合评估器 - 最终优化版"""

    def __init__(self, embeddings, batch_size: int = 32):
        self.embeddings = embeddings
        self.batch_processor = BatchEmbeddingProcessor(embeddings, batch_size)

    def evaluate_chunks(self, original_docs: List[Document], chunk_docs: List[Document]) -> Dict[str, Any]:
        """综合评估分块质量"""
        logger.info("开始WoW分块质量综合评估...")

        try:
            # 基础统计
            basic_metrics = self._calculate_basic_metrics(chunk_docs)

            # 批量处理所有评估指标
            coherence_metrics, integrity_metrics, relevance_metrics, boundary_metrics, knowledge_metrics = self._batch_evaluate_all(
                original_docs, chunk_docs
            )

            # 合并所有指标
            all_metrics = {
                **basic_metrics,
                **coherence_metrics,
                **integrity_metrics,
                **relevance_metrics,
                **boundary_metrics,
                **knowledge_metrics
            }

            # 计算综合得分
            all_metrics['overall_score'] = self._calculate_overall_score(all_metrics)

            return all_metrics

        except Exception as e:
            logger.error(f"综合评估失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return {}

    def _batch_evaluate_all(self, original_docs: List[Document], chunk_docs: List[Document]) -> Tuple:
        """批量评估所有指标"""
        # 1. 语义连贯性评估 - 优化版
        coherence_metrics = self._batch_evaluate_semantic_coherence_optimized(chunk_docs)

        # 2. 信息完整性评估
        integrity_metrics = self._evaluate_information_integrity(original_docs, chunk_docs)

        # 3. 检索相关性评估 - 优化版
        relevance_metrics = self._batch_evaluate_retrieval_relevance_optimized(chunk_docs)

        # 4. 边界质量评估
        boundary_metrics = self._batch_evaluate_boundary_quality(chunk_docs)

        # 5. 知识质量评估
        knowledge_metrics = self._evaluate_knowledge_quality(chunk_docs)

        return coherence_metrics, integrity_metrics, relevance_metrics, boundary_metrics, knowledge_metrics

    def _batch_evaluate_semantic_coherence_optimized(self, chunk_docs: List[Document]) -> Dict[str, float]:
        """优化版：批量评估语义连贯性 - 大幅减少主题文本数量"""
        logger.info("批量评估语义连贯性(优化版)...")

        # 收集所有需要嵌入的文本
        all_dialog_flow_texts = []  # 对话流文本
        all_topic_texts = []  # 主题文本 - 优化后大幅减少
        all_knowledge_texts = []  # 知识文本

        # 记录每个块的文本索引
        chunk_flow_indices = []
        chunk_topic_indices = []
        chunk_knowledge_indices = []

        for chunk in chunk_docs:
            turns = chunk.metadata.get('original_turns', [])

            # 收集对话流文本
            flow_texts = self._extract_dialog_flow_pairs(turns)
            flow_start_idx = len(all_dialog_flow_texts)
            all_dialog_flow_texts.extend(flow_texts)
            chunk_flow_indices.append((flow_start_idx, len(all_dialog_flow_texts)))

            # 收集主题文本 - 优化：大幅减少数量
            topics = self._extract_topics_optimized(turns, chunk.metadata.get('topic', 'unknown'))
            topic_start_idx = len(all_topic_texts)
            all_topic_texts.extend(topics)
            chunk_topic_indices.append((topic_start_idx, len(all_topic_texts)))

            # 收集知识文本
            knowledge_sentences = self._extract_knowledge_sentences_optimized(turns)
            knowledge_start_idx = len(all_knowledge_texts)
            all_knowledge_texts.extend(knowledge_sentences)
            chunk_knowledge_indices.append((knowledge_start_idx, len(all_knowledge_texts)))

        # 批量嵌入所有文本
        logger.info(f"嵌入对话流文本: {len(all_dialog_flow_texts)} 条")
        flow_embeddings = self.batch_processor.batch_embed(all_dialog_flow_texts)

        logger.info(f"嵌入主题文本: {len(all_topic_texts)} 条 (优化后)")
        topic_embeddings = self.batch_processor.batch_embed(all_topic_texts)

        logger.info(f"嵌入知识文本: {len(all_knowledge_texts)} 条")
        knowledge_embeddings = self.batch_processor.batch_embed(all_knowledge_texts)

        # 批量计算指标
        return self._calculate_coherence_metrics_optimized(
            chunk_docs, flow_embeddings, topic_embeddings, knowledge_embeddings,
            chunk_flow_indices, chunk_topic_indices, chunk_knowledge_indices
        )

    def _extract_topics_optimized(self, turns: List[Dict], main_topic: str) -> List[str]:
        """优化版：提取主题 - 大幅减少数量并提高质量"""
        topics_set = set()

        # 1. 首先添加主主题
        if main_topic and main_topic != 'unknown':
            topics_set.add(main_topic)

        # 2. 从对话轮次中提取唯一主题 - 限制数量
        for turn in turns:
            turn_topics = turn.get('retrieved_topics', [])
            if turn_topics:
                if isinstance(turn_topics, list):
                    for topic in turn_topics[:2]:  # 每个轮次最多取2个主题
                        if isinstance(topic, str) and len(topic.strip()) > 5:  # 过滤短文本
                            clean_topic = topic.strip()
                            if len(clean_topic) < 100:  # 过滤过长的主题
                                topics_set.add(clean_topic)
                elif isinstance(turn_topics, str):
                    clean_topic = turn_topics.strip()
                    if len(clean_topic) > 5 and len(clean_topic) < 100:
                        topics_set.add(clean_topic)

            # 如果已经收集到足够主题，提前退出
            if len(topics_set) >= 3:  # 每个块最多3个主题
                break

        # 3. 如果没有提取到主题，使用主主题
        if not topics_set and main_topic and main_topic != 'unknown':
            topics_set.add(main_topic)

        return list(topics_set)

    def _extract_knowledge_sentences_optimized(self, turns: List[Dict]) -> List[str]:
        """优化版：提取知识句子"""
        knowledge_sentences = []
        for turn in turns:
            checked = turn.get('checked_sentence')
            if checked:
                if isinstance(checked, list):
                    for sentence in checked[:2]:  # 每个轮次最多取2个知识句子
                        if isinstance(sentence, str) and len(sentence.strip()) > 10:
                            knowledge_sentences.append(sentence.strip())
                elif isinstance(checked, str) and len(checked.strip()) > 10:
                    knowledge_sentences.append(checked.strip())
        return knowledge_sentences

    def _calculate_coherence_metrics_optimized(self, chunk_docs, flow_embeddings, topic_embeddings,
                                               knowledge_embeddings, flow_indices, topic_indices, knowledge_indices):
        """优化版：计算语义连贯性指标"""
        coherence_scores = []
        dialog_flow_scores = []
        topic_coherence_scores = []
        knowledge_coherence_scores = []

        for i, chunk in enumerate(chunk_docs):
            turns = chunk.metadata.get('original_turns', [])

            # 获取正确的嵌入切片
            flow_start, flow_end = flow_indices[i]
            topic_start, topic_end = topic_indices[i]
            knowledge_start, knowledge_end = knowledge_indices[i]

            chunk_flow_embeddings = flow_embeddings[flow_start:flow_end] if len(
                flow_embeddings) > flow_start else np.array([])
            chunk_topic_embeddings = topic_embeddings[topic_start:topic_end] if len(
                topic_embeddings) > topic_start else np.array([])
            chunk_knowledge_embeddings = knowledge_embeddings[knowledge_start:knowledge_end] if len(
                knowledge_embeddings) > knowledge_start else np.array([])

            # 1. 对话流连贯性
            flow_score = self._calculate_dialog_flow_coherence(turns, chunk_flow_embeddings)
            dialog_flow_scores.append(flow_score)

            # 2. 话题连贯性
            topic_score = self._calculate_topic_coherence_optimized(turns, chunk_topic_embeddings)
            topic_coherence_scores.append(topic_score)

            # 3. 知识连贯性
            knowledge_score = self._calculate_knowledge_coherence_optimized(turns, chunk_knowledge_embeddings)
            knowledge_coherence_scores.append(knowledge_score)

            # 综合语义连贯性
            overall_coherence = 0.5 * flow_score + 0.3 * topic_score + 0.2 * knowledge_score
            coherence_scores.append(overall_coherence)

        return {
            'semantic_coherence': np.mean(coherence_scores) if coherence_scores else 0.5,
            'dialog_flow_coherence': np.mean(dialog_flow_scores) if dialog_flow_scores else 0.5,
            'topic_coherence': np.mean(topic_coherence_scores) if topic_coherence_scores else 0.5,
            'knowledge_coherence': np.mean(knowledge_coherence_scores) if knowledge_coherence_scores else 0.5
        }

    def _extract_dialog_flow_pairs(self, turns: List[Dict]) -> List[str]:
        """提取对话流文本对"""
        flow_pairs = []
        for i in range(len(turns) - 1):
            pair_text = f"{turns[i]['speaker']}: {turns[i]['text']} → {turns[i + 1]['speaker']}: {turns[i + 1]['text']}"
            flow_pairs.append(pair_text)
        return flow_pairs

    def _calculate_dialog_flow_coherence(self, turns: List[Dict], flow_embeddings: np.ndarray) -> float:
        """计算对话流连贯性"""
        if len(turns) < 2:
            return 1.0

        if len(flow_embeddings) < 2:
            return 0.7

        try:
            adjacent_similarities = []
            for j in range(len(flow_embeddings) - 1):
                sim = cosine_similarity(
                    flow_embeddings[j].reshape(1, -1),
                    flow_embeddings[j + 1].reshape(1, -1)
                )[0][0]
                adjacent_similarities.append(sim)

            if not adjacent_similarities:
                return 0.5

            flow_coherence = np.mean(adjacent_similarities)
            speaker_coherence = self._calculate_speaker_coherence(turns)

            return 0.7 * flow_coherence + 0.3 * speaker_coherence

        except Exception as e:
            logger.warning(f"对话流连贯性计算失败: {e}")
            return 0.5

    def _calculate_speaker_coherence(self, turns: List[Dict]) -> float:
        """计算说话者交替连贯性"""
        if len(turns) < 2:
            return 1.0

        speaker_changes = 0
        for i in range(1, len(turns)):
            if turns[i]['speaker'] != turns[i - 1]['speaker']:
                speaker_changes += 1

        expected_changes = len(turns) - 1
        speaker_coherence = speaker_changes / expected_changes if expected_changes > 0 else 1.0

        return speaker_coherence

    def _calculate_topic_coherence_optimized(self, turns: List[Dict], topic_embeddings: np.ndarray) -> float:
        """优化版：计算话题连贯性"""
        if len(topic_embeddings) < 2:
            return 1.0 if len(topic_embeddings) == 1 else 0.5

        try:
            similarities = []
            for i in range(len(topic_embeddings)):
                for j in range(i + 1, len(topic_embeddings)):
                    sim = cosine_similarity(
                        topic_embeddings[i].reshape(1, -1),
                        topic_embeddings[j].reshape(1, -1)
                    )[0][0]
                    similarities.append(sim)

            return np.mean(similarities) if similarities else 0.5

        except Exception:
            return 0.5

    def _calculate_knowledge_coherence_optimized(self, turns: List[Dict], knowledge_embeddings: np.ndarray) -> float:
        """优化版：计算知识连贯性"""
        if len(knowledge_embeddings) < 2:
            return 0.5

        try:
            similarities = []
            for i in range(len(knowledge_embeddings)):
                for j in range(i + 1, len(knowledge_embeddings)):
                    sim = cosine_similarity(
                        knowledge_embeddings[i].reshape(1, -1),
                        knowledge_embeddings[j].reshape(1, -1)
                    )[0][0]
                    similarities.append(sim)

            return np.mean(similarities) if similarities else 0.5

        except Exception:
            return 0.5

    def _batch_evaluate_retrieval_relevance_optimized(self, chunk_docs: List[Document]) -> Dict[str, float]:
        """优化版：批量评估检索相关性"""
        logger.info("批量评估检索相关性(优化版)...")

        relevance_scores = []
        rule_based_scores = []
        knowledge_density_scores = []
        topic_focus_scores = []

        # 收集所有主题用于批量处理 - 使用优化版提取
        all_topics = []
        chunk_topic_indices = []

        for chunk in chunk_docs:
            turns = chunk.metadata.get('original_turns', [])
            topics = self._extract_topics_optimized(turns, chunk.metadata.get('topic', 'unknown'))
            start_idx = len(all_topics)
            all_topics.extend(topics)
            chunk_topic_indices.append((start_idx, len(all_topics)))

            # 基于规则的评分
            rule_score = self._rule_based_relevance_score(chunk)
            rule_based_scores.append(rule_score)

            # 知识密度评分
            density_score = self._knowledge_density_score(chunk)
            knowledge_density_scores.append(density_score)

        # 批量嵌入所有主题 - 现在数量大幅减少
        topic_embeddings = self.batch_processor.batch_embed(all_topics)

        # 批量计算主题聚焦度
        for i, chunk in enumerate(chunk_docs):
            start_idx, end_idx = chunk_topic_indices[i]
            if len(topic_embeddings) > start_idx:
                chunk_topic_embeddings = topic_embeddings[start_idx:end_idx]
                focus_score = self._calculate_topic_focus(chunk_topic_embeddings)
            else:
                focus_score = 0.5
            topic_focus_scores.append(focus_score)

            # 综合相关性
            relevance_score = 0.4 * rule_based_scores[i] + 0.3 * knowledge_density_scores[i] + 0.3 * focus_score
            relevance_scores.append(relevance_score)

        return {
            'retrieval_relevance': np.mean(relevance_scores) if relevance_scores else 0.0,
            'rule_based_relevance': np.mean(rule_based_scores) if rule_based_scores else 0.0,
            'knowledge_density': np.mean(knowledge_density_scores) if knowledge_density_scores else 0.0,
            'topic_focus': np.mean(topic_focus_scores) if topic_focus_scores else 0.0
        }

    def _batch_evaluate_boundary_quality(self, chunk_docs: List[Document]) -> Dict[str, float]:
        """批量评估边界质量"""
        logger.info("批量评估边界质量...")

        boundary_scores = []

        # 按对话分组
        chunks_by_dialog = {}
        for chunk in chunk_docs:
            dialog_id = chunk.metadata['dialog_id']
            if dialog_id not in chunks_by_dialog:
                chunks_by_dialog[dialog_id] = []
            chunks_by_dialog[dialog_id].append(chunk)

        # 收集所有需要评估边界的块内容
        boundary_texts = []
        boundary_groups = []

        for chunks in chunks_by_dialog.values():
            if len(chunks) > 1:
                group_texts = [chunk.page_content for chunk in chunks]
                boundary_texts.extend(group_texts)
                boundary_groups.append(len(group_texts))

        if not boundary_texts:
            return {'boundary_quality': 0.5}

        # 批量嵌入所有边界文本
        boundary_embeddings = self.batch_processor.batch_embed(boundary_texts)

        # 批量计算边界质量
        current_idx = 0
        for group_size in boundary_groups:
            if len(boundary_embeddings) > current_idx:
                group_embeddings = boundary_embeddings[current_idx:current_idx + group_size]
                current_idx += group_size
                score = self._calculate_boundary_score_batch(group_embeddings)
                boundary_scores.append(score)
            else:
                boundary_scores.append(0.5)

        return {
            'boundary_quality': np.mean(boundary_scores) if boundary_scores else 0.5
        }

    def _calculate_boundary_score_batch(self, embeddings: np.ndarray) -> float:
        """批量计算边界质量得分"""
        try:
            adjacent_similarities = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[i + 1].reshape(1, -1)
                )[0][0]
                adjacent_similarities.append(sim)

            if not adjacent_similarities:
                return 0.5

            avg_adjacent_sim = np.mean(adjacent_similarities)
            if 0.3 <= avg_adjacent_sim <= 0.7:
                boundary_score = 1.0 - abs(avg_adjacent_sim - 0.5) * 2
            else:
                boundary_score = 0.2

            return max(0.0, min(1.0, boundary_score))

        except Exception:
            return 0.5

    def _calculate_topic_focus(self, topic_embeddings: np.ndarray) -> float:
        """计算主题聚焦度"""
        if len(topic_embeddings) < 2:
            return 1.0

        try:
            centroid = np.mean(topic_embeddings, axis=0)
            similarities = []
            for emb in topic_embeddings:
                sim = cosine_similarity(emb.reshape(1, -1), centroid.reshape(1, -1))[0][0]
                similarities.append(sim)

            return np.mean(similarities) if similarities else 0.5

        except Exception:
            return 0.5

    def _calculate_basic_metrics(self, chunk_docs: List[Document]) -> Dict[str, Any]:
        """计算基础统计指标"""
        chunk_sizes = [len(doc.page_content) for doc in chunk_docs]
        turn_counts = [doc.metadata.get('turn_count', 0) for doc in chunk_docs]

        chunks_by_dialog = {}
        for chunk in chunk_docs:
            dialog_id = chunk.metadata['dialog_id']
            if dialog_id not in chunks_by_dialog:
                chunks_by_dialog[dialog_id] = []
            chunks_by_dialog[dialog_id].append(chunk)

        total_chunks = len(chunk_docs)
        in_target_range = sum(1 for size in chunk_sizes if 200 <= size <= 1600)

        return {
            'total_chunks': total_chunks,
            'total_dialogs': len(chunks_by_dialog),
            'chunks_per_dialog': total_chunks / len(chunks_by_dialog) if chunks_by_dialog else 0,
            'avg_chunk_size': np.mean(chunk_sizes) if chunk_sizes else 0,
            'min_chunk_size': min(chunk_sizes) if chunk_sizes else 0,
            'max_chunk_size': max(chunk_sizes) if chunk_sizes else 0,
            'in_target_range': in_target_range,
            'below_target': sum(1 for size in chunk_sizes if size < 200),
            'above_target': sum(1 for size in chunk_sizes if size > 1600),
            'in_target_range_pct': (in_target_range / total_chunks) * 100 if total_chunks > 0 else 0,
            'avg_turns_per_chunk': np.mean(turn_counts) if turn_counts else 0,
            'single_turn_chunks': sum(1 for count in turn_counts if count == 1),
            'split_dialogs': sum(1 for chunks in chunks_by_dialog.values() if len(chunks) > 1),
            'complete_dialogs': sum(1 for chunks in chunks_by_dialog.values() if len(chunks) == 1),
            'split_types': Counter(doc.metadata.get('split_type', 'unknown') for doc in chunk_docs)
        }

    def _evaluate_information_integrity(self, original_docs: List[Document], chunk_docs: List[Document]) -> Dict[
        str, float]:
        """评估信息完整性"""
        integrity_scores = []
        qa_integrity_scores = []
        knowledge_integrity_scores = []

        chunks_by_dialog = {}
        for chunk in chunk_docs:
            dialog_id = chunk.metadata['dialog_id']
            if dialog_id not in chunks_by_dialog:
                chunks_by_dialog[dialog_id] = []
            chunks_by_dialog[dialog_id].append(chunk)

        for dialog_id, chunks in chunks_by_dialog.items():
            original_doc = next((doc for doc in original_docs if doc.metadata.get('dialog_id') == dialog_id), None)
            if not original_doc:
                continue

            original_turns = original_doc.metadata.get('original_turns', [])
            original_qa_pairs = self._count_qa_pairs(original_turns)
            retained_qa_pairs = sum(self._count_qa_pairs(chunk.metadata.get('original_turns', [])) for chunk in chunks)
            qa_retention = retained_qa_pairs / original_qa_pairs if original_qa_pairs > 0 else 1.0
            qa_integrity_scores.append(qa_retention)

            original_knowledge = self._count_knowledge_evidence(original_turns)
            retained_knowledge = sum(
                self._count_knowledge_evidence(chunk.metadata.get('original_turns', [])) for chunk in chunks)
            knowledge_retention = retained_knowledge / original_knowledge if original_knowledge > 0 else 1.0
            knowledge_integrity_scores.append(knowledge_retention)

            integrity_score = 0.6 * qa_retention + 0.4 * knowledge_retention
            integrity_scores.append(integrity_score)

        return {
            'information_integrity': np.mean(integrity_scores) if integrity_scores else 1.0,
            'qa_integrity': np.mean(qa_integrity_scores) if qa_integrity_scores else 1.0,
            'knowledge_integrity': np.mean(knowledge_integrity_scores) if knowledge_integrity_scores else 1.0
        }

    def _evaluate_knowledge_quality(self, chunk_docs: List[Document]) -> Dict[str, float]:
        """评估知识质量"""
        knowledge_scores = []
        evidence_quality_scores = []
        retrieval_quality_scores = []

        for chunk in chunk_docs:
            turns = chunk.metadata.get('original_turns', [])
            evidence_score = self._evaluate_evidence_quality(turns)
            evidence_quality_scores.append(evidence_score)
            retrieval_score = self._evaluate_retrieval_quality(turns)
            retrieval_quality_scores.append(retrieval_score)
            knowledge_score = 0.6 * evidence_score + 0.4 * retrieval_score
            knowledge_scores.append(knowledge_score)

        return {
            'knowledge_quality': np.mean(knowledge_scores) if knowledge_scores else 0.5,
            'evidence_quality': np.mean(evidence_quality_scores) if evidence_quality_scores else 0.5,
            'retrieval_quality': np.mean(retrieval_quality_scores) if retrieval_quality_scores else 0.5
        }

    def _rule_based_relevance_score(self, chunk: Document) -> float:
        """基于规则的检索相关性评分"""
        content = chunk.page_content
        turns = chunk.metadata.get('original_turns', [])

        score = 0.0
        qa_pairs = self._count_qa_pairs(turns)
        if qa_pairs > 0:
            score += 0.4

        knowledge_count = sum(1 for turn in turns if turn.get('checked_sentence'))
        if knowledge_count > 0:
            score += 0.3

        if "Topic:" in content:
            score += 0.2

        content_length = len(content)
        if 200 <= content_length <= 1600:
            score += 0.1

        return min(1.0, score)

    def _knowledge_density_score(self, chunk: Document) -> float:
        """知识密度评分"""
        turns = chunk.metadata.get('original_turns', [])
        content = chunk.page_content

        if not content or len(turns) == 0:
            return 0.0

        knowledge_turns = sum(1 for turn in turns if turn.get('checked_sentence'))
        total_turns = len(turns)
        knowledge_density = knowledge_turns / total_turns

        retrieved_count = sum(1 for turn in turns if turn.get('retrieved_passages'))
        retrieved_density = retrieved_count / total_turns

        return 0.6 * knowledge_density + 0.4 * retrieved_density

    def _evaluate_evidence_quality(self, turns: List[Dict]) -> float:
        """评估知识证据质量"""
        evidence_turns = [turn for turn in turns if turn.get('checked_sentence')]

        if not evidence_turns:
            return 0.3

        total_evidence = sum(len(turn.get('checked_sentence', [])) for turn in evidence_turns)
        evidence_density = total_evidence / len(turns) if turns else 0

        return min(1.0, evidence_density * 2)

    def _evaluate_retrieval_quality(self, turns: List[Dict]) -> float:
        """评估检索质量"""
        retrieval_turns = [turn for turn in turns if turn.get('retrieved_passages')]

        if not retrieval_turns:
            return 0.3

        total_retrieval = sum(len(turn.get('retrieved_passages', [])) for turn in retrieval_turns)
        retrieval_density = total_retrieval / len(turns) if turns else 0

        return min(1.0, retrieval_density * 0.5)

    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """计算综合质量得分"""
        weights = {
            'semantic_coherence': 0.25,
            'information_integrity': 0.20,
            'retrieval_relevance': 0.20,
            'boundary_quality': 0.15,
            'knowledge_quality': 0.20
        }

        overall_score = 0.0
        for metric, weight in weights.items():
            overall_score += metrics.get(metric, 0) * weight

        return max(0.0, min(1.0, overall_score))

    # 辅助方法
    def _count_qa_pairs(self, turns: List[Dict]) -> int:
        """计算问答对数量"""
        count = 0
        for i in range(len(turns) - 1):
            if (turns[i]['speaker'] == "0_Wizard" and
                    turns[i + 1]['speaker'] == "1_Apprentice"):
                count += 1
        return count

    def _count_knowledge_evidence(self, turns: List[Dict]) -> int:
        """计算知识证据数量"""
        count = 0
        for turn in turns:
            if turn.get('checked_sentence'):
                checked = turn['checked_sentence']
                if isinstance(checked, list):
                    count += len(checked)
                else:
                    count += 1
        return count

    def generate_comprehensive_report(self, metrics: Dict[str, Any]) -> str:
        """生成综合评估报告"""
        report_lines = [
            "=" * 80,
            "WoW数据集分块 - 综合质量评估报告(最终优化版)",
            "=" * 80,
            f"总对话数: {metrics.get('total_dialogs', 0)}",
            f"总块数: {metrics.get('total_chunks', 0)}",
            f"分割率: {metrics.get('chunks_per_dialog', 0):.2f} 块/对话",
            "",
            "🎯 核心质量指标:",
            f"  语义连贯性: {metrics.get('semantic_coherence', 0):.3f}",
            f"  信息完整性: {metrics.get('information_integrity', 0):.3f}",
            f"  检索相关性: {metrics.get('retrieval_relevance', 0):.3f}",
            f"  边界质量: {metrics.get('boundary_quality', 0):.3f}",
            f"  知识质量: {metrics.get('knowledge_quality', 0):.3f}",
            "",
            "📊 详细分析:",
            f"  对话流连贯性: {metrics.get('dialog_flow_coherence', 0):.3f}",
            f"  话题连贯性: {metrics.get('topic_coherence', 0):.3f}",
            f"  知识连贯性: {metrics.get('knowledge_coherence', 0):.3f}",
            f"  问答完整性: {metrics.get('qa_integrity', 0):.3f}",
            f"  知识完整性: {metrics.get('knowledge_integrity', 0):.3f}",
            f"  知识密度: {metrics.get('knowledge_density', 0):.3f}",
            f"  主题聚焦度: {metrics.get('topic_focus', 0):.3f}",
            "",
            "📈 大小分布:",
            f"  平均大小: {metrics.get('avg_chunk_size', 0):.0f} 字符",
            f"  大小范围: {metrics.get('min_chunk_size', 0)} - {metrics.get('max_chunk_size', 0)}",
            f"  目标范围(200-1600): {metrics.get('in_target_range_pct', 0):.1f}%",
            "",
            "🔄 对话结构:",
            f"  平均轮次/块: {metrics.get('avg_turns_per_chunk', 0):.1f}",
            f"  被分割对话: {metrics.get('split_dialogs', 0)}",
            f"  完整对话: {metrics.get('complete_dialogs', 0)}",
            "",
            f"🏆 综合质量得分: {metrics.get('overall_score', 0):.3f}",
            "=" * 80
        ]

        return "\n".join(report_lines)


def load_wow_data(data_path: str) -> List[Document]:
    """加载WoW数据为LangChain文档"""
    logger.info(f"加载数据: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        wow_data = json.load(f)

    documents = []
    for i, dialog_item in enumerate(wow_data):
        dialog_turns = dialog_item.get('dialog', [])
        content_parts = [f"Topic: {dialog_item.get('chosen_topic', 'Unknown')}", ""]

        for turn in dialog_turns:
            speaker = "Wizard" if turn['speaker'] == "0_Wizard" else "Apprentice"
            content_parts.append(f"{speaker}: {turn['text']}")

        content = "\n".join(content_parts)

        doc = Document(
            page_content=content,
            metadata={
                "dialog_id": f"dialog_{i}",
                "dialog_index": i,
                "original_data": dialog_item,
                "original_turns": dialog_turns
            }
        )
        documents.append(doc)

    logger.info(f"成功加载 {len(documents)} 条WoW数据")
    return documents


def main():
    """主函数"""
    # 模型路径
    model_path = str(paths.models / "chunk_model" / "all-mpnet-base-v2")
    print(model_path)

    # 数据路径
    data_path = paths.raw_data / "train_100.json"
    print(data_path)

    try:
        # 1. 加载数据
        original_docs = load_wow_data(data_path)

        # 2. 分块处理
        logger.info("开始最终优化分块处理...")
        splitter = WoWTextSplitter(max_chunk_size=1600, min_chunk_size=200)
        chunk_docs = splitter.split_documents(original_docs)

        # 统计分割情况和元数据
        chunk_ids = [doc.metadata.get('chunk_id', 'unknown') for doc in chunk_docs]
        chunk_id_counter = Counter(chunk_ids)

        content_types = [doc.metadata.get('content_type', 'unknown') for doc in chunk_docs]
        content_type_counter = Counter(content_types)

        # 检查prev_chunk_id和next_chunk_id设置情况
        prev_chunk_ids = [doc.metadata.get('prev_chunk_id', '') for doc in chunk_docs]
        next_chunk_ids = [doc.metadata.get('next_chunk_id', '') for doc in chunk_docs]
        non_empty_prev = sum(1 for pid in prev_chunk_ids if pid)
        non_empty_next = sum(1 for nid in next_chunk_ids if nid)

        logger.info(f"分块完成: {len(chunk_docs)} 个块")
        logger.info(f"chunk_id分布: {dict(chunk_id_counter)}")
        logger.info(f"content_type分布: {dict(content_type_counter)}")
        logger.info(f"分块链接统计: {non_empty_prev} 个非空prev_chunk_id, {non_empty_next} 个非空next_chunk_id")

        # 详细验证链接关系
        logger.info("详细链接关系验证:")
        for i, chunk in enumerate(chunk_docs[:10]):  # 检查前10个分块
            logger.info(
                f"分块 {chunk.metadata['chunk_id']}: prev={chunk.metadata['prev_chunk_id']}, next={chunk.metadata['next_chunk_id']}")

        # 3. 初始化嵌入模型
        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32
            }
        )

        # 4. 质量评估（使用最终优化版评估器）
        evaluator = WoWChunkQualityEvaluator(embeddings, batch_size=32)
        metrics = evaluator.evaluate_chunks(original_docs, chunk_docs)
        report = evaluator.generate_comprehensive_report(metrics)

        print("\n" + report)
        logger.info("综合质量评估报告已生成")

        # 5. 保存结果
        output_data = []
        for doc in chunk_docs:
            output_data.append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })
        # 处理后数据路径
        output_path = paths.processed_data_02 / "wow_langchain_chunks_final.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"分块结果已保存到: {output_path}")

        # 6. 质量评级（评级维度是什么？）
        overall_score = metrics.get('overall_score', 0)
        semantic_coherence = metrics.get('semantic_coherence', 0)

        logger.info(f"🏆 综合质量得分: {overall_score:.3f}")
        logger.info(f"🔄 语义连贯性: {semantic_coherence:.3f}")

        if overall_score >= 0.8:
            logger.info("🎉 分块质量优秀!")
        elif overall_score >= 0.7:
            logger.info("👍 分块质量良好")
        elif overall_score >= 0.6:
            logger.info("⚠️ 分块质量一般，需要改进")
        else:
            logger.warning("❌ 分块质量较差，需要大幅改进")

        return chunk_docs, metrics

    except Exception as e:
        logger.error(f"处理失败: {e}")
        import traceback
        logger.error(f"详细错误: {traceback.format_exc()}")
        return [], {}


if __name__ == "__main__":
    documents, metrics = main()


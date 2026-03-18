from config.paths import paths
import json
import re
import logging
import time
import os
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from rake_nltk import Rake
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WoWKeywordExtractor:
    """WoW数据集关键词提取器 - 完整优化版本"""

    def __init__(self,
                 model_path: str,
                 batch_size: int = 4,
                 max_workers: int = 2,
                 min_keywords: int = 8,
                 max_keywords: int = 25):
        """
        初始化关键词提取器

        Args:
            model_path: 本地模型路径
            batch_size: 批量处理大小
            max_workers: 最大工作线程数
            min_keywords: 最小关键词数量
            max_keywords: 最大关键词数量
        """
        # 配置参数
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.min_keywords = min_keywords
        self.max_keywords = max_keywords

        # 初始化模型
        logger.info("初始化关键词提取模型...")
        self.model_path = model_path
        self.embedding_model, self.keybert_model = self._initialize_models(model_path)

        # 初始化RAKE
        self.rake = Rake(
            max_length=3,
            min_length=1,
            include_repeated_phrases=False
        )

        # WoW专用停用词
        self.wow_stop_words = {'wizard', 'apprentice', 'hello', 'hi', 'hey', 'ok', 'okay', 'yes', 'no', 'well', 'so',
                               'like'}

        logger.info(f"关键词提取器初始化完成 - 关键词范围: {min_keywords}-{max_keywords}")

    def _initialize_models(self, model_path: str) -> Tuple[SentenceTransformer, KeyBERT]:
        """初始化嵌入模型和KeyBERT"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型路径不存在: {model_path}")

            logger.info(f"加载本地模型: {model_path}")
            embedding_model = SentenceTransformer(model_path)
            keybert_model = KeyBERT(model=embedding_model)
            logger.info("模型加载成功")
            return embedding_model, keybert_model

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def preprocess_content(self, content: str) -> str:
        """
        预处理WoW内容，过滤角色对话

        Args:
            content: 原始内容

        Returns:
            预处理后的内容
        """
        if not content:
            return ""

        lines = content.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 过滤以角色开头的行（首字母大写的Wizard和Apprentice）
            if re.match(r'^(Wizard|Apprentice):', line):
                # 检查是否有实际内容，如果有则保留内容部分
                if ':' in line and len(line.split(':', 1)[1].strip()) > 5:
                    content_part = line.split(':', 1)[1].strip()
                    cleaned_lines.append(content_part)
                continue

            # 保留其他内容
            cleaned_lines.append(line)

        cleaned_content = ' '.join(cleaned_lines)

        # 额外的清理：移除多余的空格
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()

        return cleaned_content

    def _build_combined_text(self, document: Dict[str, Any]) -> str:
        """构建综合文本，利用多源信息"""
        texts = []

        # 1. 主要内容（最高权重）
        content = document.get('content', '')
        preprocessed_content = self.preprocess_content(content)
        if preprocessed_content:
            texts.append(preprocessed_content)

        # 2. 对话历史（中等权重）
        original_turns = document.get('original_turns', [])
        if original_turns:
            # 取最近5轮对话（排除当前轮次）
            recent_turns = original_turns[-6:-1] if len(original_turns) > 5 else original_turns[:-1]
            for turn in recent_turns:
                if turn.get('text'):
                    cleaned_turn = self.preprocess_content(turn['text'])
                    if cleaned_turn:
                        texts.append(cleaned_turn)

        # 3. 主题信息（高权重）
        original_data = document.get('original_data', {})
        if original_data.get('chosen_topic'):
            texts.append(original_data['chosen_topic'])
        if original_data.get('chosen_topic_passage'):
            texts.append(original_data['chosen_topic_passage'])

        # 4. 检索段落（较低权重，但提供上下文）
        for turn in original_turns[-2:]:  # 最近2轮
            if turn.get('retrieved_passages'):
                for passage in turn['retrieved_passages'][:2]:  # 前2个段落
                    if passage and len(passage.strip()) > 10:
                        texts.append(passage.strip())

        return " ".join(texts)

    def _is_role_entity(self, entity: Dict[str, Any]) -> bool:
        """判断实体是否为角色实体"""
        try:
            entity_text = entity.get("text", "")
            if not entity_text:
                return False

            # 只对首字母大写的"Wizard"和"Apprentice"进行判断
            if entity_text not in ["Wizard", "Apprentice"]:
                return False

            # 检查角色标记字段
            is_role = entity.get("is_role", False)
            role_type = entity.get("role_type", "")

            # 只有当明确标记为角色时才认为是角色实体
            return is_role is True and role_type in ["assistant", "user"]
        except Exception as e:
            logger.warning(f"检查角色实体时出错: {entity}, 错误: {e}")
            return False

    def calculate_dynamic_top_k_enhanced(self, document: Dict[str, Any], candidate_count: int) -> int:
        """
        增强的动态Top-K计算，考虑多源信息
        """
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        original_turns = document.get('original_turns', [])
        original_data = document.get('original_data', {})

        word_count = len(content.split())

        # 1. 基础长度因素
        if word_count <= 150:
            base_k = 8
        elif word_count <= 300:
            base_k = 12
        elif word_count <= 500:
            base_k = 15
        elif word_count <= 800:
            base_k = 18
        else:
            base_k = 20

        # 2. 信息源丰富度加成
        richness_bonus = 0

        # 对话轮次丰富度
        if original_turns and len(original_turns) > 3:
            richness_bonus += min(3, (len(original_turns) - 3) // 2)

        # 检索段落丰富度
        retrieved_count = 0
        for turn in original_turns[-3:]:  # 最近3轮
            if turn.get('retrieved_passages'):
                retrieved_count += len(turn['retrieved_passages'])
        if retrieved_count > 2:
            richness_bonus += min(3, retrieved_count // 2)

        # 主题信息丰富度
        if original_data.get('chosen_topic') and original_data.get('chosen_topic_passage'):
            richness_bonus += 2

        # 实体密度加成
        if metadata.get('ner_entities'):
            entity_count = len(metadata['ner_entities'])
            # 过滤角色实体
            non_role_entities = [e for e in metadata['ner_entities']
                                 if not self._is_role_entity(e)]
            entity_density = len(non_role_entities) / word_count if word_count > 0 else 0
            if entity_density > 0.05:
                richness_bonus += min(3, int(entity_density * 100))

        # 关系密度加成
        if metadata.get('relations'):
            relation_count = len(metadata['relations'])
            if relation_count > 2:
                richness_bonus += min(2, relation_count // 2)

        # 综合计算
        comprehensive_k = base_k + richness_bonus

        # 基于候选词数量调整
        final_k = min(comprehensive_k, candidate_count)

        # 确保在合理范围内
        final_k = max(self.min_keywords, min(self.max_keywords, final_k))

        chunk_id = document.get('metadata', {}).get('chunk_id', 'unknown')
        logger.info(f"文档 {chunk_id}: 词数={word_count}, 基础K={base_k}, 加成={richness_bonus}, 最终K={final_k}")

        return final_k

    def extract_with_rake_enhanced(self, document: Dict[str, Any], top_n: int = 35) -> List[Tuple[str, float]]:
        """
        增强的RAKE提取，利用多源信息
        """
        # 构建综合文本
        combined_text = self._build_combined_text(document)

        if not combined_text.strip():
            return []

        try:
            self.rake.extract_keywords_from_text(combined_text)
            phrases_with_scores = self.rake.get_ranked_phrases_with_scores()

            # 扩展停用词列表，包含常见标识符
            extended_stop_words = self.wow_stop_words.union({
                'topic', 'topics', 'metadata', 'chunk', 'chunk_id', 'content',
                'original_data', 'original_turns', 'retrieved_passages',
                'chosen_topic', 'chosen_topic_passage', 'text', 'score',
                'frequency', 'positions', 'normalized_text', 'source', 'ngram'
            })

            # 过滤低分短语和停用词
            filtered_phrases = []
            for score, phrase in phrases_with_scores:
                phrase_lower = phrase.lower()
                words = phrase.split()

                # 增强过滤条件
                if (len(words) < 1 or len(words) > 4 or
                        any(stop_word in phrase_lower for stop_word in extended_stop_words) or
                        score < 1.0 or
                        self._is_metadata_identifier(phrase)):  # 新增标识符检查
                    continue

                filtered_phrases.append((phrase, score))

            return filtered_phrases[:top_n]

        except Exception as e:
            logger.error(f"RAKE提取失败: {e}")
            return []

    def _is_metadata_identifier(self, phrase: str) -> bool:
        """
        判断短语是否为元数据标识符或系统字段
        """
        # 常见系统标识符和字段名
        metadata_identifiers = {
            'topic', 'topics', 'metadata', 'chunk', 'chunk_id', 'content',
            'original_data', 'original_turns', 'retrieved_passages', 'passages',
            'chosen_topic', 'chosen_topic_passage', 'text', 'score', 'frequency',
            'positions', 'normalized_text', 'source', 'ngram', 'entity', 'entities',
            'relation', 'relations', 'role_type', 'is_role', 'start', 'end',
            'total_keywords', 'avg_confidence', 'keyword_density', 'source_breakdown',
            'extraction_info', 'top_k_used', 'candidate_count', 'source_used'
        }

        phrase_lower = phrase.lower()

        # 检查是否在已知标识符列表中
        if phrase_lower in metadata_identifiers:
            return True

        # 检查常见的标识符模式
        identifier_patterns = [
            r'^[a-z_]+$',  # 纯小写加下划线
            r'^[a-z]+_[a-z]+(_[a-z]+)*$',  # 多段下划线连接
            r'.*[_-]id$',  # 以_id或-id结尾
            r'.*[_-]type$',  # 以_type或-type结尾
            r'.*[_-]count$',  # 以_count或-count结尾
            r'^\d+$',  # 纯数字
        ]

        for pattern in identifier_patterns:
            if re.match(pattern, phrase_lower):
                return True

        return False

    def extract_with_keybert(self, text: str, candidates: List[str], top_k: int) -> List[Dict[str, Any]]:
        """
        使用KeyBERT对候选关键词进行重排（使用优化参数）

        Args:
            text: 输入文本
            candidates: 候选关键词列表
            top_k: 返回的关键词数量

        Returns:
            KeyBERT重排后的关键词列表
        """
        if not text.strip() or not candidates:
            return []

        try:
            # 使用优化后的KeyBERT参数
            keywords = self.keybert_model.extract_keywords(
                text,
                candidates=candidates,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',  # 使用英语停用词
                top_n=top_k,
                use_mmr=True,  # 使用MMR算法
                diversity=0.4,  # 优化后的多样性参数
                use_maxsum=False
            )

            # 转换为标准格式
            keybert_results = []
            for keyword, confidence in keywords:
                keybert_results.append({
                    'text': keyword,
                    'score': float(confidence)
                })

            logger.debug(
                f"KeyBERT提取完成: {len(keybert_results)} 个关键词，最高置信度: {max([r['score'] for r in keybert_results]) if keybert_results else 0:.4f}")

            return keybert_results

        except Exception as e:
            logger.error(f"KeyBERT提取失败: {e}")
            return []

    def calculate_frequency_and_positions(self, search_content: str, keywords: List[Dict]) -> List[Dict]:
        """
        计算关键词频率和位置（改进版本）
        """
        enhanced_keywords = []

        for keyword in keywords:
            text = keyword['text']

            # 改进的匹配逻辑：支持模糊匹配和边界检测
            positions = []

            # 方法1：精确匹配（首选）
            exact_pattern = re.escape(text)
            for match in re.finditer(exact_pattern, search_content, re.IGNORECASE):
                positions.append({
                    'start': match.start(),
                    'end': match.end()
                })

            # 方法2：如果精确匹配失败，尝试单词边界匹配
            if not positions:
                word_boundary_pattern = r'\b' + re.escape(text) + r'\b'
                for match in re.finditer(word_boundary_pattern, search_content, re.IGNORECASE):
                    positions.append({
                        'start': match.start(),
                        'end': match.end()
                    })

            # 创建增强的关键词信息
            enhanced_keyword = {
                'text': text,
                'normalized_text': text.lower().replace(' ', '_'),
                'frequency': len(positions),
                'positions': positions,
                'score': keyword['score'],
                'source': 'keybert',
                'ngram': len(text.split())
            }

            enhanced_keywords.append(enhanced_keyword)

        return enhanced_keywords

    def calculate_keyword_stats(self, keywords: List[Dict], content_length: int) -> Dict[str, Any]:
        """
        计算关键词统计信息

        Args:
            keywords: 关键词列表
            content_length: 文档长度

        Returns:
            关键词统计信息
        """
        if not keywords:
            return {
                'total_keywords': 0,
                'avg_confidence': 0.0,
                'keyword_density': 0.0,
                'source_breakdown': {'keybert': 0}
            }

        # 计算平均置信度
        scores = [kw['score'] for kw in keywords]
        avg_confidence = np.mean(scores)

        # 计算关键词密度
        keyword_density = len(keywords) / content_length if content_length > 0 else 0

        # 来源分布
        source_breakdown = defaultdict(int)
        for kw in keywords:
            source_breakdown[kw['source']] += 1

        return {
            'total_keywords': len(keywords),
            'avg_confidence': round(avg_confidence, 4),
            'keyword_density': round(keyword_density, 4),
            'source_breakdown': dict(source_breakdown)
        }

    def extract_keywords_single(self, document: Dict[str, Any]) -> Dict[str, Any]:
        try:
            chunk_id = document.get('metadata', {}).get('chunk_id', 'unknown')
            content = document.get('content', '')
            content_length = document.get('metadata', {}).get('content_length', len(content.split()))

            logger.info(f"开始处理文档: {chunk_id}")

            # 1. 使用增强的RAKE提取（基于多源信息）
            rake_candidates = self.extract_with_rake_enhanced(document, top_n=35)
            candidate_phrases = [candidate[0] for candidate in rake_candidates]

            # 过滤标识符候选词
            filtered_candidates = []
            for phrase in candidate_phrases:
                if not self._is_metadata_identifier(phrase):
                    filtered_candidates.append(phrase)

            candidate_phrases = filtered_candidates

            if not candidate_phrases:
                logger.warning(f"文档 {chunk_id} 未提取到候选关键词")
                return self.create_empty_result(chunk_id, content_length)

            # 2. 增强的动态Top-K计算
            top_k = self.calculate_dynamic_top_k_enhanced(document, len(candidate_phrases))

            # 3. KeyBERT重排（基于预处理内容）
            preprocessed_content = self.preprocess_content(content)
            keybert_keywords = self.extract_with_keybert(preprocessed_content, candidate_phrases, top_k)

            if not keybert_keywords:
                logger.warning(f"文档 {chunk_id} KeyBERT重排失败")
                return self.create_empty_result(chunk_id, content_length)

            # 4. 修复：在KeyBERT使用的同一文本中计算频率和位置
            enhanced_keywords = self.calculate_frequency_and_positions(preprocessed_content, keybert_keywords)

            # 5. 进一步过滤：移除frequency=0的关键词（可选）
            final_keywords = [kw for kw in enhanced_keywords if kw['frequency'] > 0]

            # 6. 计算统计信息
            keyword_stats = self.calculate_keyword_stats(final_keywords, content_length)

            # 7. 构建输出
            result = {
                'chunk_id': chunk_id,
                'content_length': content_length,
                'keywords': final_keywords,  # 使用过滤后的关键词
                'keyword_stats': keyword_stats,
                'extraction_info': {
                    'top_k_used': top_k,
                    'candidate_count': len(candidate_phrases),
                    'filtered_zero_freq': len(enhanced_keywords) - len(final_keywords),  # 记录过滤数量
                    'source_used': 'multi_source_enhanced'
                }
            }

            filtered_count = len(enhanced_keywords) - len(final_keywords)
            if filtered_count > 0:
                logger.info(
                    f"文档 {chunk_id} 完成: 提取 {len(final_keywords)} 个关键词，过滤 {filtered_count} 个零频率关键词")
            else:
                logger.info(f"文档 {chunk_id} 完成: 提取 {len(final_keywords)} 个关键词")

            return result

        except Exception as e:
            logger.error(f"文档 {document.get('metadata', {}).get('chunk_id', 'unknown')} 处理失败: {e}")
            return self.create_empty_result(
                document.get('metadata', {}).get('chunk_id', 'unknown'),
                document.get('metadata', {}).get('content_length', 0)
            )

    def extract_keywords_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量提取关键词

        Args:
            documents: 文档数据列表

        Returns:
            关键词提取结果列表
        """
        results = []
        total_docs = len(documents)

        logger.info(f"开始批量处理 {total_docs} 个文档，批量大小: {self.batch_size}")

        # 分批处理
        for i in range(0, total_docs, self.batch_size):
            batch_start = i
            batch_end = min(i + self.batch_size, total_docs)
            batch = documents[batch_start:batch_end]

            logger.info(
                f"处理批次 {i // self.batch_size + 1}/{(total_docs - 1) // self.batch_size + 1}: 文档 {batch_start + 1}-{batch_end}")
            batch_start_time = time.time()

            batch_results = self._process_batch(batch)
            results.extend(batch_results)

            batch_time = time.time() - batch_start_time
            logger.info(f"批次完成，耗时: {batch_time:.2f}秒，平均每文档: {batch_time / len(batch):.2f}秒")

            # 批次间短暂延迟，避免资源竞争
            if batch_end < total_docs:
                time.sleep(0.5)

        return results

    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理单个批次

        Args:
            batch: 批次文档数据

        Returns:
            处理结果列表
        """
        batch_results = []

        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_doc = {
                executor.submit(self.extract_keywords_single, doc): doc
                for doc in batch
            }

            # 收集结果
            for future in as_completed(future_to_doc):
                doc = future_to_doc[future]
                try:
                    result = future.result()
                    batch_results.append(result)

                    chunk_id = result['chunk_id']
                    keyword_count = result['keyword_stats']['total_keywords']
                    if keyword_count > 0:
                        avg_confidence = result['keyword_stats']['avg_confidence']
                        logger.debug(f"文档 {chunk_id} 完成: {keyword_count} 个关键词，平均置信度: {avg_confidence:.4f}")
                    else:
                        logger.debug(f"文档 {chunk_id} 完成: 无关键词提取")

                except Exception as e:
                    logger.error(f"文档处理失败: {e}")
                    # 添加空结果作为降级
                    chunk_id = doc.get('metadata', {}).get('chunk_id', 'unknown')
                    content_length = doc.get('metadata', {}).get('content_length', 0)
                    batch_results.append(self.create_empty_result(chunk_id, content_length))

        return batch_results

    def create_empty_result(self, chunk_id: str, content_length: int) -> Dict[str, Any]:
        """创建空结果"""
        return {
            'chunk_id': chunk_id,
            'content_length': content_length,
            'keywords': [],
            'keyword_stats': {
                'total_keywords': 0,
                'avg_confidence': 0.0,
                'keyword_density': 0.0,
                'source_breakdown': {}
            }
        }


def load_wow_data(input_path: str) -> List[Dict[str, Any]]:
    """
    加载WoW数据

    Args:
        input_path: 输入文件路径

    Returns:
        文档列表
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 处理不同的数据结构
        if isinstance(data, list):
            documents = data
        elif isinstance(data, dict):
            if 'documents' in data:
                documents = data['documents']
            elif 'chunks' in data:
                documents = data['chunks']
            else:
                # 尝试提取可能的文档字段
                documents = []
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                        documents = value
                        break
                if not documents:
                    documents = [data]
        else:
            documents = [data]

        logger.info(f"成功加载 {len(documents)} 个文档")

        # 验证文档结构
        valid_documents = []
        for doc in documents:
            if isinstance(doc, dict) and ('content' in doc or 'text' in doc):
                # 标准化字段名
                if 'text' in doc and 'content' not in doc:
                    doc['content'] = doc['text']
                valid_documents.append(doc)
            else:
                logger.warning(f"跳过无效文档格式: {type(doc)}")

        logger.info(f"有效文档数量: {len(valid_documents)}")
        return valid_documents

    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return []


def save_keyword_data(keyword_results: List[Dict[str, Any]], output_path: str):
    """
    保存关键词数据

    Args:
        keyword_results: 关键词结果列表
        output_path: 输出文件路径
    """
    try:
        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(keyword_results, f, ensure_ascii=False, indent=2)

        # 统计信息
        total_docs = len(keyword_results)
        docs_with_keywords = sum(1 for result in keyword_results if result['keywords'])
        total_keywords = sum(len(result['keywords']) for result in keyword_results)

        # 计算平均置信度（只计算有关键词的文档）
        confidences = [
            result['keyword_stats']['avg_confidence']
            for result in keyword_results
            if result['keywords']
        ]
        avg_confidence = np.mean(confidences) if confidences else 0

        # 计算Top-K使用情况
        top_k_values = []
        for result in keyword_results:
            if 'extraction_info' in result:
                top_k_values.append(result['extraction_info']['top_k_used'])

        logger.info("=" * 60)
        logger.info("关键词提取完成统计:")
        logger.info(f"处理文档总数: {total_docs}")
        logger.info(f"有关键词的文档: {docs_with_keywords} ({docs_with_keywords / total_docs * 100:.1f}%)")
        logger.info(f"总关键词数: {total_keywords}")
        logger.info(f"平均每文档关键词: {total_keywords / total_docs:.1f}")
        logger.info(f"平均置信度: {avg_confidence:.4f}")
        if top_k_values:
            logger.info(f"平均Top-K值: {np.mean(top_k_values):.1f}")
            logger.info(f"Top-K范围: {min(top_k_values)}-{max(top_k_values)}")
        logger.info(f"结果已保存到: {output_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"保存数据失败: {e}")


def main():
    """主函数"""
    input_path = str(paths.processed_data / "chunks" / "wow_langchain_ner_re_LLM_batch_100.json")
    output_path = str(paths.processed_data / "chunks" / "wow_keywords_test.json")

    # 本地模型路径
    model_path = str(paths.models / "chunk_model" / "all-mpnet-base-v2")

    # 初始化增强版提取器
    extractor = WoWKeywordExtractor(
        model_path=model_path,
        batch_size=4,
        max_workers=2,
        min_keywords=8,  # 提高最小关键词数
        max_keywords=25  # 提高最大关键词数
    )

    # 加载数据
    logger.info("开始加载数据...")
    documents = load_wow_data(input_path)
    if not documents:
        logger.error("没有加载到文档，程序退出")
        return

    # 提取关键词
    logger.info("开始关键词提取...")
    start_time = time.time()

    keyword_results = extractor.extract_keywords_batch(documents)

    total_time = time.time() - start_time
    logger.info(f"关键词提取完成，总耗时: {total_time:.2f}秒")
    logger.info(f"平均每文档处理时间: {total_time / len(documents):.2f}秒")

    # 保存结果
    save_keyword_data(keyword_results, output_path)


if __name__ == "__main__":
    main()
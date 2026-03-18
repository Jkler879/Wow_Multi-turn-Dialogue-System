from config.paths import paths
import json
import hashlib
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.documents.transformers import BaseDocumentTransformer
import spacy
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WowNERPipeline:
    """WoW数据集NER处理管道 - 完整的端到端处理流程"""

    def __init__(
            self,
            input_file: str,
            output_file: str,
            batch_size: int = 8,
            max_workers: int = 2,
            entity_types: List[str] = None,
            evaluate_performance: bool = True
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.entity_types = entity_types or ["PERSON", "ORG", "GPE", "DATE"]
        self.evaluate_performance = evaluate_performance

        # 初始化组件
        self.loader = WowDatasetLoader()
        self.processor = WowNERProcessor(
            batch_size=batch_size,
            max_workers=max_workers,
            entity_types=entity_types
        )
        self.evaluator = NERPerformanceEvaluator()

    def run(self) -> Dict[str, Any]:
        """执行完整的NER处理流程"""
        logger.info("Starting WoW NER Pipeline...")

        # 1. 加载数据
        raw_data = self.loader.load_from_json(self.input_file)
        documents = self.loader.convert_to_documents(raw_data)

        # 2. 性能评估（可选）
        performance_metrics = {}
        if self.evaluate_performance and len(documents) > 10:
            sample_size = min(50, len(documents))
            performance_metrics = self.evaluator.evaluate_processing_speed(
                documents[:sample_size], self.processor
            )

        # 3. 处理所有文档
        logger.info("Starting full NER processing...")
        processed_documents = self.processor.transform_documents(documents)

        # 4. 验证输出
        validation_result = self.evaluator.validate_ner_output(processed_documents)

        # 5. 转换格式并保存
        processed_wow_data = self.loader.convert_to_wow_format(processed_documents)

        # 确保输出目录存在
        output_dir = os.path.dirname(self.output_file)
        os.makedirs(output_dir, exist_ok=True)

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_wow_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Processed data saved to: {self.output_file}")

        # 6. 生成统计信息
        stats = self._generate_statistics(
            processed_documents, performance_metrics, validation_result
        )

        self._log_final_stats(stats)
        return stats

    def _generate_statistics(self, processed_documents: List[Document],
                             performance_metrics: Dict, validation_result: bool) -> Dict[str, Any]:
        """生成处理统计信息"""
        total_entities = sum(len(doc.metadata.get("ner_entities", [])) for doc in processed_documents)
        entities_by_type = {}
        role_entities_count = 0

        for doc in processed_documents:
            for entity in doc.metadata.get("ner_entities", []):
                entity_type = entity["entity_type"]
                entities_by_type[entity_type] = entities_by_type.get(entity_type, 0) + 1

                if entity.get("is_role"):
                    role_entities_count += 1

        return {
            "total_documents_processed": len(processed_documents),
            "total_entities_extracted": total_entities,
            "role_entities_count": role_entities_count,
            "entities_by_type": entities_by_type,
            "performance_metrics": performance_metrics,
            "validation_passed": validation_result
        }

    def _log_final_stats(self, stats: Dict[str, Any]):
        """记录最终统计信息"""
        logger.info("Processing completed successfully!")
        logger.info(
            f"Extracted {stats['total_entities_extracted']} entities from {stats['total_documents_processed']} documents")
        logger.info(f"Role entities: {stats['role_entities_count']}")
        for entity_type, count in stats['entities_by_type'].items():
            logger.info(f"  {entity_type}: {count}")


class WowNERProcessor(BaseDocumentTransformer):
    """WoW数据集专用的NER处理器 - 基于en_core_web_trf模型"""

    def __init__(self,
                 batch_size: int = 8,
                 max_workers: int = 2,
                 entity_types: List[str] = None):
        """
        初始化WoW NER处理器
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.entity_types = entity_types or ["PERSON", "ORG", "GPE", "DATE"]

        # 定义需要角色标记的特定词汇（精确匹配，区分大小写）
        self.ROLE_ENTITIES = {
            "Wizard": "assistant",
            "Apprentice": "user"
        }

        logger.info(f"Loading spaCy model: en_core_web_trf")

        try:
            self.nlp = spacy.load(
                "en_core_web_trf",
                disable=["tagger", "parser", "lemmatizer", "attribute_ruler"]
            )
            logger.info("spaCy model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            raise

    def _generate_chunk_id(self, metadata: Dict[str, Any], content: str) -> str:
        """为WoW数据集生成全局唯一的chunk_id"""
        source = metadata.get("source", "wow_dataset")
        document_id = metadata.get("document_id", "unknown")
        chunk_seq = metadata.get("chunk_seq", 0)

        # 生成文本哈希
        text_hash = hashlib.md5(content.encode()).hexdigest()[:8]

        return f"{source}_{document_id}_{chunk_seq}_{text_hash}"

    def _extract_entities_from_spacy_doc(self, spacy_doc, chunk_id: str) -> List[Dict]:
        """从spaCy文档中提取实体并格式化 - 基于文本内容标记角色"""
        ner_entities = []
        entity_counters = {}

        for ent in spacy_doc.ents:
            if ent.label_ in self.entity_types:
                # 初始化实体类型计数器
                if ent.label_ not in entity_counters:
                    entity_counters[ent.label_] = 0
                entity_counters[ent.label_] += 1

                # 基础实体数据
                entity_data = {
                    "entity_id": f"{chunk_id}_{ent.label_}_{entity_counters[ent.label_]}",
                    "text": ent.text,
                    "entity_type": ent.label_,
                    "normalized_name": ent.text.lower(),
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "ner_source": "en_core_web_trf"
                }

                # 关键修改：基于文本内容标记角色，不考虑实体类型
                # 只要文本是"Wizard"或"Apprentice"，就标记为角色
                if ent.text in self.ROLE_ENTITIES:
                    entity_data["is_role"] = True
                    entity_data["role_type"] = self.ROLE_ENTITIES[ent.text]
                    logger.debug(
                        f"Marked role entity: '{ent.text}' (spaCy type: {ent.label_}) as {entity_data['role_type']}")

                ner_entities.append(entity_data)

        return ner_entities

    def _create_ordered_metadata(self, original_metadata: Dict[str, Any], ner_entities: List[Dict]) -> Dict[str, Any]:
        """创建有序的metadata，确保ner_entities紧跟在split_type后面"""
        # 定义字段顺序
        ordered_fields = [
            "source", "document_id", "chunk_id", "chunk_seq",
            "content_hash", "content_type", "content_length",
            "dialog_id", "dialog_index", "topic", "turn_count",
            "speaker_role", "wizard_eval_score", "has_checked_evidence",
            "total_retrieved_passages", "total_retrieved_topics",
            "evidence_density", "prev_chunk_id", "next_chunk_id",
            "split_type", "ner_entities"
        ]

        # 添加其他可能存在的字段
        for key in original_metadata.keys():
            if key not in ordered_fields:
                ordered_fields.append(key)

        # 构建有序的metadata
        ordered_metadata = {}
        for field in ordered_fields:
            if field == "ner_entities":
                ordered_metadata[field] = ner_entities
            elif field in original_metadata:
                ordered_metadata[field] = original_metadata[field]

        return ordered_metadata

    def _process_single_document(self, document: Document) -> Document:
        """处理单个WoW文档"""
        try:
            content = document.page_content
            original_metadata = document.metadata.copy()

            # 生成全局唯一的chunk_id
            chunk_id = self._generate_chunk_id(original_metadata, content)
            original_metadata["chunk_id"] = chunk_id

            # 使用spaCy处理文本
            spacy_doc = self.nlp(content)

            # 提取实体（包含角色标记）
            ner_entities = self._extract_entities_from_spacy_doc(spacy_doc, chunk_id)

            # 创建有序的metadata
            ordered_metadata = self._create_ordered_metadata(original_metadata, ner_entities)

            return Document(page_content=content, metadata=ordered_metadata)

        except Exception as e:
            logger.error(f"Error processing document {document.metadata.get('document_id', 'unknown')}: {e}")
            return document

    def transform_documents(self,
                            documents: List[Document],
                            **kwargs: Any) -> List[Document]:
        """转换WoW数据集文档"""
        logger.info(f"Starting NER processing for {len(documents)} WoW documents")
        logger.info(f"Using batch_size: {self.batch_size}, max_workers: {self.max_workers}")

        processed_docs = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            total_batches = (len(documents) + self.batch_size - 1) // self.batch_size

            for batch_idx in range(total_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, len(documents))
                batch_docs = documents[start_idx:end_idx]

                logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_docs)} documents)")

                futures = [executor.submit(self._process_single_document, doc)
                           for doc in batch_docs]

                for future in tqdm(futures, desc=f"Batch {batch_idx + 1}"):
                    try:
                        processed_doc = future.result(timeout=300)
                        processed_docs.append(processed_doc)
                    except Exception as e:
                        logger.error(f"Error in batch processing: {e}")
                        processed_docs.extend(batch_docs)

        logger.info(f"Completed NER processing. Processed {len(processed_docs)} documents")
        return processed_docs

    async def atransform_documents(self,
                                   documents: List[Document],
                                   **kwargs: Any) -> List[Document]:
        """异步转换文档"""
        return self.transform_documents(documents, **kwargs)


class WowDatasetLoader:
    """WoW数据集加载器"""

    @staticmethod
    def load_from_json(file_path: str) -> List[Dict]:
        """从JSON文件加载WoW数据集"""
        logger.info(f"Loading WoW dataset from {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded {len(data)} items from WoW dataset")
            return data
        except Exception as e:
            logger.error(f"Error loading WoW dataset: {e}")
            raise

    @staticmethod
    def convert_to_documents(wow_data: List[Dict]) -> List[Document]:
        """将WoW数据转换为LangChain Document格式"""
        documents = []

        for item in wow_data:
            if isinstance(item, dict):
                content = item.get('content', '')
                metadata = item.get('metadata', {})

                if 'source' not in metadata:
                    metadata['source'] = 'wow_dataset'

                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            else:
                logger.warning(f"Unexpected data format: {type(item)}")

        logger.info(f"Converted {len(documents)} WoW items to LangChain documents")
        return documents

    @staticmethod
    def convert_to_wow_format(processed_docs: List[Document]) -> List[Dict]:
        """将处理后的文档转换回WoW格式"""
        wow_data = []

        for doc in processed_docs:
            wow_item = {
                'content': doc.page_content,
                'metadata': doc.metadata
            }
            wow_data.append(wow_item)

        return wow_data


class NERPerformanceEvaluator:
    """NER性能评估器"""

    @staticmethod
    def evaluate_processing_speed(documents: List[Document],
                                  processor: WowNERProcessor,
                                  sample_size: int = 100) -> Dict[str, float]:
        """评估处理速度 - 核心指标"""
        import time

        if len(documents) < sample_size:
            sample_docs = documents
        else:
            sample_docs = documents[:sample_size]

        logger.info(f"Running performance evaluation on {len(sample_docs)} documents")

        start_time = time.time()
        _ = processor.transform_documents(sample_docs)
        end_time = time.time()

        processing_time = end_time - start_time
        docs_per_second = len(sample_docs) / processing_time

        metrics = {
            "total_documents_processed": len(sample_docs),
            "total_processing_time_seconds": round(processing_time, 2),
            "documents_per_second": round(docs_per_second, 2),
            "seconds_per_document": round(processing_time / len(sample_docs), 4)
        }

        logger.info("Performance Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

        return metrics

    @staticmethod
    def validate_ner_output(processed_docs: List[Document]) -> bool:
        """修正后的验证函数 - 只检查角色实体是否被正确标记"""
        required_entity_fields = [
            "entity_id", "text", "entity_type", "normalized_name",
            "start_char", "end_char", "ner_source"
        ]

        validation_passed = True
        error_count = 0

        for i, doc in enumerate(processed_docs):
            # 检查chunk_id格式
            chunk_id = doc.metadata.get("chunk_id", "")
            if not chunk_id.startswith("wow_dataset_"):
                logger.warning(f"Document {i}: chunk_id format issue: {chunk_id}")
                validation_passed = False
                error_count += 1

            # 检查字段位置
            metadata_keys = list(doc.metadata.keys())
            if "split_type" in metadata_keys and "ner_entities" in metadata_keys:
                split_index = metadata_keys.index("split_type")
                ner_index = metadata_keys.index("ner_entities")
                if ner_index != split_index + 1:
                    logger.warning(f"Document {i}: ner_entities not placed after split_type")
                    validation_passed = False
                    error_count += 1

            ner_entities = doc.metadata.get("ner_entities", [])

            for j, entity in enumerate(ner_entities):
                # 检查必需字段
                missing_fields = [field for field in required_entity_fields if field not in entity]
                if missing_fields:
                    logger.warning(f"Document {i}, Entity {j}: Missing fields {missing_fields}")
                    validation_passed = False
                    error_count += 1

                # 检查entity_id格式
                if not entity["entity_id"].startswith(chunk_id):
                    logger.warning(f"Document {i}, Entity {j}: entity_id format mismatch")
                    validation_passed = False
                    error_count += 1

                # 修正：只有当实体文本是"Wizard"或"Apprentice"时才检查角色标记
                if entity["text"] in ["Wizard", "Apprentice"]:
                    if "is_role" not in entity:
                        logger.warning(
                            f"Document {i}, Entity {j}: Role entity '{entity['text']}' missing 'is_role' field")
                        validation_passed = False
                        error_count += 1
                    elif "role_type" not in entity:
                        logger.warning(
                            f"Document {i}, Entity {j}: Role entity '{entity['text']}' missing 'role_type' field")
                        validation_passed = False
                        error_count += 1
                    elif entity["text"] == "Wizard" and entity.get("role_type") != "assistant":
                        logger.warning(
                            f"Document {i}, Entity {j}: Wizard has wrong role_type '{entity.get('role_type')}', should be 'assistant'")
                        validation_passed = False
                        error_count += 1
                    elif entity["text"] == "Apprentice" and entity.get("role_type") != "user":
                        logger.warning(
                            f"Document {i}, Entity {j}: Apprentice has wrong role_type '{entity.get('role_type')}', should be 'user'")
                        validation_passed = False
                        error_count += 1

        if validation_passed:
            logger.info("NER output validation: PASSED")
        else:
            logger.warning(f"NER output validation: FAILED with {error_count} errors")

        return validation_passed


# 使用示例
if __name__ == "__main__":
    # 指向paths.py: 项目路径设计
    input_path = str(paths.processed_data / "chunks" / "wow_langchain_chunks_final.json")
    output_path = str(paths.processed_data / "chunks" / "wow_langchain_ner_final.json")
    model_path = str(paths.models / "chunk_model" / "en_core_web_trf")
    # 配置参数
    config = {
        "input_file": input_path,
        "output_file": output_path,
        "batch_size": 8,
        "max_workers": 2,
        "entity_types": ["PERSON", "ORG", "GPE", "DATE"],
        "evaluate_performance": True
    }

    # 验证输入文件
    if not os.path.exists(config["input_file"]):
        print(f"错误: 输入文件不存在 - {config['input_file']}")
        exit(1)

    print(f"开始处理 WoW 数据集...")
    print(f"输入文件: {config['input_file']}")
    print(f"输出文件: {config['output_file']}")

    try:
        # 创建并运行管道
        pipeline = WowNERPipeline(**config)
        stats = pipeline.run()

        # 打印结果
        print("\n" + "=" * 50)
        print("处理完成!")
        print("=" * 50)
        print(f"处理的文档数量: {stats['total_documents_processed']}")
        print(f"提取的实体总数: {stats['total_entities_extracted']}")
        print(f"角色实体数量: {stats['role_entities_count']}")
        print(f"验证通过: {stats['validation_passed']}")

        print("\n实体类型分布:")
        for entity_type, count in stats['entities_by_type'].items():
            print(f"  {entity_type}: {count}")

        if stats['performance_metrics']:
            print("\n性能指标:")
            for metric, value in stats['performance_metrics'].items():
                print(f"  {metric}: {value}")

    except Exception as e:
        logger.error(f"处理失败: {e}")
        print(f"处理过程中出现错误: {e}")
        raise
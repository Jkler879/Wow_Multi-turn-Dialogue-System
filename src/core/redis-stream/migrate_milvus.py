"""
Milvus 知识库迁移脚本：从旧集合迁移到支持 BM25 全文检索的新集合
- 保留所有原有字段、数据、向量
- 新增 sparse 字段，用于 BM25 全文检索
- 重建原有所有索引
"""

import time
import logging
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema,
    DataType, Function, FunctionType, utility
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== 配置 ==========
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
OLD_COLLECTION_NAME = "wow_knowledge_base"
NEW_COLLECTION_NAME = "wow_knowledge_base_v2"


def migrate():
    # 1. 连接 Milvus
    logger.info(f"正在连接 Milvus {MILVUS_HOST}:{MILVUS_PORT}")
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    logger.info("连接成功")

    # 2. 检查旧集合
    if not utility.has_collection(OLD_COLLECTION_NAME):
        raise RuntimeError(f"旧集合 {OLD_COLLECTION_NAME} 不存在")
    old_collection = Collection(OLD_COLLECTION_NAME)
    old_collection.load()
    total_old = old_collection.num_entities
    logger.info(f"旧集合 {OLD_COLLECTION_NAME} 实体数: {total_old}")

    # 3. 获取旧集合的 schema 和索引信息
    old_schema = old_collection.schema
    old_fields = old_schema.fields
    logger.info(f"旧集合字段: {[f.name for f in old_fields]}")

    # 获取所有索引信息
    old_indexes = old_collection.indexes
    index_info = {}
    for idx in old_indexes:
        field_name = idx.field_name
        # 提取索引参数
        params = idx.params.copy()
        index_type = params.pop('index_type', None)
        metric_type = params.pop('metric_type', None)
        index_info[field_name] = {
            'index_name': idx.index_name,
            'index_type': index_type,
            'metric_type': metric_type,
            'params': params
        }
        logger.info(f"字段 '{field_name}' 索引: {index_info[field_name]}")

    # 4. 定义新集合的字段（保留所有旧字段，并新增 sparse 字段）
    new_fields = []
    for field in old_fields:
        if field.dtype == DataType.VARCHAR:
            new_fields.append(FieldSchema(
                name=field.name,
                dtype=field.dtype,
                max_length=field.max_length,
                is_primary=field.is_primary,
                enable_analyzer=field.enable_analyzer,
                enable_match=field.enable_match
            ))
        elif field.dtype == DataType.ARRAY:
            new_fields.append(FieldSchema(
                name=field.name,
                dtype=field.dtype,
                element_type=field.element_type,
                max_length=field.max_length,
                max_capacity=field.max_capacity
            ))
        elif field.dtype == DataType.JSON:
            new_fields.append(FieldSchema(
                name=field.name,
                dtype=field.dtype
            ))
        elif field.dtype == DataType.FLOAT_VECTOR:
            dim = field.params.get('dim', 768)
            new_fields.append(FieldSchema(
                name=field.name,
                dtype=field.dtype,
                dim=dim
            ))
        else:
            new_fields.append(FieldSchema(
                name=field.name,
                dtype=field.dtype,
                is_primary=field.is_primary
            ))

    # 新增 sparse 字段
    new_fields.append(FieldSchema(
        name="sparse",
        dtype=DataType.SPARSE_FLOAT_VECTOR,
        description="BM25 稀疏向量"
    ))

    # 5. 定义 BM25 Function
    bm25_func = Function(
        name="content_bm25",
        input_field_names=["content"],
        output_field_names=["sparse"],
        function_type=FunctionType.BM25
    )

    # 6. 创建新集合
    new_schema = CollectionSchema(new_fields, functions=[bm25_func])
    if utility.has_collection(NEW_COLLECTION_NAME):
        logger.warning(f"新集合 {NEW_COLLECTION_NAME} 已存在，将先删除")
        utility.drop_collection(NEW_COLLECTION_NAME)
        time.sleep(2)

    new_collection = Collection(NEW_COLLECTION_NAME, new_schema)
    logger.info(f"新集合 {NEW_COLLECTION_NAME} 创建成功")

    # 7. 为 sparse 字段创建 BM25 索引
    new_collection.create_index("sparse", {
        "index_type": "AUTOINDEX",
        "metric_type": "BM25"
    })
    logger.info("sparse 字段索引创建成功")

    # 8. 迁移数据
    logger.info("开始迁移数据...")
    old_field_names = [f.name for f in old_fields]
    results = old_collection.query(expr="", output_fields=old_field_names, limit=total_old)

    entities = []
    for ent in results:
        entity = {}
        for fname in old_field_names:
            entity[fname] = ent.get(fname)
        entities.append(entity)

    new_collection.insert(entities)
    new_collection.flush()
    logger.info(f"数据迁移完成，共插入 {len(entities)} 条")

    # 9. 重建所有索引
    for field_name, idx_info in index_info.items():
        if field_name == "sparse":
            continue
        logger.info(f"正在为字段 '{field_name}' 创建索引: {idx_info}")
        try:
            # 构建索引参数字典
            index_params = {}
            if idx_info['index_type']:
                index_params["index_type"] = idx_info['index_type']
            if idx_info['metric_type']:
                index_params["metric_type"] = idx_info['metric_type']
            if idx_info['params']:
                index_params["params"] = idx_info['params']
            # 创建索引
            new_collection.create_index(field_name, index_params)
        except Exception as e:
            logger.error(f"创建索引失败 {field_name}: {e}")

    logger.info("所有索引重建完成")

    # 10. 加载集合并验证数据量
    new_collection.load()
    total_new = new_collection.num_entities
    logger.info(f"新集合实体数: {total_new}")
    if total_new == total_old:
        logger.info("✅ 数据量一致，迁移成功！")
    else:
        logger.error(f"❌ 数据量不一致！旧库 {total_old}，新库 {total_new}")


if __name__ == "__main__":
    migrate()

# long_memory_collection.py
"""
Milvus 长期记忆集合索引创建（Long Memory Collection）
独立运行一次，用于创建 Milvus 知识库集合和索引。

索引设计：
 - 向量索引： HNSW
 1、不同于知识库向量索引，长期记忆索引因每个用户拥有独立的记忆空间，记忆条数通常优先，内存占用不是主要瓶颈
 2、HNSW 索引构建是增量式的，插入新数据无需重建整张图，适合长期记忆不断追加的场景

"""

import sys
import logging
from pymilvus import (
    connections, utility, Collection, CollectionSchema,
    FieldSchema, DataType
)

# ==================== 配置 ====================
MILVUS_URI = "http://localhost:19530"          # 你的 Milvus 地址
COLLECTION_NAME = "user_long_term_memory"       # 与 Mem0 配置一致
VECTOR_DIM = 768                                 # 必须与嵌入模型维度一致

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mem0_compatible_collection():
    """创建与 Mem0 默认 schema 兼容的集合"""
    logger.info(f"连接 Milvus: {MILVUS_URI}")
    connections.connect(uri=MILVUS_URI)

    if utility.has_collection(COLLECTION_NAME):
        logger.warning(f"集合 {COLLECTION_NAME} 已存在，跳过创建")
        return

    # Mem0 默认使用的字段（基于 mem0 1.0.5 源码）
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
        FieldSchema(name="payload", dtype=DataType.VARCHAR, max_length=65535),  # 存储记忆文本
        FieldSchema(name="metadata", dtype=DataType.JSON),                      # 存储元数据
        FieldSchema(name="created_at", dtype=DataType.INT64),
        FieldSchema(name="updated_at", dtype=DataType.INT64),
    ]

    schema = CollectionSchema(fields, description="Mem0 长期记忆集合")
    collection = Collection(COLLECTION_NAME, schema)
    logger.info(f"✅ 集合 {COLLECTION_NAME} 创建成功")

    # 创建向量索引（HNSW）
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    }
    collection.create_index("vector", index_params)
    logger.info("✅ 向量索引创建成功")

    # 为 metadata 字段创建索引（可选，用于标量过滤）
    collection.create_index("metadata", {"index_type": "INVERTED"})
    logger.info("✅ metadata 索引创建成功")

    collection.load()
    logger.info("✅ 集合已加载，准备就绪")


if __name__ == "__main__":
    create_mem0_compatible_collection()
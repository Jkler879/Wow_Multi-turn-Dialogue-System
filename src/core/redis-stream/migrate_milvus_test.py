"""
验证知识库迁移结果：对比新旧集合的数据和索引
"""

import time
import logging
from pymilvus import (
    connections, Collection, utility
)
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== 配置 ==========
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
OLD_COLLECTION_NAME = "wow_knowledge_base"
NEW_COLLECTION_NAME = "wow_knowledge_base_v2"


def compare_collections():
    # 1. 连接 Milvus
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    logger.info("已连接 Milvus")

    # 2. 检查集合是否存在
    if not utility.has_collection(OLD_COLLECTION_NAME):
        logger.error(f"旧集合 {OLD_COLLECTION_NAME} 不存在")
        return
    if not utility.has_collection(NEW_COLLECTION_NAME):
        logger.error(f"新集合 {NEW_COLLECTION_NAME} 不存在")
        return

    old_collection = Collection(OLD_COLLECTION_NAME)
    new_collection = Collection(NEW_COLLECTION_NAME)

    # 加载集合
    old_collection.load()
    new_collection.load()
    logger.info("集合加载完成")

    # 3. 对比实体数量
    old_num = old_collection.num_entities
    new_num = new_collection.num_entities
    logger.info(f"旧集合实体数: {old_num}")
    logger.info(f"新集合实体数: {new_num}")
    if old_num == new_num:
        logger.info("✅ 实体数一致")
    else:
        logger.error(f"❌ 实体数不一致！旧 {old_num}，新 {new_num}")

    # 4. 获取所有字段名（用于抽样查询）
    old_fields = [f.name for f in old_collection.schema.fields]
    new_fields = [f.name for f in new_collection.schema.fields]
    logger.info(f"旧集合字段: {old_fields}")
    logger.info(f"新集合字段: {new_fields}")
    if set(old_fields).issubset(set(new_fields)):
        logger.info("✅ 新集合包含旧集合所有字段")
    else:
        logger.error("❌ 新集合缺少某些字段")

    # 5. 随机抽样对比数据（取前5条）
    logger.info("开始抽样对比数据...")
    # 使用 query 获取所有数据（数据量小，直接全量）
    old_results = old_collection.query(expr="", output_fields=old_fields, limit=old_num)
    new_results = new_collection.query(expr="", output_fields=old_fields, limit=new_num)

    # 按 chunk_id 建立映射（假设主键为 chunk_id）
    old_map = {r['chunk_id']: r for r in old_results}
    new_map = {r['chunk_id']: r for r in new_results}

    # 对比前5条 chunk_id（取交集）
    common_ids = set(old_map.keys()) & set(new_map.keys())
    sample_ids = list(common_ids)[:5]

    for idx, cid in enumerate(sample_ids):
        old_ent = old_map[cid]
        new_ent = new_map[cid]
        logger.info(f"\n样本 {idx+1}: chunk_id = {cid}")
        # 对比标量字段（排除向量和 sparse，因为向量需特殊比较）
        for fname in old_fields:
            if fname == 'vector':
                # 向量对比：计算余弦相似度（因浮点精度，应几乎相同）
                vec_old = np.array(old_ent[fname])
                vec_new = np.array(new_ent[fname])
                cos_sim = np.dot(vec_old, vec_new) / (np.linalg.norm(vec_old) * np.linalg.norm(vec_new))
                if abs(cos_sim - 1.0) < 1e-6:
                    logger.info(f"  字段 {fname}: ✅ 向量一致 (余弦相似度 ≈ {cos_sim:.6f})")
                else:
                    logger.error(f"  字段 {fname}: ❌ 向量不一致 (余弦相似度 = {cos_sim:.6f})")
            elif fname not in new_ent:
                logger.error(f"  字段 {fname}: ❌ 新集合缺失该字段")
            else:
                if old_ent[fname] == new_ent[fname]:
                    logger.info(f"  字段 {fname}: ✅ 一致")
                else:
                    logger.error(f"  字段 {fname}: ❌ 不一致 (旧: {old_ent[fname]}, 新: {new_ent[fname]})")
        # 检查新集合的 sparse 字段是否自动生成（应为非空）
        if 'sparse' in new_ent and new_ent['sparse'] is not None:
            logger.info(f"  字段 sparse: ✅ 已自动生成")
        else:
            logger.warning(f"  字段 sparse: 未生成或为空")

    # 6. 对比索引
    logger.info("\n开始对比索引...")
    old_indexes = {idx.field_name: idx for idx in old_collection.indexes}
    new_indexes = {idx.field_name: idx for idx in new_collection.indexes}

    # 旧集合的索引字段（除主键外）
    for field_name, old_idx in old_indexes.items():
        if field_name not in new_indexes:
            logger.error(f"❌ 新集合缺少字段 '{field_name}' 的索引")
            continue
        new_idx = new_indexes[field_name]
        # 比较索引类型和参数
        old_params = old_idx.params
        new_params = new_idx.params
        # 简化比较：忽略可能自动添加的字段
        old_type = old_params.get('index_type')
        new_type = new_params.get('index_type')
        if old_type == new_type:
            logger.info(f"字段 '{field_name}' 索引类型一致: {old_type}")
        else:
            logger.error(f"字段 '{field_name}' 索引类型不一致: 旧 {old_type}, 新 {new_type}")

        # 比较 metric_type（如果有）
        old_metric = old_params.get('metric_type')
        new_metric = new_params.get('metric_type')
        if old_metric == new_metric:
            logger.info(f"   metric_type 一致: {old_metric}")
        else:
            logger.warning(f"   metric_type 不一致: 旧 {old_metric}, 新 {new_metric}")

        # 比较其他参数（如 nlist, refine 等）
        old_other = {k: v for k, v in old_params.items() if k not in ['index_type', 'metric_type']}
        new_other = {k: v for k, v in new_params.items() if k not in ['index_type', 'metric_type']}
        if old_other == new_other:
            logger.info(f"   其他参数一致: {old_other}")
        else:
            logger.warning(f"   其他参数不一致: 旧 {old_other}, 新 {new_other}")

    # 检查新集合独有的 sparse 索引
    if 'sparse' in new_indexes:
        logger.info("✅ 新集合包含 sparse 字段索引")
    else:
        logger.error("❌ 新集合缺少 sparse 字段索引")

    # 7. 功能测试：执行向量检索和 BM25 检索
    logger.info("\n开始功能测试...")
    test_query = "时间旅行"
    # 向量检索
    vector_results = new_collection.search(
        data=[[0.0]*768],  # 临时使用零向量，实际应使用查询向量，这里仅测试索引是否存在
        anns_field="vector",
        param={"metric_type": "COSINE"},
        limit=3,
        output_fields=["chunk_id", "content"]
    )
    logger.info("✅ 向量检索测试通过（索引存在）")

    # BM25 检索
    try:
        bm25_results = new_collection.search(
            data=[test_query],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=3,
            output_fields=["chunk_id", "content"]
        )
        logger.info("✅ BM25 检索测试通过（索引存在）")
    except Exception as e:
        logger.error(f"❌ BM25 检索测试失败: {e}")

    logger.info("\n验证完成！")

if __name__ == "__main__":
    compare_collections()
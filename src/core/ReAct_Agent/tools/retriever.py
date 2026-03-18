"""
ReAct Agent 检索工具集 (Retrieval Tool)

该模块提供面向知识库的高效检索能力，是 RAG系统的核心组件之一。它通过多路召回、融合排序和深度重排，确保从海量文档中准确定位最相关信息。

主要功能：
    1. 向量检索：基于 Milvus 2.6+ 的 IVF_RABITQ 索引，对用户问题嵌入进行语义相似性搜索，召回语义相关 Top_10 文档块。
    2. 全文检索：利用 Milvus 内置的 BM25 Function，将content 字段实时转换为稀疏向量并执行全文检索，
       弥补纯向量检索在关键词匹配上的不足, 召回 BM25 评分最高的 TOP_10 文档块。
    3. RRF 融合：将向量检索和全文检索的结果按排序位置进行加权融合，生成综合排序的 Top_10 个候选文档列表。
    4. 重排：使用本地 BGE重排模型 bge-reranker-v2 对融合后的 Top_5 个文档进行精细化打分，并返回带置信度分数的最终结果。

索引设计：
1、向量索引: IVF_RABITQ (milvus2.6 版本新索引), 支撑后续向量检索
    - 内存占用极低（最高32倍压缩）
    - 查询性能快（超越 HNSW、ivf_sq8 ）
    - 召回精度高（超越 HNSW、ivf_sq8 ）

2、全文索引: sparse字段 + BM25 function（Milvus 2.6 + 版本新功能）
    - 对 content字段的文本内容进行分词并计算 BM25稀疏向量，存入 sparse字段
    - 为 sparse 字段创建 BM25索引，检索时能计算 BM25分数，支撑后续系统全文检索
    - 支撑 BM25 全文检索，避免了引入 Elasticsearch 增加数据库维护成本
    - 添加了集合迁移代码，在当前目录同级 migrate_milvus.py中
    - 已通过测试用例 migrate_milvus_test.py，足以支撑后续 Agent系统的 BM25 全文检索。

输入输出：
    - 输入：用户查询字符串（英文，知识库为英文数据，同语言检索精度最高），以及 top_k 参数。
    - 输出：JSON 列表，每个元素包含文档内容 (`content`) 和相关性分数 (`score`)，
      分数越高表示文档与查询越相关。该分数由重排模型直接给出，可作为 LLM 判断
      信息可信度的依据。

Author: Ke Meng
Created: 2026-01-20
Version: 1.0.1
Last Modified: 2026-03-18

变更记录：
    - 1.0.1 (2026-03-15):
                        改动1、将system prompt 有关检索工具集的内容分离到工具集的 description 中，
                              Agent在调用工具前已经获取所有工具集的自然语言描述。
                        改动2、新建支持全文检索的 Milvus 知识库集合，与长期记忆集合逻辑分离
                        改动3、修改重排后对于候选文档评分小于0.5的固定阈值判断，只设置一个最低分数（-5.0），
                              超过分数皆返回文档（设置上限Top5）
                        改动4、引入 RRF 融合算法进一步缩减候选文档数量

    - 1.0.0 (2026-01-25): 初始版本

依赖：
    - sentence-transformers: 用于加载和使用 BGE CrossEncoder 重排器。
    - pymilvus: Milvus 向量数据库 Python SDK。
    - langchain-core: 用于嵌入模型接口和工具创建。
    - numpy: 科学计算。
"""

import logging
import concurrent.futures
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field
from langchain_milvus import Milvus
from langchain_core.embeddings import Embeddings
from sentence_transformers import CrossEncoder
import numpy as np
from src.core.ReAct_Agent.tools.base import create_tool

logger = logging.getLogger(__name__)


# ==================== 重排器封装（改造：返回分数） ====================
class BGEReranker:
    """基于 BGE 的 CrossEncoder 重排器，返回带分数的结果"""
    def __init__(self, model_path: str, device: str = "cpu", batch_size: int = 32):
        self.model = CrossEncoder(model_path, device=device)
        self.batch_size = batch_size

    def rerank(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """
        返回按相关性降序的文档列表，每个元素包含 'content' 和 'score'。
        """
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        logger.debug(f"重排原始分数: {scores}")  # 添加这一行
        # 将分数和文档配对,并按分数降序排序
        scored = list(zip(scores, documents))
        scored.sort(key=lambda x: x[0], reverse=True)
        # 转换为包含 content 和 score 的字典列表
        return [{"content": doc, "score": float(score)} for score, doc in scored]


# ==================== RRF 融合双检索后的答案 ====================
def reciprocal_rank_fusion(
    results_list: List[List[Dict[str, Any]]],
    k: int = 60,
    top_k: int = 20
) -> List[Dict[str, Any]]:
    """
    将多个检索结果列表通过 RRF 算法融合，返回融合后得分最高的 top_k 条结果。
    :param results_list: 每个元素是检索结果列表，每个结果至少包含 'id' 和 'content'
    :param k: RRF常数经验值,较大K会平滑排名差异，较小K更强调前几名
    :param top_k: 返回融合后的结果数量
    :return: 融合后的结果列表，包含 'id', 'content', 'rrf_score'
    """
    # 初始化文档ID到融合分数的映射
    fused_scores = {}
    # 初始化文档ID到文档内容的映射
    doc_info = {}

    # 遍历候选结果
    for results in results_list:
        # 每一条结果的排名
        for rank, hit in enumerate(results, start=1):
            # 获取文档ID
            doc_id = hit.get('id')
            # 如果结果中没有 id，用 content 前 100 字符作为临时 id（避免重复）
            if not doc_id:
                doc_id = hit['content'][:100]
            doc_info[doc_id] = hit['content']
            # RRF 仅基于排名位置计算分数贡献，避免归一化误差
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # 按融合分数降序排序
    sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    # 取前Top_k个结果
    top_ids = sorted_ids[:top_k]

    # 组装返回结果
    fused_results = []
    for doc_id, score in top_ids:
        fused_results.append({
            'id': doc_id,
            'content': doc_info[doc_id],
            'rrf_score': score
        })
    return fused_results


# ==================== 检索工具输入 Schema ====================
class RetrieverInput(BaseModel):
    query: str = Field(description="需要检索的问题")
    top_k: int = Field(default=5, description="返回的文档数量")


# ==================== 主工具类 ====================
class RetrieverTool:
    """
    生产级检索工具：向量检索 + BM25 全文检索 + RRF 融合 + 重排（带绝对下限过滤）
    返回结果包含文档内容和相关性分数，便于 LLM 根据置信度决策。
    """
    def __init__(
        self,
        embedding_model: Embeddings,
        milvus_collection,
        reranker: BGEReranker,
        vector_top_k: int = 10,
        bm25_top_k: int = 10,
        rrf_k: int = 60,
        rrf_final_top_k: int = 10,
        absolute_low: float = -5.0,          # 绝对下限阈值，低于此值视为无可用结果
        vector_search_params: dict = None,   # 向量检索参数
    ):
        """
        :param embedding_model: 嵌入模型
        :param milvus_collection: pymilvus Collection 对象，用于 BM25 全文检索
        :param reranker: BGE 重排器实例
        :param vector_top_k: 向量检索返回的候选数（多于最终 top_k）
        :param bm25_top_k: BM25 检索返回的候选数
        :param rrf_k: RRF 融合常数
        :param rrf_final_top_k: RRF 融合后保留的候选数（送入重排）
        :param absolute_low: 绝对下限阈值，如果重排后最高分低于此值，则返回空列表
        :param vector_search_params: 向量检索参数
        """
        self.embedding_model = embedding_model
        self.collection = milvus_collection
        self.reranker = reranker
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.rrf_final_top_k = rrf_final_top_k
        self.absolute_low = absolute_low
        self.vector_search_params = vector_search_params or {
            "metric_type": "COSINE",
            "params": {"nprobe": 128}  # 可根据索引类型调整
        }

    def _vector_search(self, query: str) -> List[Dict[str, Any]]:
        """向量检索：使用 Milvus 原生 API"""
        query_embedding = self.embedding_model.embed_query(query)
        results = self.collection.search(
            data=[query_embedding],
            anns_field="vector",
            param=self.vector_search_params,
            limit=self.vector_top_k,
            output_fields=["content", "chunk_id"]
        )
        output = []
        for hits in results:
            for hit in hits:
                output.append({
                    'id': hit.entity.get('chunk_id', hit.entity.content[:100]),
                    'content': hit.entity.content,
                    'score': hit.score,
                    'source': 'vector'
                })
        return output

    def _bm25_search(self, query: str) -> List[Dict[str, Any]]:
        """BM25 全文检索，利用 Milvus 2.6+ Function 生成的 sparse 向量"""
        search_params = {"metric_type": "BM25"}
        results = self.collection.search(
            data=[query],  # BM25 检索直接传入文本
            anns_field="sparse",  # 必须是在 schema 中用 Function 定义的稀疏字段
            param=search_params,
            limit=self.bm25_top_k,
            output_fields=["content", "chunk_id"]
        )
        # 解析结果
        output = []
        for hits in results:  # 只有一个 query
            for hit in hits:
                output.append({
                    'id': hit.entity.get('chunk_id', hit.entity.content[:100]),
                    'content': hit.entity.content,
                    'score': hit.score,
                    'source': 'bm25'
                })
        return output

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        主检索入口：向量 + BM25 -> RRF 融合 -> 重排（带绝对下限过滤）
        返回前 top_k 条文档，每条包含 'content' 和 'score'。
        """
        logger.info(f"开始检索: query='{query}', top_k={top_k}")

        # 1、并行执行两路检索
        vector_results = []
        bm25_results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_vector = executor.submit(self._vector_search, query)
            future_bm25 = executor.submit(self._bm25_search, query)

            try:
                vector_results = future_vector.result(timeout=30)
                logger.debug(f"向量检索完成，返回 {len(vector_results)} 条")
            except concurrent.futures.TimeoutError:
                logger.error("向量检索超时")
            except Exception as e:
                logger.error(f"向量检索失败: {e}")

            try:
                bm25_results = future_bm25.result(timeout=30)
                logger.debug(f"BM25 检索完成，返回 {len(bm25_results)} 条")
            except concurrent.futures.TimeoutError:
                logger.error("BM25 检索超时")
            except Exception as e:
                logger.error(f"BM25 检索失败: {e}")

        if not vector_results and not bm25_results:
            logger.warning("两路检索均无结果")
            return []

        # 2、RRF 融合
        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results],
            k=self.rrf_k,
            top_k=self.rrf_final_top_k
        )
        logger.debug(f"RRF 融合后剩余 {len(fused)} 条")
        if logger.isEnabledFor(logging.DEBUG):
            for i, item in enumerate(fused[:3]):
                logger.debug(
                    f"  RRF 结果 #{i + 1}: id={item.get('id')}, rrf_score={item.get('rrf_score'):.3f}, preview={item['content'][:100]}...")

        # 3、提取文档内容列表（供重排器使用）
        candidate_docs = [item['content'] for item in fused]

        # 4、重排
        scored_docs = self.reranker.rerank(query, candidate_docs)
        logger.info(f"⚖️ 重排完成，共 {len(scored_docs)} 条评分文档")
        if logger.isEnabledFor(logging.DEBUG):
            for i, doc in enumerate(scored_docs[:3]):
                logger.debug(f"  重排结果 #{i + 1}: score={doc['score']:.3f}, preview={doc['content'][:100]}...")

        # 5、检查最高分是否低于绝对下限
        if scored_docs and scored_docs[0]["score"] < self.absolute_low:
            logger.warning(f"最高分 {scored_docs[0]['score']:.3f} 低于绝对下限 {self.absolute_low}，返回空结果")
            return []

        # 6、直接取前 top_k 条返回（不再用阈值过滤）
        result = scored_docs[:top_k]
        logger.info(f"📦 检索最终输出: 返回 {len(result)} 条文档")
        if logger.isEnabledFor(logging.DEBUG):
            for i, doc in enumerate(result):
                logger.debug(f"  最终结果 #{i + 1}: score={doc['score']:.3f}, preview={doc['content'][:100]}...")
        return result

    def as_tool(self) -> Any:
        """转换为 LangChain 工具，返回结构化结果（包含分数）"""
        return create_tool(
            name="knowledge_retriever",
            description=(
                "当用户询问事实性信息时，应优先调用此工具。\n"
                "**输入要求**：\n"
                "- `query` 参数**必须使用英文**，且应直接基于用户问题（已由前置模块改写为英文）。\n"
                "- 输入必须为 JSON 对象，例如 {\"query\": \"What are the best sci-fi TV series?\"}\n\n"
                "**返回结果**：JSON 列表，每个元素包含 'content'（文档内容）和 'score'（相关性分数）。\n"
                "**分数说明**：分数越高表示文档与查询越相关。正分（如 >0.5）表示高度相关，0 分附近表示中等相关，负分表示相关性较弱。若列表为空，则知识库中无相关信息。"
            ),
            args_schema=RetrieverInput,
            func=self.retrieve,
            return_direct=False,
        )
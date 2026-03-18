"""
ReAct Agent 实体关系验证工具集 (Relation Verifier Tool)

该模块对检索后的候选文档进行知识图谱数据库 Neo4j的实体关系验证，通过实体关系的判断、评分，输出结构化的验证结果提供给LLM，
进一步确认候选文档的准确性。


主要功能：
 - 1、定义 LLM 结构化输入：至少两个实体、最大验证的实体对数 10（防止组合爆炸）
 - 2、定义结构化输出： 包含实体名称、关系类型、置信度、证据原文、来源文档块ID，保证结果知识库可溯源
 - 3、批量执行多个实体对的查找、验证


Author: Ke Meng
Created: 2026-01-20
Version: 1.0.1
Last Modified: 2026-03-18

变更记录：
    - 1.0.1 (2026-03-15):
                        改动1、工具集Prompt描述：将system prompt 有关实体关系验证工具集的内容分离到工具集的 description 中，
                                              Agent在调用工具前已经获取所有工具集的自然语言描述。
                        改动2、结构化输出：不仅返回实体关系验证结果，
                                        还返回关系类型、置信度、证据原文（知识图谱数据入库时已提取对应元数据）。

    - 1.0.0 (2026-01-28): 初始版本

依赖：
    - langchain_neo4j: 用于验证知识图谱实体关系的数据库
    - langchain-core: 用于工具创建。
"""

import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, validator
from langchain_core.tools import StructuredTool
from langchain_neo4j import Neo4jGraph
import itertools

logger = logging.getLogger(__name__)


# ==================== 输入 Schema ====================
class VerifyRelationsInput(BaseModel):
    """实体关系验证工具输入参数"""
    entities: List[str] = Field(description="实体名称列表，至少包含两个实体")
    min_confidence: float = Field(default=0.5, description="最小置信度阈值", ge=0.0, le=1.0)
    max_pairs: int = Field(default=10, description="最大验证的实体对数，防止组合爆炸")

    @validator('entities')
    def validate_entities(cls, v):
        if len(v) < 2:
            raise ValueError('至少需要两个实体')
        return v


# ==================== 输出 Schema ====================
class RelationInfo(BaseModel):
    """单条关系信息"""
    entity_a: str = Field(description="第一个实体名称")
    entity_b: str = Field(description="第二个实体名称")
    type: str = Field(description="关系类型")
    confidence: float = Field(description="置信度")
    evidence: str = Field(description="证据原文")
    source_chunk: str = Field(description="来源文档块ID")
    direction: str = Field(description="关系方向", pattern="^(a_to_b|b_to_a|bidirectional)$")


class VerifyRelationsOutput(BaseModel):
    """实体关系验证工具输出结果"""
    relations: List[RelationInfo] = Field(default_factory=list, description="所有找到的关系")
    missing_entities: List[str] = Field(default_factory=list, description="未在知识库中找到的实体")
    stats: Dict[str, Any] = Field(default_factory=dict, description="统计信息")


# ==================== 工具实现 ====================
class RelationVerifier:
    """
    实体关系验证器：批量验证多个实体之间的所有关系
    """

    def __init__(
        self,
        graph: Neo4jGraph,
        entity_label: str = "Entity",
        default_min_confidence: float = 0.5,
    ):
        """
        :param graph: Neo4jGraph 实例（已连接）
        :param entity_label: 实体节点标签（默认 "Entity"）
        :param default_min_confidence: 默认最小置信度阈值
        """
        self.graph = graph
        self.entity_label = entity_label
        self.default_min_confidence = default_min_confidence

        # 验证连接
        try:
            self.graph.query("RETURN 1")
            logger.info("✅ Neo4j 连接成功")
        except Exception as e:
            logger.error(f"❌ Neo4j 连接失败: {e}")
            raise

    def _normalize_name(self, name: str) -> str:
        """实体名称标准化：去除空格、转小写、下划线替换空格"""
        return name.strip().lower().replace(" ", "_")

    def _find_entity_ids(self, entity_names: List[str]) -> Tuple[Dict[str, str], List[str]]:
        """
        批量查找实体ID，返回映射字典和缺失列表
        """
        entity_ids = {}
        missing = []
        for name in entity_names:
            norm = self._normalize_name(name)
            # 同时匹配 name 和 normalized_name 字段
            query = f"""
            MATCH (n:{self.entity_label})
            WHERE n.name = $name OR n.normalized_name = $norm
            RETURN n.entity_id AS entity_id, n.name AS name
            LIMIT 1
            """
            result = self.graph.query(query, {"name": name, "norm": norm})
            if result and result[0].get("entity_id"):
                entity_ids[name] = result[0]["entity_id"]
            else:
                missing.append(name)
        return entity_ids, missing

    def _query_relations_batch(
        self,
        entity_pairs: List[Tuple[str, str, str, str]],  # (name_a, id_a, name_b, id_b)
        min_confidence: float
    ) -> List[Dict[str, Any]]:
        """
        批量查询多对实体间的关系
        """
        all_relations = []
        for name_a, id_a, name_b, id_b in entity_pairs:
            query = f"""
            MATCH (a:{self.entity_label} {{entity_id: $id_a}})-[r]-(b:{self.entity_label} {{entity_id: $id_b}})
            WHERE r.confidence >= $min_confidence
            RETURN 
                type(r) AS type,
                r.confidence AS confidence,
                r.evidence AS evidence,
                r.source_chunk AS source_chunk,
                CASE 
                    WHEN startNode(r) = a AND endNode(r) = b THEN 'a_to_b'
                    WHEN startNode(r) = b AND endNode(r) = a THEN 'b_to_a'
                    ELSE 'bidirectional'
                END AS direction
            ORDER BY r.confidence DESC
            """
            params = {
                "id_a": id_a,
                "id_b": id_b,
                "min_confidence": min_confidence,
            }
            results = self.graph.query(query, params)
            for rel in results:
                all_relations.append({
                    "entity_a": name_a,
                    "entity_b": name_b,
                    "type": rel["type"],
                    "confidence": rel["confidence"],
                    "evidence": rel["evidence"],
                    "source_chunk": rel.get("source_chunk", "unknown"),
                    "direction": rel["direction"],
                })
        return all_relations

    def verify(
        self,
        entities: List[str],
        min_confidence: Optional[float] = None,
        max_pairs: int = 10
    ) -> Dict[str, Any]:
        """
        验证多个实体间的关系
        """
        start_time = time.time()
        logger.info(f"🔍 实体关系验证输入: entities={entities}, min_confidence={min_confidence}, max_pairs={max_pairs}")

        threshold = min_confidence if min_confidence is not None else self.default_min_confidence

        # 1. 查找实体ID
        id_map, missing = self._find_entity_ids(entities)
        logger.info(f"📌 实体查找结果: 找到 {len(id_map)} 个实体, 缺失 {len(missing)} 个: {missing}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"实体ID映射: {id_map}")

        # 2. 生成所有可能的实体对（组合）
        present_names = [n for n in entities if n in id_map]
        pairs = list(itertools.combinations(present_names, 2))
        if len(pairs) > max_pairs:
            logger.warning(f"实体对过多 ({len(pairs)}), 限制为 {max_pairs}")
            pairs = pairs[:max_pairs]
        logger.info(f"🔗 待验证实体对数量: {len(pairs)}")

        # 3. 构建带ID的对
        entity_pairs = [
            (name_a, id_map[name_a], name_b, id_map[name_b])
            for name_a, name_b in pairs
        ]

        # 4. 批量查询关系
        relations = self._query_relations_batch(entity_pairs, threshold)
        logger.info(f"📊 查询到 {len(relations)} 条关系")
        if logger.isEnabledFor(logging.DEBUG) and relations:
            for i, rel in enumerate(relations[:3]):
                logger.debug(
                    f"  关系 #{i + 1}: {rel['entity_a']} -[{rel['type']}]-> {rel['entity_b']} (conf={rel['confidence']:.2f})")

        # 5. 组装输出
        output = VerifyRelationsOutput(
            relations=[RelationInfo(**r) for r in relations],
            missing_entities=missing,
            stats={
                "total_entities": len(entities),
                "found_entities": len(id_map),
                "pairs_checked": len(pairs),
                "relations_found": len(relations),
                "query_time_ms": int((time.time() - start_time) * 1000),
            }
        )
        output_dict = output.dict()
        logger.info(
            f"✅ 最终输出: 共 {len(output_dict['relations'])} 条关系, 缺失实体 {len(output_dict['missing_entities'])} 个, 耗时 {output_dict['stats']['query_time_ms']}ms")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"完整输出: {output_dict}")
        return output_dict

    def as_tool(self) -> StructuredTool:
        """转换为 LangChain 工具"""
        return StructuredTool.from_function(
            func=self.verify,
            name="relation_verifier",
            description=(
                "验证多个实体之间的关系。当用户问题涉及实体间关系（如“A和B有什么关系？”）且检索结果中提到了相关实体时，调用此工具。"
                "输入必须为 JSON 格式，包含 'entities' 键，值为至少两个实体名称的列表。"
                "示例：{'entities': ['哈利·波特', '伏地魔']}。"
                "返回结构化的关系信息，包括关系类型、置信度、证据原文等。"
            ),
            args_schema=VerifyRelationsInput,
            return_direct=False,
        )


# ==================== 工厂函数 ====================
def create_relation_verifier_tool(
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str,
    entity_label: str = "Entity",
    default_min_confidence: float = 0.5,
) -> StructuredTool:
    """
    快速创建关系验证工具实例
    """
    graph = Neo4jGraph(
        url=neo4j_url,
        username=neo4j_user,
        password=neo4j_password,
        database="neo4j",
    )
    verifier = RelationVerifier(
        graph=graph,
        entity_label=entity_label,
        default_min_confidence=default_min_confidence,
    )
    return verifier.as_tool()

from config.paths import paths
import json
import hashlib
import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_openai import ChatOpenAI
import logging
import os
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityRelationOutputParser(BaseOutputParser):
    """自定义输出解析器，处理LLM的实体和关系输出"""

    def parse(self, text: str) -> Dict[str, Any]:
        """解析LLM输出为结构化数据"""
        try:
            # 清理JSON输出
            cleaned_text = self._clean_json_output(text)
            data = json.loads(cleaned_text)

            # 验证必需字段
            required_sections = ["new_entities", "relations"]
            for section in required_sections:
                if section not in data:
                    raise ValueError(f"Missing required section: {section}")

            return data

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            logger.error(f"原始内容: {text[:500]}...")
            # 返回空结构作为降级
            return {
                "new_entities": {},
                "relations": []
            }

    def _clean_json_output(self, text: str) -> str:
        """清理LLM输出的JSON"""
        # 移除可能的代码标记
        text = re.sub(r'```json\s*|\s*```', '', text).strip()

        # 提取第一个JSON对象
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)

        return text


class EntityRelationExtractor:
    """实体关系抽取器 - 基于LangChain 1.0.5"""

    def __init__(self, openai_api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        # 初始化LLM
        self.llm = ChatOpenAI(
            model_name="deepseek-chat",
            openai_api_key=openai_api_key,
            base_url=base_url,
            temperature=0.3,  # 实体关系抽取和实体识别是高确定性任务
            max_tokens=4000
        )

        # 11种适配于wow数据集的实体类型及自然语言描述(Prompt增强)
        self.entity_definitions = {
            "MEDIA_CONTENT": "具体的媒体作品如电影、电视剧、书籍、音乐专辑、游戏等",
            "MEDIA_FRANCHISE": "媒体系列、IP宇宙、品牌系列如漫威宇宙、星球大战系列等",
            "CONCEPT_ABSTRACT": "抽象概念、思想、情感状态如自由、幸福、创造力、无聊等",
            "CONCEPT_LIFESTYLE": "生活方式、习惯、日常活动如健身、旅行、阅读、社交等",
            "ENTERTAINMENT_GENRE": "娱乐类型如科幻、喜剧、恐怖、浪漫、动作等",
            "ACADEMIC_FIELD": "学术领域如物理学、心理学、计算机科学、文学等",
            "CREATIVE_WORK": "创意作品如小说、绘画、音乐、设计等",
            "TECHNOLOGY": "技术产品、数字工具、软件平台如智能手机、AI系统、应用程序等",
            "HEALTH_WELLNESS": "健康话题、医疗、健身、营养、心理健康等",
            "RELATION_CAUSAL": "因果关系实体如导致、引起、影响、结果等",
            "RELATION_SOLUTION": "解决方案实体如解决、帮助、改善、治疗等"
        }

        # 25种适配于wow数据集的关系类型及自然语言描述(Prompt增强)
        self.relation_definitions = {
            # === 创作与内容关系 ===
            "CREATED_BY": "创作关系（作品→创作者）",
            "WRITTEN_BY": "写作关系（书籍→作者）",
            "DIRECTED_BY": "导演关系（电影→导演）",
            "COMPOSED_BY": "作曲关系（音乐→作曲家）",
            "DEVELOPED_BY": "开发关系（软件/技术→开发者）",

            # === 分类与归属关系 ===
            "BELONGS_TO_GENRE": "类型归属（作品→类型）",
            "PART_OF_SERIES": "系列归属（作品→系列）",
            "CATEGORIZED_AS": "分类关系（概念→类别）",
            "IS_A": "类别关系（实例→类别）",

            # === 时空关系 ===
            "LOCATED_IN": "位置关系（实体→地点）",
            "FOUNDED_IN": "创立时间（组织→时间）",
            "OCCURRED_IN": "发生时间（事件→时间）",
            "DURING_PERIOD": "时期关系（事件→时期）",

            # === 功能与用途关系 ===
            "USES_TECHNOLOGY": "技术使用（产品→技术）",
            "SOLVES_PROBLEM": "问题解决（方案→问题）",
            "PROVIDES_SERVICE": "服务提供（组织→服务）",
            "HAS_FUNCTION": "功能描述（工具→功能）",

            # === 影响与效果关系 ===
            "CAUSES": "因果关系（原因→结果）",
            "LEADS_TO": "导致关系（行动→结果）",
            "AFFECTS": "影响关系（因素→对象）",
            "IMPROVES": "改善关系（方法→状况）",

            # === 学术与研究关系 ===
            "STUDIED_BY": "研究关系（主题→研究者）",
            "RESEARCHED_IN": "研究领域（课题→领域）",
            "DISCOVERED_BY": "发现关系（现象→发现者）",
            "INVENTED_BY": "发明关系（发明→发明者）",

            # === 组成与结构关系 ===
            "CONTAINS": "包含关系（整体→部分）",
            "MADE_OF": "材料关系（物品→材料）",

            # 解释性关系
            "DEFINED_AS": "定义关系（概念→定义）",
            "EXEMPLIFIED_BY": "例证关系（概念→例子）",

            # 比较关系
            "SIMILAR_TO": "相似关系",
            "DIFFERENT_FROM": "差异关系",

            # 程度关系
            "MORE_EFFICIENT_THAN": "效率比较",
            "LESS_EXPENSIVE_THAN": "成本比较"
        }

        # 8种典型关系few-shot示例(Prompt增强)
        self.few_shot_examples = [
            # 示例1：基础创作关系
            {
                "text": "The Harry Potter series was written by J.K. Rowling and is considered a fantasy novel.",
                "entities": [
                    {"entity_id": "ent_1", "text": "Harry Potter series", "type": "MEDIA_FRANCHISE"},
                    {"entity_id": "ent_2", "text": "J.K. Rowling", "type": "PERSON"},
                    {"entity_id": "ent_3", "text": "fantasy novel", "type": "ENTERTAINMENT_GENRE"}
                ],
                "relations": [
                    {
                        "subject_entity_id": "ent_1",
                        "predicate": "WRITTEN_BY",
                        "object_entity_id": "ent_2"
                    },
                    {
                        "subject_entity_id": "ent_1",
                        "predicate": "BELONGS_TO_GENRE",
                        "object_entity_id": "ent_3"
                    }
                ]
            },

            # 示例2：时空关系
            {
                "text": "Microsoft Corporation was founded in 1975 in Albuquerque, New Mexico.",
                "entities": [
                    {"entity_id": "ent_1", "text": "Microsoft Corporation", "type": "ORG"},
                    {"entity_id": "ent_2", "text": "1975", "type": "DATE"},
                    {"entity_id": "ent_3", "text": "Albuquerque", "type": "GPE"},
                    {"entity_id": "ent_4", "text": "New Mexico", "type": "GPE"}
                ],
                "relations": [
                    {
                        "subject_entity_id": "ent_1",
                        "predicate": "FOUNDED_IN",
                        "object_entity_id": "ent_2"
                    },
                    {
                        "subject_entity_id": "ent_1",
                        "predicate": "LOCATED_IN",
                        "object_entity_id": "ent_3"
                    },
                    {
                        "subject_entity_id": "ent_3",
                        "predicate": "LOCATED_IN",
                        "object_entity_id": "ent_4"
                    }
                ]
            },

            # 示例3：技术使用关系
            {
                "text": "Tesla electric vehicles use lithium-ion battery technology for energy storage.",
                "entities": [
                    {"entity_id": "ent_1", "text": "Tesla electric vehicles", "type": "TECHNOLOGY"},
                    {"entity_id": "ent_2", "text": "lithium-ion battery technology", "type": "TECHNOLOGY"}
                ],
                "relations": [
                    {
                        "subject_entity_id": "ent_1",
                        "predicate": "USES_TECHNOLOGY",
                        "object_entity_id": "ent_2"
                    }
                ]
            },

            # 示例4：因果关系
            {
                "text": "Smoking tobacco causes lung cancer and other respiratory diseases.",
                "entities": [
                    {"entity_id": "ent_1", "text": "Smoking tobacco", "type": "HEALTH_WELLNESS"},
                    {"entity_id": "ent_2", "text": "lung cancer", "type": "HEALTH_WELLNESS"},
                    {"entity_id": "ent_3", "text": "respiratory diseases", "type": "HEALTH_WELLNESS"}
                ],
                "relations": [
                    {
                        "subject_entity_id": "ent_1",
                        "predicate": "CAUSES",
                        "object_entity_id": "ent_2"
                    },
                    {
                        "subject_entity_id": "ent_1",
                        "predicate": "CAUSES",
                        "object_entity_id": "ent_3"
                    }
                ]
            },

            # 示例5：分类关系
            {
                "text": "Python is a high-level programming language used for web development and data science.",
                "entities": [
                    {"entity_id": "ent_1", "text": "Python", "type": "TECHNOLOGY"},
                    {"entity_id": "ent_2", "text": "programming language", "type": "ACADEMIC_FIELD"},
                    {"entity_id": "ent_3", "text": "web development", "type": "TECHNOLOGY"},
                    {"entity_id": "ent_4", "text": "data science", "type": "ACADEMIC_FIELD"}
                ],
                "relations": [
                    {
                        "subject_entity_id": "ent_1",
                        "predicate": "IS_A",
                        "object_entity_id": "ent_2"
                    },
                    {
                        "subject_entity_id": "ent_1",
                        "predicate": "USES_TECHNOLOGY",
                        "object_entity_id": "ent_3"
                    },
                    {
                        "subject_entity_id": "ent_1",
                        "predicate": "USES_TECHNOLOGY",
                        "object_entity_id": "ent_4"
                    }
                ]
            },

            # 示例6：解决问题关系
            {
                "text": "Vaccines help prevent infectious diseases like measles and polio.",
                "entities": [
                    {"entity_id": "ent_1", "text": "Vaccines", "type": "HEALTH_WELLNESS"},
                    {"entity_id": "ent_2", "text": "infectious diseases", "type": "HEALTH_WELLNESS"},
                    {"entity_id": "ent_3", "text": "measles", "type": "HEALTH_WELLNESS"},
                    {"entity_id": "ent_4", "text": "polio", "type": "HEALTH_WELLNESS"}
                ],
                "relations": [
                    {
                        "subject_entity_id": "ent_1",
                        "predicate": "SOLVES_PROBLEM",
                        "object_entity_id": "ent_2"
                    },
                    {
                        "subject_entity_id": "ent_1",
                        "predicate": "SOLVES_PROBLEM",
                        "object_entity_id": "ent_3"
                    },
                    {
                        "subject_entity_id": "ent_1",
                        "predicate": "SOLVES_PROBLEM",
                        "object_entity_id": "ent_4"
                    }
                ]
            },

            # 示例7：组成关系
            {
                "text": "The human brain contains neurons and glial cells that process information.",
                "entities": [
                    {"entity_id": "ent_1", "text": "human brain", "type": "HEALTH_WELLNESS"},
                    {"entity_id": "ent_2", "text": "neurons", "type": "HEALTH_WELLNESS"},
                    {"entity_id": "ent_3", "text": "glial cells", "type": "HEALTH_WELLNESS"}
                ],
                "relations": [
                    {
                        "subject_entity_id": "ent_1",
                        "predicate": "CONTAINS",
                        "object_entity_id": "ent_2"
                    },
                    {
                        "subject_entity_id": "ent_1",
                        "predicate": "CONTAINS",
                        "object_entity_id": "ent_3"
                    }
                ]
            },

            # 示例8：复杂多关系
            {
                "text": "The Internet, developed by DARPA researchers, uses TCP/IP protocols and enables global communication.",
                "entities": [
                    {"entity_id": "ent_1", "text": "Internet", "type": "TECHNOLOGY"},
                    {"entity_id": "ent_2", "text": "DARPA researchers", "type": "ORG"},
                    {"entity_id": "ent_3", "text": "TCP/IP protocols", "type": "TECHNOLOGY"},
                    {"entity_id": "ent_4", "text": "global communication", "type": "CONCEPT_ABSTRACT"}
                ],
                "relations": [
                    {
                        "subject_entity_id": "ent_1",
                        "predicate": "DEVELOPED_BY",
                        "object_entity_id": "ent_2"
                    },
                    {
                        "subject_entity_id": "ent_1",
                        "predicate": "USES_TECHNOLOGY",
                        "object_entity_id": "ent_3"
                    },
                    {
                        "subject_entity_id": "ent_1",
                        "predicate": "ENABLES",
                        "object_entity_id": "ent_4"
                    }
                ]
            }
        ]

        # 初始化输出解析器
        self.output_parser = EntityRelationOutputParser()

        # 创建prompt模板
        self.prompt_template = self._create_prompt_template()

        # 创建LCEL链（LangChain 1.0+核心写法）
        self.chain = self.prompt_template | self.llm | self.output_parser

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """创建实体识别与实体关系抽取的System/User prompt模板"""

        # 1. System Prompt - 固定规则和定义
        entity_definitions_str = "\n".join(
            [f"- {entity_type}: {description}"
             for entity_type, description in self.entity_definitions.items()]
        )

        relation_definitions_str = "\n".join(
            [f"- {rel_type}: {description}"
             for rel_type, description in self.relation_definitions.items()]
        )

        few_shot_str = "\n\n".join([
            f"示例 {i + 1}:\n文本: {example['text']}\n实体: {json.dumps(example['entities'], ensure_ascii=False)}\n关系: {json.dumps(example['relations'], ensure_ascii=False)}"
            for i, example in enumerate(self.few_shot_examples[:3])  # 限制示例数量
        ])

        system_template = """
    # 🔍 WoW数据集实体关系抽取专家

    ## 📋 角色定位
    您是一个专门处理**WoW英文数据集**的实体抽取及关系抽取专家，具备以下核心能力：
    - 精准识别11种领域特定实体
    - 基于25种预定义关系类型，识别实体间的语义关联
    - 基于文本证据进行合理推断
    - 提供准确的置信度评估

    ## 📊 数据源说明
    您处理的**单个JSON文档**包含：
    - 完整的文本内容
    - 已有的NER实体元数据（4种基础类型）
    - 需要在此基础上扩展11种领域实体和实体关系的抽取

    ## 任务1: 扩展11种领域适配实体类型
    基于以下定义，识别文档中符合要求的**新实体**：

    {entity_definitions}

    ### 实体识别要求 ###:
    - 扫描整个文档内容，识别符合定义的实体
    - 为每个识别实体提供合理的置信度(0-1)
    - 只识别11种领域适配实体类型，不要自行添加新的字段
    - 每种实体类型独立判断，识别到就添加，没识别到就保持空列表

    ## 任务2: 抽取实体关系
    请基于以下25种关系类型的定义与介绍，识别实体之间的关系：

    {relation_definitions}

    ### 关系抽取要求 ###：
    - 只识别文本中明确提及或合理推断的实体和关系
    - 为每个实体和关系提供合理的置信度(0-1)
    - 每个关系必须有明确的文本证据支持 - 这是关键要求！
    - 证据文本必须直接从原文中提取或合理推断，用于证明关系存在
    - 不要编造evidence，如果找不到合适的证据就不要创建该关系

    ## 输出格式要求：
    请严格按照以下JSON格式输出：

    {{
      "new_entities": {{
        "MEDIA_CONTENT": [
          {{
            "text": "实体在文档中的原文",
            "type": "MEDIA_CONTENT",
            "normalized_name": "标准化名称",
            "start_char": 在文档中的起始位置,
            "end_char": 在文档中的结束位置,
            "confidence": 置信度,
            "source": "LLM"
          }}
        ],
        // ... 其他实体类型结构相同
      }},
      "relations": [
        {{
          "subject_entity_text": "主语实体原文",
          "object_entity_text": "宾语实体原文", 
          "predicate": "关系类型",
          "confidence": 关系置信度,
          "evidence": "支持关系的具体文本证据，必须直接从原文中提取"
        }}
      ]
    }}

    ## 处理范式说明：
    - 以下是few-shot示例，展示了不同关系类型的界定。请仔细学习这些示例的处理模式：

    {few_shot_examples}

    ## ⚡ 核心处理原则
    1. **证据驱动**: 每个实体和关系必须有明确的文本证据支持
    2. **置信度合理**: 基于文本明确程度提供0-1的置信度评分  
    3. **位置精确**: 尽可能提供准确的字符位置信息
    4. **格式严格**: 完全遵循指定的JSON输出格式
    5. **独立判断**: 每种实体类型独立识别，无实体则保持空列表
    """

        # 2. User Prompt - 具体数据和执行指令
        human_template = """
    ## 🎪 当前处理文档

    **文档内容** (来自JSON的content字段):
    {content}
    **已有基础实体** (来自JSON的ner_entities字段):
    {existing_entities}
    
    ✅ 执行指令
    基于您的专家能力和上述定义，请完成以下任务：

    1、扫描全文识别11种新实体，严格按类型分类并提供准确位置和合理置信度      
    2、构建所有实体关系，确保证据充分并提供准确位置和合理置信度
    3、严格输出符合JSON格式要求
    
    质量检查
    ✅ 每种实体类型都独立判断了吗？
    ✅ 每个关系都有具体的文本证据吗？
    ✅ 置信度评分合理吗？
    ✅ 位置信息尽量准确吗？
    ✅ 输出格式完全符合要求吗？
    
    请基于当前文档内容，输出完整的JSON结果。
    """
        # 创建system和user message模板
        system_message = SystemMessagePromptTemplate.from_template(
            template=system_template,
            partial_variables={
                "entity_definitions": entity_definitions_str,
                "relation_definitions": relation_definitions_str,
                "few_shot_examples": few_shot_str
            }
        )

        human_message = HumanMessagePromptTemplate.from_template(
            template=human_template,
            input_variables=["content", "existing_entities"]
        )

        # 组合成ChatPromptTemplate
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

        return chat_prompt
    
    def _generate_entity_id(self, chunk_id: str, entity_type: str, index: int) -> str:
        """生成实体唯一ID"""
        return f"{chunk_id}_{entity_type}_{index}"

    def _generate_relation_id(self, subject_id: str, predicate: str, object_id: str) -> str:
        """生成关系唯一ID"""
        relation_hash = hashlib.md5(f"{subject_id}{predicate}{object_id}".encode()).hexdigest()[:8]
        return f"rel_{subject_id}_{predicate}_{object_id}_{relation_hash}"

    def _convert_to_neo4j_label(self, entity_type: str) -> str:
        """将实体类型转换为Neo4j标签（驼峰命名）"""
        # 将下划线分隔转换为驼峰命名
        parts = entity_type.lower().split('_')
        return ''.join(part.capitalize() for part in parts)

    def extract_entities_relations(self, document: Document) -> Dict[str, Any]:
        """从文档中抽取实体和关系"""
        try:
            content = document.page_content
            metadata = document.metadata

            # 准备现有实体信息
            existing_entities = metadata.get("ner_entities", [])
            existing_entities_info = [
                {
                    "text": entity.get("text", ""),
                    "type": entity.get("entity_type", ""),
                    "entity_id": entity.get("entity_id", "")
                }
                for entity in existing_entities
            ]

            # 调用LCEL链
            logger.info(f"开始处理文档: {metadata.get('chunk_id', 'unknown')}")
            result = self.chain.invoke({
                "content": content,
                "existing_entities": json.dumps(existing_entities_info, ensure_ascii=False)
            })

            return self._post_process_result(result, document)

        except Exception as e:
            logger.error(f"实体关系抽取失败: {e}")
            return {
                "new_entities": {},
                "relations": [],
                "knowledge_graph": {"nodes": [], "relationships": []}
            }

    def _post_process_result(self, llm_result: Dict[str, Any], document: Document) -> Dict[str, Any]:
        """后处理LLM结果，恢复完整的关系字段"""
        metadata = document.metadata
        chunk_id = metadata.get("chunk_id", "unknown")
        content = document.page_content

        # 处理新实体 - 将字典格式的实体转换为平铺列表
        new_entities_processed = []
        entity_text_to_id = {}

        # 首先建立现有实体的文本到ID映射，并过滤角色实体
        existing_entities = metadata.get("ner_entities", [])
        for entity in existing_entities:
            if self._is_role_entity(entity):
                logger.info(f"跳过角色实体: {entity.get('text', 'N/A')}")
                continue
            entity_text_to_id[entity["text"]] = entity["entity_id"]

        # 处理LLM生成的新实体
        entity_type_counters = {}
        new_entities_dict = llm_result.get("new_entities", {})

        for entity_type, entities in new_entities_dict.items():
            if not entities:
                continue

            for entity_data in entities:
                if self._is_role_entity(entity_data):
                    logger.info(f"跳过LLM提取的角色实体: {entity_data.get('text', 'N/A')}")
                    continue

                if entity_type not in entity_type_counters:
                    entity_type_counters[entity_type] = 0
                entity_type_counters[entity_type] += 1

                entity_id = self._generate_entity_id(
                    chunk_id, entity_type, entity_type_counters[entity_type]
                )

                processed_entity = {
                    "entity_id": entity_id,
                    "text": entity_data.get("text", ""),
                    "type": entity_type,
                    "normalized_name": entity_data.get("normalized_name", entity_data.get("text", "").lower()),
                    "start_char": entity_data.get("start_char", -1),
                    "end_char": entity_data.get("end_char", -1),
                    "confidence": entity_data.get("confidence", 0.5),
                    "source": "LLM"
                }

                new_entities_processed.append(processed_entity)
                entity_text_to_id[entity_data["text"]] = entity_id

        # 处理关系 - 恢复完整的关系字段
        relations_processed = []
        for relation in llm_result.get("relations", []):
            subject_text = relation.get("subject_entity_text", "")
            object_text = relation.get("object_entity_text", "")
            predicate = relation.get("predicate", "")
            evidence = relation.get("evidence", "")
            confidence = relation.get("confidence", 0.5)

            # 检查关系中的实体是否包含角色实体
            if self._is_role_entity_text(subject_text) or self._is_role_entity_text(object_text):
                logger.info(f"跳过包含角色实体的关系: {subject_text} -> {predicate} -> {object_text}")
                continue

            # 查找实体ID
            subject_id = entity_text_to_id.get(subject_text)
            object_id = entity_text_to_id.get(object_text)

            if subject_id and object_id and predicate and evidence:
                relation_id = self._generate_relation_id(subject_id, predicate, object_id)

                # 关键：恢复完整的关系字段
                processed_relation = {
                    "relation_id": relation_id,
                    "subject_entity_id": subject_id,
                    "object_entity_id": object_id,
                    "predicate": predicate,  # 恢复：关系类型
                    "confidence": confidence,  # 恢复：置信度
                    "evidence": evidence  # 恢复：证据文本
                }
                relations_processed.append(processed_relation)
            else:
                logger.warning(f"关系数据不完整，跳过: {subject_text} -> {predicate} -> {object_text}")

        # 生成知识图谱
        knowledge_graph = self._generate_knowledge_graph(
            [e for e in existing_entities if not self._is_role_entity(e)] + new_entities_processed,
            relations_processed,
            chunk_id
        )

        return {
            "new_entities": new_entities_processed,
            "relations": relations_processed,
            "knowledge_graph": knowledge_graph
        }

    def _is_role_entity(self, entity: Dict[str, Any]) -> bool:
        """判断实体是否为角色实体"""
        entity_text = entity.get("text", "")

        # 只对首字母大写的"Wizard"和"Apprentice"进行判断
        if entity_text not in ["Wizard", "Apprentice"]:
            return False

        # 检查角色标记字段
        is_role = entity.get("is_role", False)
        role_type = entity.get("role_type", "")

        # 只有当明确标记为角色时才认为是角色实体
        return is_role is True and role_type in ["assistant", "user"]

    def _is_role_entity_text(self, entity_text: str) -> bool:
        """通过实体文本判断是否为角色实体（用于关系过滤）"""
        # 这里我们只检查文本，因为关系数据中没有完整的实体信息
        # 由于角色实体只有特定的两个，所以误判率很低
        return entity_text in ["Wizard", "Apprentice"]

    def _generate_knowledge_graph(self, all_entities: List[Dict], relations: List[Dict], chunk_id: str) -> Dict[
        str, Any]:
        """生成知识图谱格式，过滤角色实体"""
        nodes = []
        relationships = []

        # 处理节点 - 过滤角色实体
        for entity in all_entities:
            # 再次检查是否为角色实体（确保安全）
            if self._is_role_entity(entity):
                continue

            node = {
                "node_id": entity["entity_id"],
                "labels": [self._convert_to_neo4j_label(entity["type"])],
                "properties": {
                    "name": entity.get("normalized_name", entity["text"].lower()),
                    "original_text": entity["text"],
                    "entity_type": entity["type"],
                    "confidence": entity.get("confidence", 0.5),
                    "source": entity.get("source", "unknown")
                }
            }
            nodes.append(node)

        # 处理关系 - 这里的关系已经过滤过角色实体
        for relation in relations:
            relationship = {
                "relationship_id": relation["relation_id"],
                "type": relation["predicate"],
                "start_node_id": relation["subject_entity_id"],
                "end_node_id": relation["object_entity_id"],
                "properties": {
                    "confidence": relation["confidence"],
                    "evidence": relation["evidence"],
                    "source_chunk": chunk_id
                }
            }
            relationships.append(relationship)

        logger.info(f"生成知识图谱: {len(nodes)} 节点, {len(relationships)} 关系")
        return {
            "nodes": nodes,
            "relationships": relationships
        }


class WowDataEnhancer:
    """WoW数据增强处理器"""

    def __init__(self, extractor: EntityRelationExtractor):
        self.extractor = extractor

    def process_document(self, document: Document) -> tuple:
        """处理单个文档，修复列表连接错误"""
        try:
            # 统计角色实体
            role_entities_count = 0
            if 'ner_entities' in document.metadata:
                for entity in document.metadata['ner_entities']:
                    if self.extractor._is_role_entity(entity):
                        role_entities_count += 1

            logger.info(f"发现 {role_entities_count} 个角色实体将被过滤")

            # 抽取实体和关系
            extraction_result = self.extractor.extract_entities_relations(document)

            # 关键修复：按理想顺序构建metadata，正确处理实体列表
            updated_metadata = document.metadata.copy()  # 先复制所有原始字段

            # 获取原始基础实体和新实体
            original_ner_entities = updated_metadata.get("ner_entities", [])
            new_entities = extraction_result.get("new_entities", [])  # 现在已经是列表

            logger.info(f"原始基础实体数量: {len(original_ner_entities)}")
            logger.info(f"新提取实体数量: {len(new_entities)}")

            # 合并实体：原始基础实体 + LLM新提取的实体
            updated_metadata["ner_entities"] = original_ner_entities + new_entities
            logger.info(f"合并后实体总数: {len(updated_metadata['ner_entities'])}")

            # 添加关系数据到metadata - 紧跟在ner_entities后面
            updated_metadata["relations"] = extraction_result.get("relations", [])
            logger.info(f"提取关系数量: {len(updated_metadata['relations'])}")

            # 添加处理时间戳和版本
            updated_metadata["enhancement_timestamp"] = datetime.now().isoformat()
            updated_metadata["enhancement_version"] = "1.0"

            # 重新排序字段，让relations紧跟在ner_entities后面
            reordered_metadata = {}
            fields_before = []  # ner_entities之前的字段
            fields_after = []  # ner_entities之后的字段（除了relations）

            # 分离字段
            for key in updated_metadata.keys():
                if key == "ner_entities":
                    break
                if key not in ["relations", "enhancement_timestamp", "enhancement_version"]:
                    fields_before.append(key)

            for key in updated_metadata.keys():
                if key not in fields_before and key not in ["ner_entities", "relations", "enhancement_timestamp",
                                                            "enhancement_version"]:
                    fields_after.append(key)

            # 按理想顺序构建metadata
            for key in fields_before:
                reordered_metadata[key] = updated_metadata[key]

            reordered_metadata["ner_entities"] = updated_metadata["ner_entities"]
            reordered_metadata["relations"] = updated_metadata["relations"]

            for key in fields_after:
                reordered_metadata[key] = updated_metadata[key]

            reordered_metadata["enhancement_timestamp"] = updated_metadata["enhancement_timestamp"]
            reordered_metadata["enhancement_version"] = updated_metadata["enhancement_version"]

            logger.info(f"优化后metadata字段顺序: {list(reordered_metadata.keys())}")

            # 创建处理后的文档
            processed_doc = Document(
                page_content=document.page_content,
                metadata=reordered_metadata
            )

            # 返回文档和单独的知识图谱
            knowledge_graph = extraction_result.get("knowledge_graph", {"nodes": [], "relationships": []})
            return processed_doc, knowledge_graph

        except Exception as e:
            logger.error(f"文档处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return document, {"nodes": [], "relationships": []}

    def process_documents(self, documents: List[Document]) -> tuple:
        """批量处理文档，返回处理后的文档列表和知识图谱列表"""
        processed_docs = []
        knowledge_graphs = []

        for i, doc in enumerate(documents):
            logger.info(f"处理进度: {i + 1}/{len(documents)}")
            processed_doc, kg = self.process_document(doc)
            processed_docs.append(processed_doc)

            # 只有当知识图谱有内容时才保存
            if kg.get("nodes") or kg.get("relationships"):
                kg_item = {
                    "chunk_id": doc.metadata.get("chunk_id", f"chunk_{i}"),
                    "knowledge_graph": kg,
                    "source_preview": doc.page_content[:100] + "..." if len(
                        doc.page_content) > 100 else doc.page_content
                }
                knowledge_graphs.append(kg_item)

            logger.info(
                f"文档 {i + 1} 处理完成: {len(kg.get('nodes', []))} 节点, {len(kg.get('relationships', []))} 关系")

        return processed_docs, knowledge_graphs


def load_wow_data(input_path: str) -> List[Document]:
    """加载WoW数据集，处理各种数据结构"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []
        logger.info(f"加载的数据类型: {type(data)}")

        # 处理不同的数据结构
        if isinstance(data, list):
            logger.info(f"数据为列表，包含 {len(data)} 个元素")
            items = data
        elif isinstance(data, dict):
            logger.info(f"数据为字典，键: {list(data.keys())}")

            # 尝试从常见键中提取文档列表
            if 'documents' in data and isinstance(data['documents'], list):
                items = data['documents']
                logger.info(f"从'documents'键获取 {len(items)} 个文档")
            elif 'records' in data and isinstance(data['records'], list):
                items = data['records']
                logger.info(f"从'records'键获取 {len(items)} 个文档")
            elif 'data' in data and isinstance(data['data'], list):
                items = data['data']
                logger.info(f"从'data'键获取 {len(items)} 个文档")
            else:
                # 如果找不到列表键，将整个字典视为单个文档
                items = [data]
                logger.info("将整个字典视为单个文档")
        else:
            raise ValueError(f"不支持的数据结构: {type(data)}")

        for i, item in enumerate(items[:1]):  # 只处理第一条用于测试
            logger.info(f"处理第 {i + 1} 条数据，类型: {type(item)}")

            if isinstance(item, dict):
                # 页面内容
                page_content = item.get("content", item.get("text", ""))

                # 元数据：从metadata字段获取或使用整个item
                if "metadata" in item and isinstance(item["metadata"], dict):
                    metadata = item["metadata"].copy()
                    logger.info(f"从metadata字段获取数据，包含字段: {list(metadata.keys())}")

                    # 标准化基础实体字段名
                    if 'ner_entities' in metadata and isinstance(metadata['ner_entities'], list):
                        logger.info(f"找到ner_entities，数量: {len(metadata['ner_entities'])}")

                        # 标准化基础实体字段名
                        standardized_entities = []
                        for entity in metadata['ner_entities']:
                            standardized_entity = entity.copy()

                            # 字段名映射：entity_type -> type, ner_source -> source
                            if 'entity_type' in standardized_entity:
                                standardized_entity['type'] = standardized_entity.pop('entity_type')
                            if 'ner_source' in standardized_entity:
                                standardized_entity['source'] = standardized_entity.pop('ner_source')

                            standardized_entities.append(standardized_entity)

                        # 用标准化后的实体替换原始实体
                        metadata['ner_entities'] = standardized_entities
                        logger.info(f"标准化后基础实体数量: {len(metadata['ner_entities'])}")

                        # 显示前3个标准化后的实体
                        for j, entity in enumerate(metadata['ner_entities'][:3]):
                            logger.info(
                                f"  基础实体 {j + 1}: {entity.get('text', 'N/A')} -> {entity.get('type', 'N/A')} (source: {entity.get('source', 'N/A')})")
                    else:
                        logger.warning("在metadata中未找到ner_entities字段或格式不正确")
                else:
                    # 如果没有metadata字段，尝试直接从item中获取ner_entities
                    metadata = item.copy()
                    logger.info(f"直接从item获取数据，包含字段: {list(metadata.keys())}")

                    # 检查是否有ner_entities在顶层
                    if 'ner_entities' in metadata and isinstance(metadata['ner_entities'], list):
                        logger.info(f"在顶层找到ner_entities，数量: {len(metadata['ner_entities'])}")

                        # 标准化基础实体字段名
                        standardized_entities = []
                        for entity in metadata['ner_entities']:
                            standardized_entity = entity.copy()

                            # 字段名映射：entity_type -> type, ner_source -> source
                            if 'entity_type' in standardized_entity:
                                standardized_entity['type'] = standardized_entity.pop('entity_type')
                            if 'ner_source' in standardized_entity:
                                standardized_entity['source'] = standardized_entity.pop('ner_source')

                            standardized_entities.append(standardized_entity)

                        # 用标准化后的实体替换原始实体
                        metadata['ner_entities'] = standardized_entities
                        logger.info(f"标准化后基础实体数量: {len(metadata['ner_entities'])}")
                    else:
                        logger.warning("在顶层也未找到ner_entities字段")

                # 确保有chunk_id
                if 'chunk_id' not in metadata:
                    import hashlib
                    content_hash = hashlib.md5(page_content.encode()).hexdigest()[:8]
                    metadata['chunk_id'] = f"chunk_{content_hash}"
                    logger.info(f"生成chunk_id: {metadata['chunk_id']}")

                document = Document(
                    page_content=page_content,
                    metadata=metadata
                )
                documents.append(document)
            else:
                logger.warning(f"跳过非字典类型的数据: {type(item)}")

        logger.info(f"成功加载 {len(documents)} 个文档")
        return documents

    except Exception as e:
        logger.error(f"加载WoW数据失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def save_enhanced_data(processed_documents: List[Document], knowledge_graphs: List[Dict],
                       output_path: str, kg_output_path: str):
    """分别保存增强后的数据和知识图谱数据"""
    try:
        # 保存主数据（用于Milvus）
        output_data = []
        for doc in processed_documents:
            item = {
                "content": doc.page_content,
                "metadata": doc.metadata  # 这里包含所有原始metadata + 新增的relations和合并的ner_entities
            }
            output_data.append(item)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"增强数据已保存到: {output_path}, 共 {len(output_data)} 条记录")

        # 保存知识图谱数据（用于Neo4j）
        if knowledge_graphs:
            with open(kg_output_path, 'w', encoding='utf-8') as f:
                json.dump(knowledge_graphs, f, ensure_ascii=False, indent=2)
            logger.info(f"知识图谱数据已保存到: {kg_output_path}, 共 {len(knowledge_graphs)} 条知识图谱")
        else:
            logger.warning("没有知识图谱数据需要保存")

    except Exception as e:
        logger.error(f"保存数据失败: {e}")
        import traceback
        logger.error(traceback.format_exc())


# 使用示例
def main():
    """主执行函数"""

    # 配置API密钥
    api_key = "sk-1c8dc5bdda034300ac759617ea625722"

    # 输入输出路径
    # input_path = "data/processed_data/chunks/wow_langchain_ner_final.json"
    # input_path = "data/processed_data/chunks/wow_first_record_test.json"
    input_path = str(paths.processed_data / "chunks" / "wow_first_record_test.json")
    # output_path = "data/processed_data/chunks/wow_langchain_ner_re_LLM_test1.json"
    output_path = str(paths.processed_data / "chunks" / "wow_langchain_ner_re_LLM_test1.json")
    kg_output_path = str(paths.processed_data / "chunks" /"wow_knowledge_graphs.json")  # 新增知识图谱输出路径

    # 确保输入文件存在
    if not os.path.exists(input_path):
        logger.error(f"输入文件不存在: {input_path}")
        return

    # 初始化抽取器
    extractor = EntityRelationExtractor(openai_api_key=api_key)

    # 初始化增强器
    enhancer = WowDataEnhancer(extractor)

    # 加载WoW数据
    documents = load_wow_data(input_path)

    if not documents:
        logger.error("没有加载到任何文档，程序退出")
        return

    # 处理文档（这里只处理前1条作为测试）
    sample_documents = documents[:1]
    logger.info(f"开始处理 {len(sample_documents)} 个文档")

    # 关键修改：分别获取处理后的文档和知识图谱
    processed_documents, knowledge_graphs = enhancer.process_documents(sample_documents)

    # 验证处理结果
    if processed_documents:
        first_doc = processed_documents[0]
        logger.info("=== 处理结果验证 ===")
        logger.info(f"文档内容长度: {len(first_doc.page_content)}")
        logger.info(f"元数据字段: {list(first_doc.metadata.keys())}")

        # 检查实体合并情况
        ner_entities = first_doc.metadata.get("ner_entities", [])
        logger.info(f"合并后实体总数: {len(ner_entities)}")

        # 统计实体类型
        entity_types = {}
        for entity in ner_entities:
            entity_type = entity.get("type", "unknown")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

        logger.info(f"实体类型分布: {entity_types}")

        # 检查关系
        relations = first_doc.metadata.get("relations", [])
        logger.info(f"关系数量: {len(relations)}")

    # 保存结果
    save_enhanced_data(processed_documents, knowledge_graphs, output_path, kg_output_path)

    logger.info("实体关系抽取和知识图谱生成完成！")


if __name__ == "__main__":
    main()

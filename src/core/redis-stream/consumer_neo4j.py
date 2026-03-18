"""
消息队列 - 消费者 - Neo4j：（Consumer_Neo4j）
消费者：从 Redis Stream 读取消息，写入 Neo4j
包含死信队列处理：重试超过 MAX_RETRIES 的消息转入死信队列。
"""

import json
import time
import redis
from neo4j import GraphDatabase
import logging
import os

# ========== 配置 ==========
REDIS_HOST = "localhost"
REDIS_PORT = 6380
REDIS_STREAM = "knowledge_pipeline"
REDIS_DLQ = "knowledge_pipeline_dlq"
CONSUMER_GROUP = "group_neo4j"
CONSUMER_NAME = f"consumer_neo4j_{os.getpid()}"
MAX_RETRIES = 3
BATCH_SIZE = 10
BLOCK_MS = 5000
REDIS_CONNECT_RETRIES = 5
REDIS_RETRY_DELAY = 2

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "jkler879"

# ========== 日志 ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ========== Redis 连接（带重试） ==========
def connect_redis():
    for i in range(REDIS_CONNECT_RETRIES):
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True, socket_connect_timeout=5)
            r.ping()
            logger.info("Redis 连接成功")
            return r
        except redis.ConnectionError as e:
            logger.warning(f"Redis 连接失败 ({i+1}/{REDIS_CONNECT_RETRIES}): {e}")
            if i == REDIS_CONNECT_RETRIES - 1:
                raise
            time.sleep(REDIS_RETRY_DELAY)


# ========== 确保消费者组存在 ==========
def ensure_consumer_group(redis_client):
    try:
        redis_client.xgroup_create(REDIS_STREAM, CONSUMER_GROUP, id='0', mkstream=True)
        logger.info(f"消费者组 {CONSUMER_GROUP} 创建成功")
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            logger.info(f"消费者组 {CONSUMER_GROUP} 已存在")
        else:
            raise


# ========== Neo4j 连接与初始化 ==========
class Neo4jClient:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._create_constraints_and_indexes()

    def close(self):
        self.driver.close()

    def _create_constraints_and_indexes(self):
        """创建必要的唯一约束和索引（针对公共标签 :Entity）"""
        with self.driver.session() as session:
            try:
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.node_id IS UNIQUE")
                logger.info("唯一约束 :Entity(node_id) 已创建/存在")
            except Exception as e:
                logger.warning(f"创建唯一约束失败: {e}")

            try:
                session.run("CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)")
                logger.info("索引 :Entity(name) 已创建/存在")
            except Exception as e:
                logger.warning(f"创建 name 索引失败: {e}")

    def write_knowledge_graph(self, chunk_id, kg_data):
        """
        将单个 chunk 的知识图谱写入 Neo4j
        kg_data 应包含 nodes 和 relationships 列表
        """
        nodes = kg_data.get("nodes", [])
        relationships = kg_data.get("relationships", [])

        # 调试：打印第一个关系，查看结构
        if relationships:
            print("\n=== 调试信息 ===")
            print(f"Chunk {chunk_id} 的第一个关系原始数据:")
            print(json.dumps(relationships[0], indent=2, ensure_ascii=False))
            print("================\n")

        with self.driver.session() as session:
            tx = session.begin_transaction()
            try:
                # 1. 创建或更新所有节点
                for node in nodes:
                    node_id = node["node_id"]
                    original_labels = node.get("labels", [])
                    all_labels = list(set(original_labels + ["Entity"]))
                    label_str = "".join([f":{label}" for label in all_labels])
                    properties = node.get("properties", {})
                    properties["chunk_id"] = chunk_id

                    query = f"""
                    MERGE (n {{node_id: $node_id}})
                    SET n{label_str}
                    SET n += $properties
                    """
                    tx.run(query, node_id=node_id, properties=properties)

                # 2. 创建或更新所有关系
                for rel in relationships:
                    rel_id = rel["relationship_id"]
                    rel_type = rel["type"]
                    start_id = rel["start_node_id"]
                    end_id = rel["end_node_id"]
                    rel_props = rel.get("properties", {})
                    # 添加来源 chunk_id
                    rel_props["chunk_id"] = chunk_id

                    # 调试：打印详细信息
                    print(f"准备写入关系: {rel_id}")
                    print(f"  start_id: {start_id}")
                    print(f"  end_id: {end_id}")
                    print(f"  rel_type: {rel_type}")
                    print(f"  rel_props: {rel_props}")

                    # 如果 rel_props 为空，打印警告
                    if not rel_props:
                        print(f"警告: 关系 {rel_id} 的 properties 为空")

                    query = f"""
                    MATCH (a {{node_id: $start_id}})
                    MATCH (b {{node_id: $end_id}})
                    MERGE (a)-[r:{rel_type} {{relationship_id: $rel_id}}]->(b)
                    SET r += $rel_props
                    """
                    tx.run(query, start_id=start_id, end_id=end_id,
                           rel_id=rel_id, rel_props=rel_props)

                tx.commit()
                logger.info(f"Chunk {chunk_id} 写入成功: {len(nodes)} 节点, {len(relationships)} 关系")
                return True
            except Exception as e:
                tx.rollback()
                logger.error(f"Chunk {chunk_id} 写入失败: {e}")
                return False


# ========== 消息反序列化 ==========
def deserialize_message(msg_data):
    processed = {}
    for k, v in msg_data.items():
        if k in ["vector", "keywords_text", "ner_entities_text", "knowledge_graph"]:
            try:
                processed[k] = json.loads(v) if isinstance(v, str) else v
            except json.JSONDecodeError:
                processed[k] = v
        elif k in ["chunk_seq", "content_length", "dialog_index", "turn_count",
                   "wizard_eval_score", "total_retrieved_passages", "total_retrieved_topics", "retry_count"]:
            processed[k] = int(v) if v else 0
        elif k in ["evidence_density"]:
            processed[k] = float(v) if v else 0.0
        elif k in ["has_checked_evidence"]:
            processed[k] = v in ("True", "true", True)
        else:
            processed[k] = v
    return processed


# ========== 主循环 ==========
def main():
    r = connect_redis()
    ensure_consumer_group(r)

    neo4j_client = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    logger.info("Neo4j 连接成功，消费者启动，等待消息...")

    while True:
        try:
            results = r.xreadgroup(
                groupname=CONSUMER_GROUP,
                consumername=CONSUMER_NAME,
                streams={REDIS_STREAM: '>'},
                count=BATCH_SIZE,
                block=BLOCK_MS
            )
            if not results:
                continue

            for stream, messages in results:
                for msg_id, msg_data in messages:
                    processed = deserialize_message(msg_data)
                    chunk_id = processed.get("chunk_id")
                    if not chunk_id:
                        logger.error("消息缺少 chunk_id，直接 ACK")
                        r.xack(REDIS_STREAM, CONSUMER_GROUP, msg_id)
                        continue

                    kg = processed.get("knowledge_graph")
                    if not kg:
                        logger.warning(f"消息 {chunk_id} 无 knowledge_graph 字段，跳过")
                        r.xack(REDIS_STREAM, CONSUMER_GROUP, msg_id)
                        continue

                    # 调试：打印第一条关系的属性（可选）
                    if chunk_id == "wow_dataset_doc_0_0_d37d3663" and kg.get("relationships"):
                        logger.info(f"第一条关系属性示例: {kg['relationships'][0].get('properties')}")

                    retry_count = processed.get("retry_count", 0)
                    success = neo4j_client.write_knowledge_graph(chunk_id, kg)

                    if success:
                        r.xack(REDIS_STREAM, CONSUMER_GROUP, msg_id)
                    else:
                        new_retry = retry_count + 1
                        if new_retry > MAX_RETRIES:
                            logger.warning(f"消息 {msg_id} 重试超过 {MAX_RETRIES} 次，移入死信队列")
                            r.xadd(REDIS_DLQ, msg_data)
                            r.xack(REDIS_STREAM, CONSUMER_GROUP, msg_id)
                        else:
                            msg_data["retry_count"] = str(new_retry)
                            r.xadd(REDIS_STREAM, msg_data)
                            r.xack(REDIS_STREAM, CONSUMER_GROUP, msg_id)
                            logger.info(f"消息 {msg_id} 重试 {new_retry} 次，已重新加入队列")

        except KeyboardInterrupt:
            logger.info("消费者被手动停止")
            break
        except Exception as e:
            logger.exception(f"主循环异常: {e}")
            time.sleep(5)

    neo4j_client.close()


if __name__ == "__main__":
    main()

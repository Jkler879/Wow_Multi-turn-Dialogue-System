"""
消息队列 - 生产者（Producer）：
读取已处理好的数据文件（包含知识库数据和知识图谱数据），
向量化后推入 Redis Stream。
"""

import json
import redis
import ollama
from tqdm import tqdm
from pathlib import Path
from config.paths import paths  # 你的路径配置，如无则直接指定路径

# ========== 配置 ==========
REDIS_HOST = "localhost"
REDIS_PORT = 6380
REDIS_STREAM = "knowledge_pipeline"
EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 50  # 每批推送的消息数量，避免一次性加载过大
INPUT_FILE = paths.processed_data / "chunks" / "wow_ner_keywords_merged.jsonl"
KG_INPUT_FILE = paths.processed_data / "chunks" / "wow_knowledge_graphs_batch_100.json"


def connect_redis(host, port):
    """连接 Redis，测试连通性"""
    r = redis.Redis(host=host, port=port, decode_responses=True)
    try:
        r.ping()
        print("Redis 连接成功")
        return r
    except redis.ConnectionError:
        print("无法连接到 Redis，请检查服务")
        raise


def extract_keywords_text(keywords_list):
    """提取关键词文本列表（去重）"""
    texts = []
    for kw in keywords_list:
        if kw.get("normalized_text"):
            texts.append(kw["normalized_text"])
        elif kw.get("text"):
            texts.append(kw["text"].replace(" ", "_"))
    return list(set(texts))


def extract_entities_text(entities_list):
    """提取实体文本列表（去重）"""
    texts = []
    for ent in entities_list:
        if ent.get("text"):
            texts.append(ent["text"])
    return list(set(texts))


def generate_embedding(text, model=EMBED_MODEL):
    """生成向量（使用 search_document 前缀）"""
    prompt = f"search_document: {text}"
    resp = ollama.embeddings(model=model, prompt=prompt)
    return resp["embedding"]


def read_kg_items(file_path):
    """读取知识图谱 JSON 文件，逐条返回条目（每个条目包含 chunk_id 和 knowledge_graph）"""
    with open(file_path, 'r', encoding='utf-8') as f:
        # 支持 JSON 数组或 JSON Lines
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
            for item in data:
                yield item
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _ensure_redis_compatible(value):
    """将值转换为 Redis 可接受的类型（字符串、整数、浮点数）"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, (list, dict)):
        # 理论上这些应该在之前已被 json.dumps 处理，但若遗漏则转为 JSON 字符串
        return json.dumps(value)
    return value  # 字符串原样返回


def build_message(item):
    """
    根据 item 类型构建消息：
    - 如果 item 包含 "content" 和 "metadata"，则视为完整数据（用于 Milvus + Neo4j）
    - 如果 item 包含 "chunk_id" 和 "knowledge_graph"，则视为纯知识图谱数据（仅用于 Neo4j）
    """
    msg = {}

    # 公共字段：chunk_id
    if "chunk_id" in item:
        msg["chunk_id"] = item["chunk_id"]
    elif "metadata" in item and "chunk_id" in item["metadata"]:
        msg["chunk_id"] = item["metadata"]["chunk_id"]
    else:
        raise ValueError("条目缺少 chunk_id")

    # 处理知识图谱字段（如果存在）
    if "knowledge_graph" in item:
        msg["knowledge_graph"] = json.dumps(item["knowledge_graph"])
    elif "metadata" in item and "knowledge_graph" in item["metadata"]:
        # 兼容旧结构（如果 knowledge_graph 在 metadata 中）
        msg["knowledge_graph"] = json.dumps(item["metadata"]["knowledge_graph"])
    else:
        msg["knowledge_graph"] = json.dumps({})  # 空图谱

    # 如果是完整数据（含 content），则进行向量化并填充其他字段
    if "content" in item:
        content = item["content"]
        msg["content"] = content
        # 向量化
        msg["vector"] = json.dumps(generate_embedding(content))
        # 填充元数据字段
        metadata = item.get("metadata", {})
        for field in ["source", "document_id", "chunk_seq", "content_length", "content_hash",
                      "dialog_id", "dialog_index", "topic", "turn_count", "speaker_role",
                      "wizard_eval_score", "has_checked_evidence", "total_retrieved_passages",
                      "total_retrieved_topics", "evidence_density", "prev_chunk_id",
                      "next_chunk_id", "split_type"]:
            msg[field] = metadata.get(field)
        # 处理数组字段
        keywords_list = metadata.get("keywords", [])
        msg["keywords_text"] = json.dumps(extract_keywords_text(keywords_list))
        msg["keywords_json"] = json.dumps(keywords_list)
        entities_list = metadata.get("ner_entities", [])
        msg["ner_entities_text"] = json.dumps(extract_entities_text(entities_list))
        msg["ner_entities_json"] = json.dumps(entities_list)
        msg["relations_json"] = json.dumps(metadata.get("relations", []))
    else:
        # 纯知识图谱数据：其他字段置为空或默认值（避免消费者报错）
        msg["content"] = ""
        msg["vector"] = json.dumps([])  # 空向量
        # 标量字段设为 None 或空字符串
        for field in ["source", "document_id", "chunk_seq", "content_length", "content_hash",
                      "dialog_id", "dialog_index", "topic", "turn_count", "speaker_role",
                      "wizard_eval_score", "has_checked_evidence", "total_retrieved_passages",
                      "total_retrieved_topics", "evidence_density", "prev_chunk_id",
                      "next_chunk_id", "split_type"]:
            msg[field] = None
        # 数组字段置为空列表
        msg["keywords_text"] = json.dumps([])
        msg["ner_entities_text"] = json.dumps([])
        msg["keywords_json"] = json.dumps([])
        msg["ner_entities_json"] = json.dumps([])
        msg["relations_json"] = json.dumps([])

    msg["retry_count"] = 0

    # 确保所有值都是 Redis 可接受的类型
    msg = {k: _ensure_redis_compatible(v) for k, v in msg.items()}
    return msg


def read_items(file_path):
    """
    读取文件，自动识别 JSON 数组或 JSON Lines 格式。
    返回生成器，每次 yield 一个数据项（字典）。
    """
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        # 读取第一个非空字符，判断是否为数组
        first_char = None
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_char = ch
                break
        f.seek(0)  # 重置指针

        if first_char == '[':
            # JSON 数组格式
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    yield item
            else:
                raise ValueError("文件根元素是数组，但解析后不是列表")
        else:
            # JSON Lines 格式
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def push_batch(redis_client, stream_name, batch):
    """将一批消息推入 Redis Stream"""
    for msg in batch:
        redis_client.xadd(stream_name, msg, maxlen=100000)


def process_and_push(redis_client, items, data_type):
    """通用处理并推送函数"""
    batch = []
    for idx, item in enumerate(tqdm(items, desc=f"处理 {data_type}")):
        try:
            msg = build_message(item)
            batch.append(msg)
            if len(batch) >= BATCH_SIZE:
                push_batch(redis_client, REDIS_STREAM, batch)
                batch = []
        except Exception as e:
            # 打印失败信息，继续处理下一条
            chunk_id = item.get("chunk_id") or item.get("metadata", {}).get("chunk_id", "unknown")
            print(f"处理失败 {chunk_id}: {e}")
            continue
    if batch:
        push_batch(redis_client, REDIS_STREAM, batch)
    print(f"{data_type} 推送完成")


def main():
    r = connect_redis(REDIS_HOST, REDIS_PORT)

    # 处理原有合并文件（含 content）
    if INPUT_FILE.exists():
        print(f"处理合并数据文件: {INPUT_FILE}")
        items = list(read_items(INPUT_FILE))
        process_and_push(r, items, "合并数据")
    else:
        print(f"合并文件不存在: {INPUT_FILE}")

    # 处理知识图谱文件
    kg_path = Path(KG_INPUT_FILE)
    if kg_path.exists():
        print(f"处理知识图谱文件: {kg_path}")
        kg_items = list(read_kg_items(kg_path))
        process_and_push(r, kg_items, "知识图谱数据")
    else:
        print(f"知识图谱文件不存在: {kg_path}")


if __name__ == "__main__":
    main()
import json
from collections import defaultdict, Counter
from typing import Dict, Any, List, Set
from pathlib import Path


def analyze_wow_dataset_fields(data_path: str, sample_size: int = 10) -> Dict[str, Any]:
    """
    深度分析WoW数据集中的所有字段结构

    Args:
        data_path: 数据文件路径
        sample_size: 分析的样本数量

    Returns:
        字段结构分析报告
    """
    print(f"🔍 开始分析WoW数据集字段结构: {data_path}")

    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 限制样本数量
    sample_data = data[:sample_size]

    # 存储字段分析结果
    field_analysis = {
        "root_level_fields": defaultdict(set),  # 根级别字段
        "dialog_turn_fields": defaultdict(set),  # 对话轮次字段
        "nested_structures": defaultdict(list),  # 嵌套结构
        "field_value_samples": defaultdict(list),  # 字段值样本
        "field_frequency": defaultdict(int),  # 字段出现频率
        "data_types": defaultdict(set)  # 数据类型
    }

    print(f"📊 分析 {len(sample_data)} 条样本数据...")

    for i, item in enumerate(sample_data):
        print(f"  分析第 {i + 1} 条数据...")

        # 分析根级别字段
        analyze_root_level_fields(item, field_analysis, i)

        # 分析对话轮次字段
        if 'dialog' in item:
            analyze_dialog_fields(item['dialog'], field_analysis, i)

        # 分析嵌套结构
        analyze_nested_structures(item, field_analysis, i)

    # 生成分析报告
    report = generate_field_analysis_report(field_analysis, len(sample_data))

    return report


def analyze_root_level_fields(item: Dict, field_analysis: Dict, item_index: int):
    """分析根级别字段"""
    for field, value in item.items():
        field_analysis["root_level_fields"][field].add(type(value).__name__)
        field_analysis["field_frequency"][f"root.{field}"] += 1
        field_analysis["data_types"][f"root.{field}"].add(type(value).__name__)

        # 存储样本值（限制长度）
        if len(str(value)) < 100:  # 只存储较短的样本
            field_analysis["field_value_samples"][f"root.{field}"].append(str(value))

        # 特殊处理重要字段
        if field == 'wizard_eval':
            field_analysis["field_value_samples"][f"root.{field}"].append(f"值: {value}")


def analyze_dialog_fields(dialog: List[Dict], field_analysis: Dict, item_index: int):
    """分析对话轮次字段"""
    for turn_index, turn in enumerate(dialog):
        for field, value in turn.items():
            field_key = f"dialog.{field}"
            field_analysis["dialog_turn_fields"][field].add(type(value).__name__)
            field_analysis["field_frequency"][field_key] += 1
            field_analysis["data_types"][field_key].add(type(value).__name__)

            # 存储样本值
            if field in ['checked_sentence', 'retrieved_passages', 'retrieved_topics']:
                sample = str(value)[:200]  # 限制长度
                field_analysis["field_value_samples"][field_key].append(sample)

            # 分析检索段落结构
            if field in ['retrieved_passages', 'retrieved_docs', 'evidence_passages']:
                analyze_retrieved_passages_structure(value, field_analysis, item_index, turn_index)


def analyze_retrieved_passages_structure(passages, field_analysis: Dict, item_index: int, turn_index: int):
    """分析检索段落的结构"""
    if not passages or not isinstance(passages, list):
        return

    # 分析第一个段落的结构
    if len(passages) > 0:
        first_passage = passages[0]
        if isinstance(first_passage, dict):
            passage_fields = list(first_passage.keys())
            field_analysis["nested_structures"]["retrieved_passage_fields"].append(passage_fields)

            # 存储样本结构
            if item_index == 0 and turn_index == 0:  # 只存储第一个样本的完整结构
                field_analysis["field_value_samples"]["retrieved_passage_structure"] = first_passage


def analyze_nested_structures(item: Dict, field_analysis: Dict, item_index: int):
    """分析嵌套数据结构"""
    # 检查是否有嵌套的列表或字典
    for field, value in item.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            # 这是一个对象列表
            field_analysis["nested_structures"][f"root.{field}"] = list(value[0].keys())
        elif isinstance(value, dict):
            # 这是一个嵌套字典
            field_analysis["nested_structures"][f"root.{field}"] = list(value.keys())


def generate_field_analysis_report(field_analysis: Dict, total_items: int) -> Dict[str, Any]:
    """生成字段分析报告"""
    report = {
        "summary": {
            "total_items_analyzed": total_items,
            "total_root_fields": len(field_analysis["root_level_fields"]),
            "total_dialog_fields": len(field_analysis["dialog_turn_fields"]),
        },
        "root_level_fields": {},
        "dialog_turn_fields": {},
        "field_frequency": {},
        "data_types": {},
        "field_samples": {},
        "recommendations": []
    }

    # 处理根级别字段
    for field, types in field_analysis["root_level_fields"].items():
        frequency = field_analysis["field_frequency"][f"root.{field}"]
        report["root_level_fields"][field] = {
            "data_types": list(types),
            "frequency": frequency,
            "frequency_percentage": (frequency / total_items) * 100,
            "samples": field_analysis["field_value_samples"].get(f"root.{field}", [])[:3]  # 前3个样本
        }

    # 处理对话轮次字段
    for field, types in field_analysis["dialog_turn_fields"].items():
        frequency = field_analysis["field_frequency"][f"dialog.{field}"]
        total_turns = sum(len(item.get('dialog', [])) for item in [{}] * total_items)  # 简化计算

        report["dialog_turn_fields"][field] = {
            "data_types": list(types),
            "frequency": frequency,
            "samples": field_analysis["field_value_samples"].get(f"dialog.{field}", [])[:3]
        }

    # 字段频率排序
    report["field_frequency"] = dict(sorted(
        field_analysis["field_frequency"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:20])  # 只显示前20个

    # 生成建议
    report["recommendations"] = generate_recommendations(field_analysis, total_items)

    return report


def generate_recommendations(field_analysis: Dict, total_items: int) -> List[str]:
    """基于分析结果生成建议"""
    recommendations = []

    # 检查关键字段是否存在
    key_fields = [
        "dialog", "chosen_topic", "wizard_eval",
        "checked_sentence", "retrieved_passages", "retrieved_topics"
    ]

    for field in key_fields:
        if field in field_analysis["root_level_fields"]:
            recommendations.append(f"✅ 根级别字段 '{field}' 存在")
        elif f"dialog.{field}" in field_analysis["field_frequency"]:
            recommendations.append(f"✅ 对话轮次字段 '{field}' 存在")
        else:
            recommendations.append(f"❌ 关键字段 '{field}' 未找到")

    # 检查检索段落结构
    if "retrieved_passage_structure" in field_analysis["field_value_samples"]:
        structure = field_analysis["field_value_samples"]["retrieved_passage_structure"]
        recommendations.append(f"📖 检索段落结构: {list(structure.keys())}")

    # 检查字段完整性
    root_fields_count = len(field_analysis["root_level_fields"])
    dialog_fields_count = len(field_analysis["dialog_turn_fields"])

    recommendations.append(f"📊 发现 {root_fields_count} 个根级别字段和 {dialog_fields_count} 个对话轮次字段")

    return recommendations


def save_analysis_report(report: Dict[str, Any], output_path: str):
    """保存分析报告"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"📄 分析报告已保存到: {output_path}")


def print_summary_report(report: Dict[str, Any]):
    """打印摘要报告"""
    print("\n" + "=" * 80)
    print("🎯 WoW数据集字段分析摘要报告")
    print("=" * 80)

    summary = report["summary"]
    print(f"分析数据量: {summary['total_items_analyzed']} 条")
    print(f"根级别字段: {summary['total_root_fields']} 个")
    print(f"对话轮次字段: {summary['total_dialog_fields']} 个")

    print("\n📋 关键字段检查:")
    for rec in report["recommendations"]:
        print(f"  {rec}")

    print("\n🏆 最常出现的字段:")
    for field, freq in list(report["field_frequency"].items())[:10]:
        print(f"  {field}: {freq} 次")

    print("\n📝 根级别字段详情:")
    for field, info in report["root_level_fields"].items():
        print(f"  {field}: {info['data_types']} (出现 {info['frequency']} 次, {info['frequency_percentage']:.1f}%)")

    print("\n💬 对话轮次字段详情:")
    for field, info in report["dialog_turn_fields"].items():
        print(f"  {field}: {info['data_types']} (出现 {info['frequency']} 次)")


# 使用示例
if __name__ == "__main__":
    # 配置路径
    data_path = r"C:\Users\么么哒吟\Desktop\pythonProject\data\raw\train_100.json"  # 替换为您的实际路径
    output_path = r"C:\Users\么么哒吟\Desktop\pythonProject\data\raw\wow_field_analysis_report.json"

    try:
        # 运行分析
        report = analyze_wow_dataset_fields(data_path, sample_size=100)

        # 保存详细报告
        save_analysis_report(report, output_path)

        # 打印摘要
        print_summary_report(report)

        print(f"\n🎉 分析完成！详细报告已保存到: {output_path}")

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback

        traceback.print_exc()
# wow_advanced_analysis.py
import json
import logging
from typing import List, Dict, Any
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from langchain_core.documents import Document

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wow_visualization.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WoWDataAnalyzer:
    """WoW数据集分析器（简化版，用于可视化）"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.raw_data = []

    def load_data(self):
        """加载数据"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            logger.info(f"✅ 加载了 {len(self.raw_data)} 条数据")
            return True
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            return False

    def analyze_distribution(self):
        """分析数据分布"""
        if not self.raw_data:
            logger.error("❌ 没有数据可分析")
            return None

        # 主题分布
        topics = [item.get('chosen_topic', '未知') for item in self.raw_data]
        topic_counter = Counter(topics)

        # 对话轮次分布
        turn_counts = [len(item.get('dialog', [])) for item in self.raw_data]
        turn_stats = {
            '总数': len(turn_counts),
            '平均轮次': sum(turn_counts) / len(turn_counts),
            '最大轮次': max(turn_counts),
            '最小轮次': min(turn_counts),
            '轮次分布': dict(Counter(turn_counts))
        }

        # 创建语义分块并分析尺寸
        documents = self.create_semantic_chunks()
        chunk_sizes = [len(doc.page_content) for doc in documents]

        chunk_quality = {
            '总块数': len(documents),
            '平均字符数': sum(chunk_sizes) / len(chunk_sizes),
            '最大块大小': max(chunk_sizes),
            '最小块大小': min(chunk_sizes),
            'chunk_sizes': chunk_sizes  # 添加实际尺寸数据
        }

        return {
            'topics': topic_counter,
            'turns': turn_stats,
            'chunk_quality': chunk_quality,
            'documents': documents
        }

    def create_semantic_chunks(self) -> List[Document]:
        """创建语义分块"""
        documents = []

        for i, item in enumerate(self.raw_data):
            content = self._build_chunk_content(item)
            metadata = self._build_chunk_metadata(item, i)

            documents.append(Document(
                page_content=content,
                metadata=metadata
            ))

        return documents

    def _build_chunk_content(self, item: Dict) -> str:
        """构建块内容"""
        content_parts = []

        content_parts.append(f"主题: {item.get('chosen_topic', '未知')}")
        content_parts.append(f"角色: {item.get('persona', '')}")
        content_parts.append("")

        for j, turn in enumerate(item.get('dialog', [])):
            speaker = "助手" if turn['speaker'] == "0_Wizard" else "用户"
            content_parts.append(f"{speaker}: {turn['text']}")

            if turn.get('checked_sentence'):
                for evidence in turn['checked_sentence'].values():
                    content_parts.append(f"  [知识支持] {evidence}")

            if turn.get('retrieved_topics'):
                topics = "、".join(turn['retrieved_topics'][:3])
                content_parts.append(f"  [相关主题] {topics}")

        return "\n".join(content_parts)

    def _build_chunk_metadata(self, item: Dict, index: int) -> Dict[str, Any]:
        """构建块元数据"""
        dialog = item.get('dialog', [])
        return {
            "source": "wow_dataset",
            "dialog_id": f"dialog_{index}",
            "topic": item.get('chosen_topic', 'unknown'),
            "turn_count": len(dialog),
        }


class WowVisualAnalyzer:
    """WoW数据可视化分析"""

    def __init__(self, analyzer: WoWDataAnalyzer = None):
        self.analyzer = analyzer

    def create_visualizations(self, results: Dict):
        """创建可视化图表"""
        try:
            plt.style.use('default')

            # 1. 主题分布图
            self._plot_topic_distribution(results['topics'])

            # 2. 对话轮次分布图
            self._plot_turn_distribution(results['turns'])

            # 3. 块尺寸分布图
            self._plot_chunk_size_distribution(results['chunk_quality'])

            # 4. 词云
            self._create_word_cloud(results['documents'])

            logger.info("🎨 所有可视化图表已生成！")

        except Exception as e:
            logger.error(f"❌ 可视化生成失败: {e}")

    def _plot_topic_distribution(self, topic_counter):
        """绘制主题分布图"""
        try:
            top_topics = topic_counter.most_common(15)
            if not top_topics:
                logger.warning("⚠️ 没有主题数据可绘制")
                return

            topics, counts = zip(*top_topics)

            plt.figure(figsize=(12, 8))
            plt.barh(range(len(topics)), counts)
            plt.yticks(range(len(topics)), topics)
            plt.xlabel('对话数量')
            plt.title('Top 15 主题分布')
            plt.tight_layout()
            plt.savefig('topic_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("📊 主题分布图已保存为 topic_distribution.png")

        except Exception as e:
            logger.error(f"❌ 主题分布图生成失败: {e}")

    def _plot_turn_distribution(self, turn_stats):
        """绘制轮次分布图"""
        try:
            turn_dist = turn_stats.get('轮次分布', {})
            if not turn_dist:
                logger.warning("⚠️ 没有轮次数据可绘制")
                return

            plt.figure(figsize=(10, 6))
            plt.bar(list(turn_dist.keys()), list(turn_dist.values()))
            plt.xlabel('对话轮次')
            plt.ylabel('对话数量')
            plt.title('对话轮次分布')
            plt.tight_layout()
            plt.savefig('turn_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("📊 轮次分布图已保存为 turn_distribution.png")

        except Exception as e:
            logger.error(f"❌ 轮次分布图生成失败: {e}")

    def _plot_chunk_size_distribution(self, chunk_quality):
        """绘制块尺寸分布图"""
        try:
            sizes = chunk_quality.get('chunk_sizes', [])
            if not sizes:
                logger.warning("⚠️ 没有块尺寸数据可绘制")
                return

            plt.figure(figsize=(10, 6))
            plt.hist(sizes, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
            plt.xlabel('块尺寸（字符数）')
            plt.ylabel('频率')
            plt.title('块尺寸分布')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('chunk_size_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("📊 块尺寸分布图已保存为 chunk_size_distribution.png")

        except Exception as e:
            logger.error(f"❌ 块尺寸分布图生成失败: {e}")

    def _create_word_cloud(self, documents):
        """创建词云"""
        try:
            if not documents:
                logger.warning("⚠️ 没有文档数据可生成词云")
                return

            all_text = " ".join([doc.page_content for doc in documents])

            if not all_text.strip():
                logger.warning("⚠️ 文本内容为空，无法生成词云")
                return

            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                font_path=None,  # 使用默认字体，如果需要中文可以指定字体路径
                collocations=False
            ).generate(all_text)

            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('WoW数据集词云')
            plt.tight_layout()
            plt.savefig('word_cloud.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("☁️ 词云图已保存为 word_cloud.png")

        except Exception as e:
            logger.error(f"❌ 词云生成失败: {e}")


def generate_complete_analysis(data_path: str):
    """生成完整的分析和可视化"""
    logger.info("🚀 开始WoW数据集完整分析...")

    # 1. 数据分析
    analyzer = WoWDataAnalyzer(data_path)
    if not analyzer.load_data():
        return

    results = analyzer.analyze_distribution()
    if not results:
        logger.error("❌ 数据分析失败")
        return

    # 2. 打印基础统计
    logger.info("\n📋 基础统计信息:")
    logger.info(f"   总对话数: {len(analyzer.raw_data)}")
    logger.info(f"   主题数量: {len(results['topics'])}")
    logger.info(f"   平均对话轮次: {results['turns']['平均轮次']:.2f}")
    logger.info(f"   平均块尺寸: {results['chunk_quality']['平均字符数']:.0f} 字符")

    # 3. 可视化
    visual_analyzer = WowVisualAnalyzer(analyzer)
    visual_analyzer.create_visualizations(results)

    return results


def main():
    """主函数"""
    data_path = r"C:\Users\么么哒吟\Desktop\pythonProject\data\train_100.json"

    try:
        results = generate_complete_analysis(data_path)
        if results:
            logger.info("✅ 分析和可视化完成！")

            # 显示生成的文件
            import os
            generated_files = [
                'topic_distribution.png',
                'turn_distribution.png',
                'chunk_size_distribution.png',
                'word_cloud.png',
                'wow_visualization.log'
            ]

            logger.info("\n📁 生成的文件:")
            for file in generated_files:
                if os.path.exists(file):
                    logger.info(f"   ✅ {file}")
                else:
                    logger.warning(f"   ❌ {file} 未生成")

    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")


# 兼容性检查
def check_dependencies():
    """检查依赖是否安装"""
    required_libs = {
        'matplotlib': '可视化库',
        'seaborn': '统计图表',
        'wordcloud': '词云生成'
    }

    missing_libs = []
    for lib, desc in required_libs.items():
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append((lib, desc))

    if missing_libs:
        logger.warning("⚠️ 缺少以下依赖库:")
        for lib, desc in missing_libs:
            logger.warning(f"   {lib} - {desc}")
        logger.info("💡 安装命令: pip install matplotlib seaborn wordcloud")
        return False

    logger.info("✅ 所有依赖库已安装")
    return True


if __name__ == "__main__":
    # 检查依赖
    if check_dependencies():
        main()
    else:
        logger.error("❌ 请先安装缺失的依赖库")

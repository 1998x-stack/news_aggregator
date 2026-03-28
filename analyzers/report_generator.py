#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告生成模块

生成多种格式的新闻分析报告:
- Markdown格式报告
- JSON格式数据
- HTML格式报告

作者: Claude
日期: 2024-12
"""

import sys
import json
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from analyzers.trend_analyzer import TrendReport, TrendItem
from analyzers.executive_summarizer import ExecutiveSummarizer, ExecutiveSummary
from config.settings import ContentCategory, IMPORTANCE_LEVELS


@dataclass
class ReportConfig:
    """报告配置"""

    title: str = "新闻聚合分析报告"
    subtitle: str = ""
    author: str = "AI News Aggregator"
    language: str = "zh"  # zh/en
    include_toc: bool = True  # 目录
    include_summary: bool = True  # 执行摘要
    include_charts: bool = False  # 图表（仅HTML）
    max_items_per_section: int = 10
    output_dir: str = "outputs"


class ReportGenerator:
    """
    报告生成器

    支持多种输出格式和自定义模板

    Attributes:
        config: 报告配置
    """

    def __init__(self, config: ReportConfig = None):
        """
        初始化报告生成器

        Args:
            config: 报告配置
        """
        self.config = config or ReportConfig()
        self.executive_summarizer = ExecutiveSummarizer()

        # 确保输出目录存在
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"ReportGenerator初始化完成, 输出目录: {self.config.output_dir}")

    def generate_daily_report(
        self,
        trend_report: TrendReport,
        articles: List[Dict[str, Any]],
        date: str = None,
        format: str = "markdown",
    ) -> str:
        """
        生成每日报告

        Args:
            trend_report: 趋势分析报告
            articles: 文章列表
            date: 报告日期
            format: 输出格式 (markdown/json/html)

        Returns:
            生成的报告文件路径
        """
        date = date or datetime.now().strftime("%Y-%m-%d")

        if format == "markdown":
            return self._generate_markdown_report(trend_report, articles, date)
        elif format == "json":
            return self._generate_json_report(trend_report, articles, date)
        elif format == "html":
            return self._generate_html_report(trend_report, articles, date)
        else:
            raise ValueError(f"不支持的格式: {format}")

    def _generate_markdown_report(
        self, trend_report: TrendReport, articles: List[Dict[str, Any]], date: str
    ) -> str:
        """生成Markdown格式报告"""
        lines = []

        # 标题
        lines.append(f"# {self.config.title}")
        if self.config.subtitle:
            lines.append(f"## {self.config.subtitle}")
        lines.append("")
        lines.append(f"**日期**: {date}")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**分析文章数**: {trend_report.total_articles}")
        lines.append("")

        # 目录
        if self.config.include_toc:
            lines.append("## 目录")
            lines.append("")
            lines.append("1. [执行摘要](#执行摘要)")
            lines.append("2. [热点话题](#热点话题)")
            lines.append("3. [重要新闻](#重要新闻)")
            lines.append("4. [行业动态](#行业动态)")
            lines.append("5. [趋势展望](#趋势展望)")
            lines.append("6. [数据统计](#数据统计)")
            lines.append("")

        # 执行摘要
        if self.config.include_summary:
            lines.append("## 执行摘要")
            lines.append("")

            try:
                # Generate AI-powered executive summary
                executive_summary = self.executive_summarizer.generate_summary(
                    trend_report, articles, date
                )

                # Key insights
                if executive_summary.key_insights:
                    lines.append("### 🔍 关键洞察")
                    lines.append("")
                    for insight in executive_summary.key_insights:
                        lines.append(f"- {insight}")
                    lines.append("")

                # Market impact
                if executive_summary.market_impact:
                    lines.append("### 📊 市场影响")
                    lines.append("")
                    lines.append(executive_summary.market_impact)
                    lines.append("")

                # Recommended actions
                if executive_summary.recommended_actions:
                    lines.append("### ✅ 建议行动")
                    lines.append("")
                    for action in executive_summary.recommended_actions:
                        lines.append(f"- {action}")
                    lines.append("")

                # Opportunities
                if executive_summary.opportunities:
                    lines.append("### 🚀 机会")
                    lines.append("")
                    for opportunity in executive_summary.opportunities:
                        lines.append(f"- {opportunity}")
                    lines.append("")

                # Risk factors
                if executive_summary.risk_factors:
                    lines.append("### ⚠️ 风险因素")
                    lines.append("")
                    for risk in executive_summary.risk_factors:
                        lines.append(f"- {risk}")
                    lines.append("")

            except Exception as e:
                logger.error(f"Error generating executive summary: {e}")
                # Fallback to basic summary
                if trend_report.hot_topics:
                    top_topics = [t.name for t in trend_report.hot_topics[:3]]
                    lines.append(
                        f"今日热点集中在: **{', '.join(top_topics)}** 等领域。"
                    )

                if trend_report.outlook:
                    lines.append(f"\n{trend_report.outlook}")

                if trend_report.recommendations:
                    lines.append("\n**建议关注**:")
                    for rec in trend_report.recommendations[:3]:
                        lines.append(f"- {rec}")

            lines.append("")

        # 热点话题
        lines.append("## 热点话题")
        lines.append("")

        if trend_report.hot_topics:
            lines.append("| 排名 | 话题 | 热度 | 提及次数 | 类别 |")
            lines.append("|------|------|------|----------|------|")

            for i, topic in enumerate(
                trend_report.hot_topics[: self.config.max_items_per_section], 1
            ):
                heat_bar = "🔥" * min(5, int(topic.heat_score / 20))
                lines.append(
                    f"| {i} | {topic.name} | {heat_bar} {topic.heat_score:.0f} | {topic.mention_count} | {topic.category} |"
                )

            lines.append("")
        else:
            lines.append("*暂无热点话题数据*")
            lines.append("")

        # 新兴趋势
        if trend_report.emerging_trends:
            lines.append("### 新兴趋势")
            lines.append("")
            for trend in trend_report.emerging_trends[:5]:
                lines.append(
                    f"- **{trend.name}**: {trend.summary or f'热度 {trend.heat_score:.0f}'}"
                )
            lines.append("")

        # 重要新闻
        lines.append("## 重要新闻")
        lines.append("")

        # 按重要性排序
        important_articles = sorted(
            articles, key=lambda x: x.get("importance", 0), reverse=True
        )[: self.config.max_items_per_section]

        if important_articles:
            for article in important_articles:
                title = article.get("title", "无标题")
                source = article.get("source", "未知来源")
                importance = article.get("importance", 3)
                category = article.get("category", "other")
                if isinstance(category, ContentCategory):
                    category = category.value

                importance_label = IMPORTANCE_LEVELS.get(importance, {}).get(
                    "label", ""
                )

                lines.append(f"### {title}")
                lines.append("")
                lines.append(f"- **来源**: {source}")
                lines.append(f"- **类别**: {category}")
                lines.append(f"- **重要性**: {'⭐' * importance} ({importance_label})")

                if article.get("summary"):
                    lines.append(f"- **摘要**: {article['summary']}")

                if article.get("url"):
                    lines.append(f"- **链接**: [{title}]({article['url']})")

                lines.append("")
        else:
            lines.append("*暂无重要新闻*")
            lines.append("")

        # 行业动态
        lines.append("## 行业动态")
        lines.append("")

        if trend_report.industry_dynamics:
            for dynamic in trend_report.industry_dynamics:
                sentiment_emoji = {
                    "positive": "📈",
                    "negative": "📉",
                    "neutral": "➡️",
                }.get(dynamic.sentiment, "➡️")
                lines.append(f"### {dynamic.industry} {sentiment_emoji}")
                lines.append("")

                if dynamic.key_events:
                    lines.append("**关键事件**:")
                    for event in dynamic.key_events[:3]:
                        lines.append(f"- {event}")

                if dynamic.major_players:
                    lines.append(
                        f"\n**主要参与者**: {', '.join(dynamic.major_players[:5])}"
                    )

                if dynamic.outlook:
                    lines.append(f"\n**展望**: {dynamic.outlook}")

                lines.append("")
        else:
            lines.append("*暂无行业动态数据*")
            lines.append("")

        # 关键事件
        if trend_report.key_events:
            lines.append("### 关键事件时间线")
            lines.append("")
            for event in trend_report.key_events[:10]:
                time_str = event.get("time", "")[:10] if event.get("time") else ""
                lines.append(
                    f"- **[{time_str}]** {event.get('title', '')} ({event.get('source', '')})"
                )
            lines.append("")

        # 趋势展望
        lines.append("## 趋势展望")
        lines.append("")

        if trend_report.outlook:
            lines.append(trend_report.outlook)
        else:
            lines.append("*基于当前数据，以下是值得关注的趋势:*")
            if trend_report.hot_topics:
                lines.append(f"\n1. {trend_report.hot_topics[0].name} 持续受到关注")
            if trend_report.emerging_trends:
                lines.append(f"2. {trend_report.emerging_trends[0].name} 正在兴起")

        lines.append("")

        # 数据统计
        lines.append("## 数据统计")
        lines.append("")

        # 类别分布
        if trend_report.category_distribution:
            lines.append("### 类别分布")
            lines.append("")
            lines.append("| 类别 | 数量 | 占比 |")
            lines.append("|------|------|------|")

            total = sum(trend_report.category_distribution.values())
            for category, count in sorted(
                trend_report.category_distribution.items(), key=lambda x: -x[1]
            )[:10]:
                pct = count / total * 100 if total > 0 else 0
                lines.append(f"| {category} | {count} | {pct:.1f}% |")

            lines.append("")

        # 来源分布
        if trend_report.source_distribution:
            lines.append("### 来源分布")
            lines.append("")
            lines.append("| 来源 | 数量 |")
            lines.append("|------|------|")

            for source, count in sorted(
                trend_report.source_distribution.items(), key=lambda x: -x[1]
            )[:10]:
                lines.append(f"| {source} | {count} |")

            lines.append("")

        # 页脚
        lines.append("---")
        lines.append("")
        lines.append(f"*报告由 {self.config.author} 自动生成*")
        lines.append(f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        # 写入文件
        content = "\n".join(lines)
        filename = f"{date}_daily_report.md"
        filepath = os.path.join(self.config.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Markdown报告已生成: {filepath}")
        return filepath

    def _generate_json_report(
        self, trend_report: TrendReport, articles: List[Dict[str, Any]], date: str
    ) -> str:
        """生成JSON格式报告"""
        # 处理文章数据
        processed_articles = []
        for article in articles:
            processed = dict(article)
            # 转换枚举类型
            if isinstance(processed.get("category"), ContentCategory):
                processed["category"] = processed["category"].value
            processed_articles.append(processed)

        report_data = {
            "metadata": {
                "title": self.config.title,
                "date": date,
                "generated_at": datetime.now().isoformat(),
                "total_articles": len(articles),
                "author": self.config.author,
            },
            "trend_analysis": trend_report.to_dict(),
            "articles": processed_articles[:100],  # 限制数量
        }

        filename = f"{date}_daily_report.json"
        filepath = os.path.join(self.config.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        logger.info(f"JSON报告已生成: {filepath}")
        return filepath

    def _generate_html_report(
        self, trend_report: TrendReport, articles: List[Dict[str, Any]], date: str
    ) -> str:
        """生成HTML格式报告"""
        html_template = """<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - {date}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        header .meta {{ opacity: 0.9; }}
        section {{ 
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        section h2 {{ 
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .hot-topic {{ 
            display: flex;
            align-items: center;
            padding: 15px;
            border-bottom: 1px solid #eee;
        }}
        .hot-topic:last-child {{ border-bottom: none; }}
        .hot-topic .rank {{ 
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
            width: 50px;
        }}
        .hot-topic .info {{ flex: 1; }}
        .hot-topic .name {{ font-weight: bold; font-size: 1.1em; }}
        .hot-topic .heat {{ 
            color: #e74c3c;
            font-size: 0.9em;
        }}
        .article-card {{ 
            border: 1px solid #eee;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
        }}
        .article-card h3 {{ color: #333; margin-bottom: 10px; }}
        .article-card .meta {{ color: #666; font-size: 0.9em; }}
        .stats-grid {{ 
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .stat-card {{ 
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card .value {{ font-size: 2em; color: #667eea; font-weight: bold; }}
        .stat-card .label {{ color: #666; }}
        footer {{ text-align: center; color: #666; padding: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <div class="meta">
                <p>📅 {date} | 📊 分析文章数: {total_articles}</p>
                <p>🕐 生成时间: {generated_at}</p>
            </div>
        </header>

        <section>
            <h2>📈 核心数据</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="value">{total_articles}</div>
                    <div class="label">分析文章</div>
                </div>
                <div class="stat-card">
                    <div class="value">{hot_topic_count}</div>
                    <div class="label">热点话题</div>
                </div>
                <div class="stat-card">
                    <div class="value">{category_count}</div>
                    <div class="label">覆盖类别</div>
                </div>
                <div class="stat-card">
                    <div class="value">{source_count}</div>
                    <div class="label">信息来源</div>
                </div>
            </div>
        </section>

        <section>
            <h2>🔥 热点话题</h2>
            {hot_topics_html}
        </section>

        <section>
            <h2>📰 重要新闻</h2>
            {articles_html}
        </section>

        <section>
            <h2>🔮 趋势展望</h2>
            <p>{outlook}</p>
        </section>

        <footer>
            <p>报告由 {author} 自动生成</p>
        </footer>
    </div>
</body>
</html>"""

        # 生成热点话题HTML
        hot_topics_html = ""
        for i, topic in enumerate(trend_report.hot_topics[:10], 1):
            hot_topics_html += f"""
            <div class="hot-topic">
                <div class="rank">#{i}</div>
                <div class="info">
                    <div class="name">{topic.name}</div>
                    <div class="heat">🔥 热度: {topic.heat_score:.0f} | 提及: {topic.mention_count}次 | 类别: {topic.category}</div>
                </div>
            </div>"""

        if not hot_topics_html:
            hot_topics_html = "<p>暂无热点话题数据</p>"

        # 生成文章HTML
        articles_html = ""
        important_articles = sorted(
            articles, key=lambda x: x.get("importance", 0), reverse=True
        )[:10]
        for article in important_articles:
            title = article.get("title", "无标题")
            source = article.get("source", "未知来源")
            importance = article.get("importance", 3)
            category = article.get("category", "other")
            if isinstance(category, ContentCategory):
                category = category.value

            articles_html += f"""
            <div class="article-card">
                <h3>{title}</h3>
                <div class="meta">
                    📌 {source} | 📁 {category} | ⭐ 重要性: {"⭐" * importance}
                </div>
            </div>"""

        if not articles_html:
            articles_html = "<p>暂无重要新闻</p>"

        # 填充模板
        html_content = html_template.format(
            title=self.config.title,
            date=date,
            total_articles=trend_report.total_articles,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            hot_topic_count=len(trend_report.hot_topics),
            category_count=len(trend_report.category_distribution),
            source_count=len(trend_report.source_distribution),
            hot_topics_html=hot_topics_html,
            articles_html=articles_html,
            outlook=trend_report.outlook or "持续关注AI、云计算等领域的发展动态。",
            author=self.config.author,
        )

        filename = f"{date}_daily_report.html"
        filepath = os.path.join(self.config.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML报告已生成: {filepath}")
        return filepath

    def generate_category_report(
        self, articles: List[Dict[str, Any]], category: str, date: str = None
    ) -> str:
        """
        生成特定类别的专题报告

        Args:
            articles: 文章列表
            category: 类别名称
            date: 报告日期

        Returns:
            报告文件路径
        """
        date = date or datetime.now().strftime("%Y-%m-%d")

        # 过滤指定类别的文章
        category_articles = [
            a
            for a in articles
            if str(a.get("category", "")).lower() == category.lower()
        ]

        if not category_articles:
            logger.warning(f"类别 {category} 无文章数据")
            return ""

        lines = []
        lines.append(f"# {category.upper()} 专题报告")
        lines.append("")
        lines.append(f"**日期**: {date}")
        lines.append(f"**文章数**: {len(category_articles)}")
        lines.append("")

        # 按重要性排序
        sorted_articles = sorted(
            category_articles, key=lambda x: x.get("importance", 0), reverse=True
        )

        lines.append("## 重要文章")
        lines.append("")

        for article in sorted_articles[:20]:
            title = article.get("title", "无标题")
            source = article.get("source", "")
            importance = article.get("importance", 3)

            lines.append(f"### {title}")
            lines.append(f"- 来源: {source}")
            lines.append(f"- 重要性: {'⭐' * importance}")
            if article.get("summary"):
                lines.append(f"- 摘要: {article['summary']}")
            lines.append("")

        content = "\n".join(lines)
        filename = f"{date}_{category}_report.md"
        filepath = os.path.join(self.config.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"类别报告已生成: {filepath}")
        return filepath

    def generate_all_formats(
        self,
        trend_report: TrendReport,
        articles: List[Dict[str, Any]],
        date: str = None,
    ) -> Dict[str, str]:
        """
        生成所有格式的报告

        Returns:
            格式到文件路径的映射
        """
        date = date or datetime.now().strftime("%Y-%m-%d")

        results = {}

        for fmt in ["markdown", "json", "html"]:
            try:
                filepath = self.generate_daily_report(trend_report, articles, date, fmt)
                results[fmt] = filepath
            except Exception as e:
                logger.error(f"生成{fmt}报告失败: {e}")
                results[fmt] = ""

        return results


def create_report_generator(config: ReportConfig = None) -> ReportGenerator:
    """工厂函数: 创建报告生成器"""
    return ReportGenerator(config)


# 测试代码
if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    # 测试数据
    test_trend = TrendReport(
        report_time=datetime.now().isoformat(),
        total_articles=50,
        hot_topics=[
            TrendItem(name="GPT-5", category="ai_ml", heat_score=95, mention_count=15),
            TrendItem(
                name="量子计算", category="research", heat_score=80, mention_count=10
            ),
            TrendItem(
                name="云原生", category="cloud_infra", heat_score=70, mention_count=8
            ),
        ],
        category_distribution={
            "ai_ml": 20,
            "cloud_infra": 15,
            "security": 10,
            "other": 5,
        },
        source_distribution={"TechCrunch": 15, "The Verge": 12, "Wired": 10, "36氪": 8},
        outlook="AI领域持续火热，预计未来一周将有更多重大发布。",
    )

    test_articles = [
        {
            "title": "OpenAI发布GPT-5",
            "source": "TechCrunch",
            "category": "ai_ml",
            "importance": 5,
        },
        {
            "title": "谷歌云新功能发布",
            "source": "The Verge",
            "category": "cloud_infra",
            "importance": 4,
        },
        {
            "title": "安全漏洞警报",
            "source": "Wired",
            "category": "security",
            "importance": 4,
        },
    ]

    print("=" * 60)
    print("报告生成测试")
    print("=" * 60)

    generator = create_report_generator()

    # 生成所有格式
    results = generator.generate_all_formats(test_trend, test_articles)

    print("\n生成的报告:")
    for fmt, path in results.items():
        print(f"  {fmt}: {path}")

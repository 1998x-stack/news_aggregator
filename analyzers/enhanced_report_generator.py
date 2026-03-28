"""
Enhanced report generator with interactive elements
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analyzers.report_generator import ReportGenerator, ReportConfig
from analyzers.trend_analyzer import TrendReport
from loguru import logger


class EnhancedReportGenerator(ReportGenerator):
    """Enhanced report generator with interactive elements"""

    def __init__(self, config: ReportConfig = None, enable_collapsible: bool = True):
        super().__init__(config)
        self.enable_collapsible = enable_collapsible

    def _generate_enhanced_markdown_report(
        self, trend_report: TrendReport, articles: List[Dict[str, Any]], date: str
    ) -> str:
        """Generate enhanced Markdown report with interactive elements"""
        lines = []

        # Title
        lines.append(f"# {self.config.title}")
        if self.config.subtitle:
            lines.append(f"## {self.config.subtitle}")
        lines.append("")
        lines.append(f"**日期**: {date}")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**分析文章数**: {trend_report.total_articles}")
        lines.append("")

        # Table of contents with collapsible sections
        if self.config.include_toc:
            lines.append("## 📋 目录")
            lines.append("")
            if self.enable_collapsible:
                lines.append("<details>")
                lines.append("<summary>点击展开目录</summary>")
                lines.append("")
            lines.append("1. [执行摘要](#执行摘要)")
            lines.append("2. [热点话题](#热点话题)")
            lines.append("3. [重要新闻](#重要新闻)")
            lines.append("4. [行业动态](#行业动态)")
            lines.append("5. [趋势展望](#趋势展望)")
            lines.append("6. [数据统计](#数据统计)")
            lines.append("")
            if self.enable_collapsible:
                lines.append("</details>")
            lines.append("")

        # Executive Summary with collapsible sections
        if self.config.include_summary:
            lines.append("## 🎯 执行摘要")
            lines.append("")

            try:
                # Generate AI-powered executive summary
                executive_summary = self.executive_summarizer.generate_summary(
                    trend_report, articles, date
                )

                # Key insights (collapsible)
                if executive_summary.key_insights:
                    lines.append("### 🔍 关键洞察")
                    lines.append("")
                    if self.enable_collapsible:
                        lines.append("<details>")
                        lines.append("<summary>点击查看关键洞察</summary>")
                        lines.append("")
                    for insight in executive_summary.key_insights:
                        lines.append(f"> **{insight}**")
                        lines.append("")
                    if self.enable_collapsible:
                        lines.append("</details>")
                    lines.append("")

                # Market impact (collapsible)
                if executive_summary.market_impact:
                    lines.append("### 📊 市场影响")
                    lines.append("")
                    if self.enable_collapsible:
                        lines.append("<details>")
                        lines.append("<summary>点击查看市场影响分析</summary>")
                        lines.append("")
                    lines.append(executive_summary.market_impact)
                    lines.append("")
                    if self.enable_collapsible:
                        lines.append("</details>")
                    lines.append("")

                # Recommended actions (collapsible list)
                if executive_summary.recommended_actions:
                    lines.append("### ✅ 建议行动")
                    lines.append("")
                    if self.enable_collapsible:
                        lines.append("<details>")
                        lines.append("<summary>点击查看建议行动</summary>")
                        lines.append("")
                    for action in executive_summary.recommended_actions:
                        lines.append(f"- **{action}**")
                    lines.append("")
                    if self.enable_collapsible:
                        lines.append("</details>")
                    lines.append("")

                # Opportunities (collapsible)
                if executive_summary.opportunities:
                    lines.append("### 🚀 机会")
                    lines.append("")
                    if self.enable_collapsible:
                        lines.append("<details>")
                        lines.append("<summary>点击查看机会</summary>")
                        lines.append("")
                    for opportunity in executive_summary.opportunities:
                        lines.append(f"- **{opportunity}**")
                    lines.append("")
                    if self.enable_collapsible:
                        lines.append("</details>")
                    lines.append("")

                # Risk factors (collapsible)
                if executive_summary.risk_factors:
                    lines.append("### ⚠️ 风险因素")
                    lines.append("")
                    if self.enable_collapsible:
                        lines.append("<details>")
                        lines.append("<summary>点击查看风险因素</summary>")
                        lines.append("")
                    for risk in executive_summary.risk_factors:
                        lines.append(f"- **{risk}**")
                    lines.append("")
                    if self.enable_collapsible:
                        lines.append("</details>")
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

        # Hot topics with collapsible sections
        lines.append("## 🔥 热点话题")
        lines.append("")

        if self.enable_collapsible:
            lines.append("<details>")
            lines.append("<summary>点击查看热点话题详情</summary>")
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

        if self.enable_collapsible:
            lines.append("</details>")
        lines.append("")

        # Emerging trends (collapsible)
        if trend_report.emerging_trends:
            lines.append("### 📈 新兴趋势")
            lines.append("")
            if self.enable_collapsible:
                lines.append("<details>")
                lines.append("<summary>点击查看新兴趋势</summary>")
                lines.append("")
            for trend in trend_report.emerging_trends[:5]:
                lines.append(
                    f"- **{trend.name}**: {trend.summary or f'热度 {trend.heat_score:.0f}'}"
                )
            lines.append("")
            if self.enable_collapsible:
                lines.append("</details>")
            lines.append("")

        # Important articles (collapsible)
        lines.append("## 📰 重要新闻")
        lines.append("")

        if self.enable_collapsible:
            lines.append("<details>")
            lines.append("<summary>点击查看重要新闻</summary>")
            lines.append("")

        # Sort by importance
        important_articles = sorted(
            articles, key=lambda x: x.get("importance", 0), reverse=True
        )[: self.config.max_items_per_section]

        for article in important_articles:
            lines.append(f"### {article.get('title', '无标题')}")
            lines.append("")
            lines.append(
                f"**来源**: {article.get('source', '未知')} | **类别**: {article.get('category', '其他')} | **重要性**: {article.get('importance', 0)}/5"
            )

            if article.get("summary"):
                lines.append("")
                lines.append(f"{article['summary'][:200]}...")

            if article.get("url"):
                lines.append("")
                lines.append(f"[阅读原文]({article['url']})")

            lines.append("---")
            lines.append("")

        if self.enable_collapsible:
            lines.append("</details>")
        lines.append("")

        # Statistics section (collapsible)
        lines.append("## 📊 数据统计")
        lines.append("")

        if self.enable_collapsible:
            lines.append("<details>")
            lines.append("<summary>点击查看详细统计数据</summary>")
            lines.append("")

        # Category distribution
        if trend_report.category_distribution:
            lines.append("### 类别分布")
            lines.append("")
            lines.append("| 类别 | 文章数 | 占比 |")
            lines.append("|------|--------|------|")
            total = sum(trend_report.category_distribution.values())
            for category, count in sorted(
                trend_report.category_distribution.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                percentage = (count / total * 100) if total > 0 else 0
                lines.append(f"| {category} | {count} | {percentage:.1f}% |")
            lines.append("")

        # Source distribution
        if trend_report.source_distribution:
            lines.append("### 来源分布")
            lines.append("")
            lines.append("| 来源 | 文章数 | 占比 |")
            lines.append("|------|--------|------|")
            total = sum(trend_report.source_distribution.values())
            for source, count in sorted(
                trend_report.source_distribution.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                percentage = (count / total * 100) if total > 0 else 0
                lines.append(f"| {source} | {count} | {percentage:.1f}% |")
            lines.append("")

        # Key events
        if trend_report.key_events:
            lines.append("### 关键事件")
            lines.append("")
            for event in trend_report.key_events[:5]:
                lines.append(f"- **{event.title}** ({event.source})")
            lines.append("")

        if self.enable_collapsible:
            lines.append("</details>")
        lines.append("")

        # Footer
        lines.append("---")
        lines.append(f"*报告由新闻聚合分析系统自动生成*")
        lines.append(f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        return "\n".join(lines)

    def generate_daily_report(
        self,
        trend_report: TrendReport,
        articles: List[Dict[str, Any]],
        date: str = None,
        format: str = "markdown",
        enhanced: bool = True,
    ) -> str:
        """Generate daily report with optional enhancements"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        if format == "markdown":
            if enhanced:
                return self._generate_enhanced_markdown_report(
                    trend_report, articles, date
                )
            else:
                return self._generate_markdown_report(trend_report, articles, date)
        else:
            return super().generate_daily_report(trend_report, articles, date, format)

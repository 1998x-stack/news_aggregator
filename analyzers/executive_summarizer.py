"""
Executive summary generator using LLM
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dashscope_client import DashScopeClient, get_dashscope_client
from analyzers.trend_analyzer import TrendReport
from loguru import logger


@dataclass
class ExecutiveSummary:
    """Executive summary data structure"""

    key_insights: List[str]
    market_impact: str
    recommended_actions: List[str]
    risk_factors: List[str]
    opportunities: List[str]


class ExecutiveSummarizer:
    """Generate executive summaries using LLM"""

    def __init__(self):
        self.ollama_client = get_dashscope_client()

    def generate_summary(
        self, trend_report: TrendReport, articles: List[Dict[str, Any]], date: str
    ) -> ExecutiveSummary:
        """Generate executive summary from trend report and articles"""
        try:
            # Prepare context for LLM
            context = self._prepare_context(trend_report, articles, date)

            # Create prompt for executive summary
            prompt = self._create_executive_summary_prompt(context)

            # Get response from LLM
            response = self.dashscope_client.generate(
                prompt=prompt, model="qwen-max", format="json", temperature=0.3
            )

            # Parse response
            import json

            summary_data = json.loads(response["response"])

            return ExecutiveSummary(
                key_insights=summary_data.get("key_insights", []),
                market_impact=summary_data.get("market_impact", ""),
                recommended_actions=summary_data.get("recommended_actions", []),
                risk_factors=summary_data.get("risk_factors", []),
                opportunities=summary_data.get("opportunities", []),
            )

        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            # Return default summary on error
            return self._generate_default_summary(trend_report, articles)

    def _prepare_context(
        self, trend_report: TrendReport, articles: List[Dict[str, Any]], date: str
    ) -> Dict[str, Any]:
        """Prepare context data for LLM"""
        # Top hot topics
        top_topics = [
            {
                "name": topic.name,
                "heat_score": topic.heat_score,
                "mention_count": topic.mention_count,
                "category": topic.category,
            }
            for topic in trend_report.hot_topics[:5]
        ]

        # Top important articles
        important_articles = sorted(
            articles, key=lambda x: x.get("importance", 0), reverse=True
        )[:10]

        top_articles = [
            {
                "title": article.get("title", ""),
                "source": article.get("source", ""),
                "importance": article.get("importance", 0),
                "category": article.get("category", ""),
                "sentiment": article.get("sentiment", "unknown"),
            }
            for article in important_articles
        ]

        # Category distribution
        category_dist = trend_report.category_distribution

        # Source distribution
        source_dist = trend_report.source_distribution

        return {
            "date": date,
            "total_articles": trend_report.total_articles,
            "hot_topics": top_topics,
            "important_articles": top_articles,
            "category_distribution": category_dist,
            "source_distribution": source_dist,
            "emerging_trends": len(trend_report.emerging_trends),
            "key_events": len(trend_report.key_events),
        }

    def _create_executive_summary_prompt(self, context: Dict[str, Any]) -> str:
        """Create prompt for executive summary generation"""
        return f"""
        You are a senior analyst specializing in technology and business intelligence.
        Generate an executive summary based on the following news data from {context["date"]}: 
        
        DATA OVERVIEW:
        - Total articles analyzed: {context["total_articles"]}
        - Top hot topics: {len(context["hot_topics"])}
        - Emerging trends: {context["emerging_trends"]}
        - Key events: {context["key_events"]}
        
        TOP HOT TOPICS:
        {self._format_topics(context["hot_topics"])}
        
        MOST IMPORTANT ARTICLES:
        {self._format_articles(context["important_articles"])}
        
        CATEGORY DISTRIBUTION:
        {self._format_distribution(context["category_distribution"])}
        
        SOURCE DISTRIBUTION:
        {self._format_distribution(context["source_distribution"])}
        
        Generate a comprehensive executive summary in JSON format with the following structure:
        {{
          "key_insights": ["3-5 key insights about market trends and developments"],
          "market_impact": "Brief analysis of potential market impact (2-3 sentences)",
          "recommended_actions": ["3-5 specific recommended actions for decision makers"],
          "risk_factors": ["2-3 potential risk factors to monitor"],
          "opportunities": ["2-3 opportunities identified in the data"]
        }}
        
        Guidelines:
        - Focus on actionable intelligence
        - Highlight significant market movements or technological shifts
        - Identify patterns across multiple sources
        - Consider both risks and opportunities
        - Be concise but comprehensive
        - Use professional business language
        
        Response (JSON only):
        """

    def _format_topics(self, topics: List[Dict[str, Any]]) -> str:
        """Format topics for prompt"""
        lines = []
        for topic in topics:
            lines.append(
                f"- {topic['name']} (Heat: {topic['heat_score']:.1f}, "
                f"Mentions: {topic['mention_count']}, Category: {topic['category']})"
            )
        return "\n".join(lines)

    def _format_articles(self, articles: List[Dict[str, Any]]) -> str:
        """Format articles for prompt"""
        lines = []
        for article in articles[:5]:  # Top 5
            lines.append(
                f"- {article['title']} (Source: {article['source']}, "
                f"Importance: {article['importance']}/5, Sentiment: {article['sentiment']})"
            )
        return "\n".join(lines)

    def _format_distribution(self, distribution: Dict[str, int]) -> str:
        """Format distribution for prompt"""
        lines = []
        for name, count in sorted(
            distribution.items(), key=lambda x: x[1], reverse=True
        )[:5]:
            lines.append(f"- {name}: {count}")
        return "\n".join(lines)

    def _generate_default_summary(
        self, trend_report: TrendReport, articles: List[Dict[str, Any]]
    ) -> ExecutiveSummary:
        """Generate default summary when LLM fails"""
        # Simple heuristic-based summary
        key_insights = []

        if trend_report.hot_topics:
            top_topic = trend_report.hot_topics[0]
            key_insights.append(
                f"'{top_topic.name}' is the hottest topic with {top_topic.mention_count} mentions "
                f"and a heat score of {top_topic.heat_score:.1f}"
            )

        # Most active category
        if trend_report.category_distribution:
            top_category = max(
                trend_report.category_distribution.items(), key=lambda x: x[1]
            )
            key_insights.append(
                f"'{top_category[0]}' category leads with {top_category[1]} articles"
            )

        # Sentiment analysis if available
        sentiment_articles = [a for a in articles if a.get("sentiment")]
        if sentiment_articles:
            positive = len(
                [a for a in sentiment_articles if a["sentiment"] == "positive"]
            )
            negative = len(
                [a for a in sentiment_articles if a["sentiment"] == "negative"]
            )
            key_insights.append(
                f"Sentiment analysis shows {positive} positive vs {negative} negative articles"
            )

        return ExecutiveSummary(
            key_insights=key_insights,
            market_impact="Market impact analysis requires LLM processing",
            recommended_actions=["Monitor hot topics for developments"],
            risk_factors=["Limited analysis due to processing error"],
            opportunities=["Review trending topics for opportunities"],
        )

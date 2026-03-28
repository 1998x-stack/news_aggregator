"""
Temporal trend analysis for day-over-day comparisons
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from db.database import db_manager
from db.models import Article, Trend
from loguru import logger


@dataclass
class TemporalComparison:
    """Day-over-day comparison data"""

    date: str
    previous_date: str

    # Article metrics
    article_count: int
    prev_article_count: int
    article_growth_rate: float

    # Trend metrics
    trend_count: int
    prev_trend_count: int
    trend_growth_rate: float

    # Category changes
    category_changes: Dict[str, Dict[str, int]]

    # Top movers
    top_growing_topics: List[Dict[str, Any]]
    top_declining_topics: List[Dict[str, Any]]

    # Insights
    insights: List[str]


class TemporalAnalyzer:
    """Analyze trends over time periods"""

    def __init__(self):
        self.comparison_window = 1  # days

    def compare_periods(self, date: str, days: int = 1) -> TemporalComparison:
        """
        Compare two time periods

        Args:
            date: End date for comparison (YYYY-MM-DD)
            days: Number of days to compare

        Returns:
            TemporalComparison object
        """
        try:
            # Parse dates
            end_date = datetime.fromisoformat(date)
            start_date = end_date - timedelta(days=days)
            prev_end_date = start_date
            prev_start_date = prev_end_date - timedelta(days=days)

            logger.info(
                f"Comparing {start_date.date()} to {end_date.date()} "
                f"vs {prev_start_date.date()} to {prev_end_date.date()}"
            )

            # Get data for both periods
            current_articles = self._get_articles_in_period(start_date, end_date)
            prev_articles = self._get_articles_in_period(prev_start_date, prev_end_date)

            current_trends = self._get_trends_in_period(start_date, end_date)
            prev_trends = self._get_trends_in_period(prev_start_date, prev_end_date)

            # Calculate metrics
            article_metrics = self._calculate_article_metrics(
                current_articles, prev_articles
            )
            trend_metrics = self._calculate_trend_metrics(current_trends, prev_trends)
            category_changes = self._analyze_category_changes(
                current_articles, prev_articles
            )

            # Identify top movers
            top_growing, top_declining = self._identify_topic_movers(
                current_trends, prev_trends
            )

            # Generate insights
            insights = self._generate_insights(
                article_metrics,
                trend_metrics,
                category_changes,
                top_growing,
                top_declining,
            )

            return TemporalComparison(
                date=date,
                previous_date=prev_end_date.isoformat(),
                article_count=len(current_articles),
                prev_article_count=len(prev_articles),
                article_growth_rate=article_metrics["growth_rate"],
                trend_count=len(current_trends),
                prev_trend_count=len(prev_trends),
                trend_growth_rate=trend_metrics["growth_rate"],
                category_changes=category_changes,
                top_growing_topics=top_growing,
                top_declining_topics=top_declining,
                insights=insights,
            )

        except Exception as e:
            logger.error(f"Error in temporal comparison: {e}")
            raise

    def get_temporal_trend_series(
        self, topic: str, days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get historical trend data for a specific topic

        Args:
            topic: Topic name to track
            days: Number of days to look back

        Returns:
            List of daily trend data points
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            with db_manager.get_session() as session:
                # Get daily mentions of the topic
                daily_counts = []

                for i in range(days):
                    day_start = start_date + timedelta(days=i)
                    day_end = day_start + timedelta(days=1)

                    # Count articles mentioning the topic
                    count = (
                        session.query(Article)
                        .filter(
                            Article.publish_time >= day_start,
                            Article.publish_time < day_end,
                            (Article.title.contains(topic))
                            | (Article.content.contains(topic))
                            | (Article.keywords.contains(topic)),
                        )
                        .count()
                    )

                    daily_counts.append(
                        {"date": day_start.strftime("%Y-%m-%d"), "mention_count": count}
                    )

                return daily_counts

        except Exception as e:
            logger.error(f"Error getting temporal trend series: {e}")
            return []

    def identify_emerging_trends(
        self, days: int = 7, growth_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Identify emerging trends based on growth rate

        Args:
            days: Number of days to analyze
            growth_threshold: Minimum growth rate to consider (0.5 = 50% growth)

        Returns:
            List of emerging trends with growth metrics
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            mid_date = start_date + timedelta(days=days // 2)

            with db_manager.get_session() as session:
                # Get trends from both halves of the period
                early_trends = (
                    session.query(Trend)
                    .filter(Trend.start_date >= start_date, Trend.start_date < mid_date)
                    .all()
                )

                recent_trends = (
                    session.query(Trend)
                    .filter(Trend.start_date >= mid_date, Trend.start_date <= end_date)
                    .all()
                )

                # Calculate growth rates
                early_dict = {t.name: t.mention_count for t in early_trends}
                recent_dict = {t.name: t.mention_count for t in recent_trends}

                emerging = []
                for topic, recent_count in recent_dict.items():
                    early_count = early_dict.get(topic, 0)

                    if early_count > 0:
                        growth_rate = (recent_count - early_count) / early_count
                    else:
                        growth_rate = float("inf") if recent_count > 0 else 0

                    if growth_rate >= growth_threshold:
                        emerging.append(
                            {
                                "topic": topic,
                                "early_count": early_count,
                                "recent_count": recent_count,
                                "growth_rate": growth_rate,
                            }
                        )

                # Sort by growth rate
                emerging.sort(key=lambda x: x["growth_rate"], reverse=True)

                return emerging[:20]  # Top 20 emerging trends

        except Exception as e:
            logger.error(f"Error identifying emerging trends: {e}")
            return []

    def _get_articles_in_period(
        self, start_date: datetime, end_date: datetime
    ) -> List[Article]:
        """Get articles within a time period"""
        with db_manager.get_session() as session:
            return (
                session.query(Article)
                .filter(
                    Article.publish_time >= start_date, Article.publish_time <= end_date
                )
                .all()
            )

    def _get_trends_in_period(
        self, start_date: datetime, end_date: datetime
    ) -> List[Trend]:
        """Get trends within a time period"""
        with db_manager.get_session() as session:
            return (
                session.query(Trend)
                .filter(Trend.start_date >= start_date, Trend.start_date <= end_date)
                .all()
            )

    def _calculate_article_metrics(
        self, current: List[Article], previous: List[Article]
    ) -> Dict[str, float]:
        """Calculate article-related metrics"""
        current_count = len(current)
        prev_count = len(previous)

        if prev_count > 0:
            growth_rate = (current_count - prev_count) / prev_count
        else:
            growth_rate = float("inf") if current_count > 0 else 0

        return {
            "current_count": current_count,
            "previous_count": prev_count,
            "growth_rate": growth_rate,
        }

    def _calculate_trend_metrics(
        self, current: List[Trend], previous: List[Trend]
    ) -> Dict[str, float]:
        """Calculate trend-related metrics"""
        return self._calculate_article_metrics(current, previous)

    def _analyze_category_changes(
        self, current: List[Article], previous: List[Article]
    ) -> Dict[str, Dict[str, int]]:
        """Analyze changes in category distribution"""
        current_dist = defaultdict(int)
        prev_dist = defaultdict(int)

        for article in current:
            current_dist[article.category] += 1

        for article in previous:
            prev_dist[article.category] += 1

        changes = {}
        all_categories = set(current_dist.keys()) | set(prev_dist.keys())

        for category in all_categories:
            current_count = current_dist[category]
            prev_count = prev_dist[category]

            changes[category] = {
                "current": current_count,
                "previous": prev_count,
                "change": current_count - prev_count,
            }

        return changes

    def _identify_topic_movers(
        self, current: List[Trend], previous: List[Trend]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Identify topics with significant growth or decline"""
        current_dict = {t.name: t for t in current}
        prev_dict = {t.name: t for t in previous}

        growing = []
        declining = []

        # Check for growing topics
        for name, current_trend in current_dict.items():
            prev_trend = prev_dict.get(name)

            if prev_trend:
                growth_rate = (
                    (current_trend.heat_score - prev_trend.heat_score)
                    / prev_trend.heat_score
                    if prev_trend.heat_score > 0
                    else 0
                )

                if growth_rate > 0.3:  # 30% growth
                    growing.append(
                        {
                            "topic": name,
                            "growth_rate": growth_rate,
                            "current_score": current_trend.heat_score,
                            "previous_score": prev_trend.heat_score,
                        }
                    )
                elif growth_rate < -0.3:  # 30% decline
                    declining.append(
                        {
                            "topic": name,
                            "growth_rate": growth_rate,
                            "current_score": current_trend.heat_score,
                            "previous_score": prev_trend.heat_score,
                        }
                    )

        # Sort by absolute growth rate
        growing.sort(key=lambda x: x["growth_rate"], reverse=True)
        declining.sort(key=lambda x: x["growth_rate"])

        return growing[:10], declining[:10]

    def _generate_insights(
        self,
        article_metrics: Dict[str, float],
        trend_metrics: Dict[str, float],
        category_changes: Dict[str, Dict[str, int]],
        growing: List[Dict[str, Any]],
        declining: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate human-readable insights"""
        insights = []

        # Article volume insights
        if article_metrics["growth_rate"] > 0.2:
            insights.append(
                f"Article volume increased by {article_metrics['growth_rate']:.1%} "
                f"({article_metrics['previous_count']} → {article_metrics['current_count']})"
            )
        elif article_metrics["growth_rate"] < -0.2:
            insights.append(
                f"Article volume decreased by {abs(article_metrics['growth_rate']):.1%} "
                f"({article_metrics['previous_count']} → {article_metrics['current_count']})"
            )

        # Trend insights
        if trend_metrics["growth_rate"] > 0.3:
            insights.append(
                f"Significant increase in trending topics "
                f"({trend_metrics['previous_count']} → {trend_metrics['current_count']})"
            )

        # Category insights
        top_growing_categories = sorted(
            [(cat, data["change"]) for cat, data in category_changes.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        for category, change in top_growing_categories:
            if change > 5:
                insights.append(f"'{category}' category gained {change} articles")
            elif change < -5:
                insights.append(f"'{category}' category lost {abs(change)} articles")

        # Topic movement insights
        if growing:
            top_grower = growing[0]
            insights.append(
                f"'{top_grower['topic']}' is the fastest growing topic "
                f"({top_grower['growth_rate']:.1%} growth)"
            )

        if declining:
            top_decliner = declining[0]
            insights.append(
                f"'{top_decliner['topic']}' is declining "
                f"({top_decliner['growth_rate']:.1%} change)"
            )

        return insights


# Global temporal analyzer instance
temporal_analyzer = TemporalAnalyzer()

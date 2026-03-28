"""
Comparative analysis API endpoints
"""

from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from db.database import get_db
from analyzers.temporal_analyzer import temporal_analyzer
from utils.cache_manager import cache_response
from loguru import logger

router = APIRouter()


@router.get("/day-over-day")
@cache_response(expire=600)
async def compare_day_over_day(
    db: Session = Depends(get_db),
    date: Optional[str] = Query(
        None, description="Date to compare (YYYY-MM-DD), defaults to yesterday"
    ),
    days: int = Query(1, ge=1, le=30, description="Number of days to compare"),
):
    """Compare two time periods day-over-day"""
    try:
        # Use provided date or default to yesterday
        if not date:
            date = (datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

        # Validate date format
        try:
            datetime.fromisoformat(date)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid date format. Use YYYY-MM-DD"
            )

        # Perform comparison
        comparison = temporal_analyzer.compare_periods(date, days)

        return {
            "comparison_type": "day_over_day",
            "current_period": {
                "end_date": comparison.date,
                "duration_days": days,
                "article_count": comparison.article_count,
                "trend_count": comparison.trend_count,
            },
            "previous_period": {
                "end_date": comparison.previous_date,
                "article_count": comparison.prev_article_count,
                "trend_count": comparison.prev_trend_count,
            },
            "growth_rates": {
                "articles": comparison.article_growth_rate,
                "trends": comparison.trend_growth_rate,
            },
            "category_changes": comparison.category_changes,
            "top_growing_topics": comparison.top_growing_topics,
            "top_declining_topics": comparison.top_declining_topics,
            "insights": comparison.insights,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Day-over-day comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/topic-timeline/{topic}")
@cache_response(expire=1800)
async def get_topic_timeline(
    topic: str,
    db: Session = Depends(get_db),
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
):
    """Get historical timeline for a specific topic"""
    try:
        timeline = temporal_analyzer.get_temporal_trend_series(topic, days)

        if not timeline:
            raise HTTPException(
                status_code=404, detail=f"No data found for topic '{topic}'"
            )

        # Calculate statistics
        mention_counts = [day["mention_count"] for day in timeline]
        avg_mentions = (
            sum(mention_counts) / len(mention_counts) if mention_counts else 0
        )
        max_mentions = max(mention_counts) if mention_counts else 0
        min_mentions = min(mention_counts) if mention_counts else 0

        # Identify peak days
        peak_threshold = avg_mentions * 1.5
        peak_days = [day for day in timeline if day["mention_count"] >= peak_threshold]

        return {
            "topic": topic,
            "period_days": days,
            "timeline": timeline,
            "statistics": {
                "average_mentions": avg_mentions,
                "maximum_mentions": max_mentions,
                "minimum_mentions": min_mentions,
                "peak_days": len(peak_days),
            },
            "peak_days": peak_days,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Topic timeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/emerging-trends")
@cache_response(expire=900)
async def get_emerging_trends(
    db: Session = Depends(get_db),
    days: int = Query(7, ge=1, le=30, description="Analysis period in days"),
    growth_threshold: float = Query(
        0.5, ge=0.1, le=5.0, description="Minimum growth rate"
    ),
):
    """Identify emerging trends based on growth rate"""
    try:
        emerging = temporal_analyzer.identify_emerging_trends(days, growth_threshold)

        if not emerging:
            return {
                "period_days": days,
                "growth_threshold": growth_threshold,
                "emerging_trends": [],
                "message": "No emerging trends found with the specified threshold",
            }

        # Calculate summary statistics
        avg_growth_rate = sum(t["growth_rate"] for t in emerging) / len(emerging)
        top_performers = [
            t for t in emerging if t["growth_rate"] > avg_growth_rate * 1.5
        ]

        return {
            "period_days": days,
            "growth_threshold": growth_threshold,
            "total_emerging": len(emerging),
            "average_growth_rate": avg_growth_rate,
            "top_performers_count": len(top_performers),
            "emerging_trends": emerging,
            "top_performers": top_performers[:5],
        }

    except Exception as e:
        logger.error(f"Emerging trends error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/today-vs-yesterday")
@cache_response(expire=300)
async def compare_today_vs_yesterday(db: Session = Depends(get_db)):
    """Quick comparison of today vs yesterday"""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

        # Get comparison data
        comparison = temporal_analyzer.compare_periods(today, 1)

        # Get topic timeline for top growing topic
        top_growing_topic = None
        timeline_data = None

        if comparison.top_growing_topics:
            top_growing_topic = comparison.top_growing_topics[0]["topic"]
            timeline_data = temporal_analyzer.get_temporal_trend_series(
                top_growing_topic, days=14
            )

        return {
            "comparison_date": today,
            "previous_date": yesterday,
            "summary": {
                "article_change": comparison.article_growth_rate,
                "trend_change": comparison.trend_growth_rate,
                "articles_today": comparison.article_count,
                "articles_yesterday": comparison.prev_article_count,
            },
            "key_insights": comparison.insights[:3],  # Top 3 insights
            "top_growing_topic": {"topic": top_growing_topic, "timeline": timeline_data}
            if top_growing_topic
            else None,
            "detailed_comparison": comparison.to_dict()
            if hasattr(comparison, "to_dict")
            else vars(comparison),
        }

    except Exception as e:
        logger.error(f"Today vs yesterday comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/weekly-summary")
@cache_response(expire=3600)
async def get_weekly_summary(db: Session = Depends(get_db)):
    """Get comprehensive weekly comparison summary"""
    try:
        today = datetime.now()

        # This week vs last week
        this_week_end = today
        this_week_start = today - datetime.timedelta(days=7)
        last_week_end = this_week_start
        last_week_start = last_week_end - datetime.timedelta(days=7)

        # Get comparisons for different periods
        day_comparison = temporal_analyzer.compare_periods(
            (today - datetime.timedelta(days=1)).strftime("%Y-%m-%d"), 1
        )

        # Calculate weekly trends
        emerging = temporal_analyzer.identify_emerging_trends(
            days=7, growth_threshold=0.3
        )

        # Generate summary
        summary = {
            "period": "7_days",
            "comparison_date": today.strftime("%Y-%m-%d"),
            "article_metrics": {
                "current_week": day_comparison.article_count,
                "previous_week": day_comparison.prev_article_count,
                "growth_rate": day_comparison.article_growth_rate,
            },
            "trend_metrics": {
                "current_week": day_comparison.trend_count,
                "previous_week": day_comparison.prev_trend_count,
                "growth_rate": day_comparison.trend_growth_rate,
            },
            "emerging_trends_count": len(emerging),
            "top_emerging_trends": emerging[:5] if emerging else [],
            "key_insights": day_comparison.insights,
            "category_performance": day_comparison.category_changes,
        }

        return summary

    except Exception as e:
        logger.error(f"Weekly summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

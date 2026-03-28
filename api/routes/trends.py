"""
Trend analysis API endpoints
"""

from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from db.database import get_db
from db.models import Trend, Article, TrendArticle
from utils.cache_manager import cache_manager, cache_response
from loguru import logger

router = APIRouter()


@router.get("/")
@cache_response(expire=300)
async def get_trends(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    category: Optional[str] = Query(None),
    trend_type: Optional[str] = Query(None, regex="^(hot|emerging|declining)$"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    min_heat_score: Optional[float] = Query(None, ge=0),
    sort_by: Optional[str] = Query(
        "heat_score", regex="^(heat_score|mention_count|start_date)$"
    ),
    sort_order: Optional[str] = Query("desc", regex="^(asc|desc)$"),
):
    """Get trends with filtering and pagination"""
    try:
        query = db.query(Trend)

        # Apply filters
        if category:
            query = query.filter(Trend.category == category)

        if trend_type:
            query = query.filter(Trend.trend_type == trend_type)

        if start_date:
            query = query.filter(Trend.start_date >= start_date)

        if end_date:
            query = query.filter(Trend.end_date <= end_date)

        if min_heat_score is not None:
            query = query.filter(Trend.heat_score >= min_heat_score)

        # Apply sorting
        sort_column = getattr(Trend, sort_by)
        if sort_order == "desc":
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())

        # Get total count
        total = query.count()

        # Apply pagination
        trends = query.offset(skip).limit(limit).all()

        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "trends": [trend.to_dict() for trend in trends],
        }

    except Exception as e:
        logger.error(f"Error getting trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{trend_id}")
@cache_response(expire=600)
async def get_trend(trend_id: int, db: Session = Depends(get_db)):
    """Get a single trend by ID"""
    try:
        trend = db.query(Trend).filter(Trend.id == trend_id).first()

        if not trend:
            raise HTTPException(status_code=404, detail="Trend not found")

        # Get related articles
        related_articles = (
            db.query(Article)
            .join(TrendArticle)
            .filter(TrendArticle.trend_id == trend_id)
            .limit(10)
            .all()
        )

        trend_dict = trend.to_dict()
        trend_dict["related_articles"] = [
            article.to_dict() for article in related_articles
        ]

        return trend_dict

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trend {trend_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hot/top")
@cache_response(expire=180)
async def get_top_hot_topics(
    db: Session = Depends(get_db),
    limit: int = Query(10, ge=1, le=50),
    category: Optional[str] = Query(None),
):
    """Get top hot topics"""
    try:
        query = db.query(Trend).filter(Trend.trend_type == "hot")

        if category:
            query = query.filter(Trend.category == category)

        # Get trends from last 7 days
        from datetime import datetime, timedelta

        week_ago = datetime.now() - timedelta(days=7)
        query = query.filter(Trend.start_date >= week_ago)

        top_trends = query.order_by(Trend.heat_score.desc()).limit(limit).all()

        return {
            "trends": [trend.to_dict() for trend in top_trends],
            "period": "last_7_days",
        }

    except Exception as e:
        logger.error(f"Error getting top hot topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/emerging/recent")
@cache_response(expire=180)
async def get_recent_emerging_trends(
    db: Session = Depends(get_db),
    limit: int = Query(10, ge=1, le=50),
    days: int = Query(3, ge=1, le=30),
):
    """Get recently emerging trends"""
    try:
        from datetime import datetime, timedelta

        start_date = datetime.now() - timedelta(days=days)

        emerging_trends = (
            db.query(Trend)
            .filter(
                and_(Trend.trend_type == "emerging", Trend.start_date >= start_date)
            )
            .order_by(Trend.heat_score.desc())
            .limit(limit)
            .all()
        )

        return {
            "trends": [trend.to_dict() for trend in emerging_trends],
            "period_days": days,
        }

    except Exception as e:
        logger.error(f"Error getting emerging trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/timeline/daily")
@cache_response(expire=300)
async def get_daily_trend_timeline(
    db: Session = Depends(get_db),
    days: int = Query(7, ge=1, le=30),
    category: Optional[str] = Query(None),
):
    """Get daily trend timeline"""
    try:
        from datetime import datetime, timedelta

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get trends by day
        daily_stats = []

        for i in range(days):
            day_start = start_date + timedelta(days=i)
            day_end = day_start + timedelta(days=1)

            day_query = db.query(Trend).filter(
                and_(Trend.start_date >= day_start, Trend.start_date < day_end)
            )

            if category:
                day_query = day_query.filter(Trend.category == category)

            trend_count = day_query.count()
            avg_heat_score = (
                db.query(func.avg(Trend.heat_score))
                .filter(and_(Trend.start_date >= day_start, Trend.start_date < day_end))
                .scalar()
                or 0
            )

            daily_stats.append(
                {
                    "date": day_start.strftime("%Y-%m-%d"),
                    "trend_count": trend_count,
                    "avg_heat_score": float(avg_heat_score),
                }
            )

        return {"period_days": days, "daily_stats": daily_stats}

    except Exception as e:
        logger.error(f"Error getting daily trend timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories/distribution")
@cache_response(expire=600)
async def get_trend_category_distribution(db: Session = Depends(get_db)):
    """Get trend distribution by category"""
    try:
        distribution = (
            db.query(
                Trend.category,
                func.count(Trend.id).label("count"),
                func.avg(Trend.heat_score).label("avg_heat_score"),
            )
            .group_by(Trend.category)
            .all()
        )

        return {
            "distribution": [
                {
                    "category": item.category,
                    "count": item.count,
                    "avg_heat_score": float(item.avg_heat_score)
                    if item.avg_heat_score
                    else 0,
                }
                for item in distribution
            ]
        }

    except Exception as e:
        logger.error(f"Error getting category distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

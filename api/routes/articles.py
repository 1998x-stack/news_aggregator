"""
Article API endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from db.database import get_db
from db.models import Article
from utils.cache_manager import cache_manager, cache_response
from loguru import logger

router = APIRouter()


@router.get("/")
@cache_response(expire=300)
async def get_articles(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    category: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    importance_min: Optional[int] = Query(None, ge=1, le=5),
    importance_max: Optional[int] = Query(None, ge=1, le=5),
    search: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    sort_by: Optional[str] = Query(
        "publish_time", regex="^(publish_time|importance|score)$"
    ),
    sort_order: Optional[str] = Query("desc", regex="^(asc|desc)$"),
):
    """Get articles with filtering and pagination"""
    try:
        query = db.query(Article)

        # Apply filters
        if category:
            query = query.filter(Article.category == category)

        if source:
            query = query.filter(Article.source.contains(source))

        if importance_min is not None:
            query = query.filter(Article.importance >= importance_min)

        if importance_max is not None:
            query = query.filter(Article.importance <= importance_max)

        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    Article.title.contains(search_term),
                    Article.content.contains(search_term),
                    Article.summary.contains(search_term),
                )
            )

        if start_date:
            query = query.filter(Article.publish_time >= start_date)

        if end_date:
            query = query.filter(Article.publish_time <= end_date)

        # Apply sorting
        sort_column = getattr(Article, sort_by)
        if sort_order == "desc":
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())

        # Get total count
        total = query.count()

        # Apply pagination
        articles = query.offset(skip).limit(limit).all()

        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "articles": [article.to_dict() for article in articles],
        }

    except Exception as e:
        logger.error(f"Error getting articles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{article_id}")
@cache_response(expire=600)
async def get_article(article_id: str, db: Session = Depends(get_db)):
    """Get a single article by ID"""
    try:
        article = db.query(Article).filter(Article.id == article_id).first()

        if not article:
            raise HTTPException(status_code=404, detail="Article not found")

        return article.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting article {article_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{article_id}/related")
@cache_response(expire=300)
async def get_related_articles(
    article_id: str, db: Session = Depends(get_db), limit: int = Query(5, ge=1, le=20)
):
    """Get related articles for a given article"""
    try:
        # Get the source article
        article = db.query(Article).filter(Article.id == article_id).first()

        if not article:
            raise HTTPException(status_code=404, detail="Article not found")

        # Find related articles by category and keywords
        query = db.query(Article).filter(Article.id != article_id)

        # Filter by same category
        if article.category:
            query = query.filter(Article.category == article.category)

        # Filter by similar time period
        if article.publish_time:
            from datetime import timedelta

            start_time = article.publish_time - timedelta(days=1)
            end_time = article.publish_time + timedelta(days=1)
            query = query.filter(
                and_(
                    Article.publish_time >= start_time, Article.publish_time <= end_time
                )
            )

        # Order by importance and get top results
        related_articles = query.order_by(Article.importance.desc()).limit(limit).all()

        return {
            "article_id": article_id,
            "related_articles": [ra.to_dict() for ra in related_articles],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting related articles for {article_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories/summary")
@cache_response(expire=600)
async def get_categories_summary(db: Session = Depends(get_db)):
    """Get summary statistics by category"""
    try:
        from sqlalchemy import func

        summary = (
            db.query(
                Article.category,
                func.count(Article.id).label("count"),
                func.avg(Article.importance).label("avg_importance"),
                func.max(Article.publish_time).label("latest_article"),
            )
            .group_by(Article.category)
            .all()
        )

        return {
            "categories": [
                {
                    "category": item.category,
                    "count": item.count,
                    "avg_importance": float(item.avg_importance)
                    if item.avg_importance
                    else 0,
                    "latest_article": item.latest_article.isoformat()
                    if item.latest_article
                    else None,
                }
                for item in summary
            ]
        }

    except Exception as e:
        logger.error(f"Error getting categories summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources/summary")
@cache_response(expire=600)
async def get_sources_summary(db: Session = Depends(get_db)):
    """Get summary statistics by source"""
    try:
        from sqlalchemy import func

        summary = (
            db.query(
                Article.source,
                func.count(Article.id).label("count"),
                func.avg(Article.importance).label("avg_importance"),
                func.max(Article.publish_time).label("latest_article"),
            )
            .group_by(Article.source)
            .all()
        )

        return {
            "sources": [
                {
                    "source": item.source,
                    "count": item.count,
                    "avg_importance": float(item.avg_importance)
                    if item.avg_importance
                    else 0,
                    "latest_article": item.latest_article.isoformat()
                    if item.latest_article
                    else None,
                }
                for item in summary
            ]
        }

    except Exception as e:
        logger.error(f"Error getting sources summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

"""
Data export API endpoints
"""

from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pathlib import Path

from db.database import get_db
from db.models import Article, Trend
from utils.export_utils import DataExporter
from utils.cache_manager import cache_response
from loguru import logger

router = APIRouter()


@router.get("/articles/csv")
@cache_response(expire=300)
async def export_articles_csv(
    db: Session = Depends(get_db),
    category: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    """Export articles to CSV"""
    try:
        query = db.query(Article)

        if category:
            query = query.filter(Article.category == category)
        if source:
            query = query.filter(Article.source.contains(source))
        if start_date:
            query = query.filter(Article.publish_time >= start_date)
        if end_date:
            query = query.filter(Article.publish_time <= end_date)

        articles = query.all()

        if not articles:
            raise HTTPException(status_code=404, detail="No articles found for export")

        articles_data = [article.to_dict() for article in articles]

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"articles_export_{timestamp}.csv"
        filepath = Path("exports") / filename

        # Export to CSV
        if DataExporter.export_articles_to_csv(articles_data, str(filepath)):
            return FileResponse(str(filepath), media_type="text/csv", filename=filename)
        else:
            raise HTTPException(status_code=500, detail="CSV export failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Articles CSV export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/articles/json")
@cache_response(expire=300)
async def export_articles_json(
    db: Session = Depends(get_db),
    category: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    """Export articles to JSON"""
    try:
        query = db.query(Article)

        if category:
            query = query.filter(Article.category == category)
        if source:
            query = query.filter(Article.source.contains(source))
        if start_date:
            query = query.filter(Article.publish_time >= start_date)
        if end_date:
            query = query.filter(Article.publish_time <= end_date)

        articles = query.all()

        if not articles:
            raise HTTPException(status_code=404, detail="No articles found for export")

        articles_data = [article.to_dict() for article in articles]

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"articles_export_{timestamp}.json"
        filepath = Path("exports") / filename

        # Export to JSON
        if DataExporter.export_articles_to_json(articles_data, str(filepath)):
            return FileResponse(
                str(filepath), media_type="application/json", filename=filename
            )
        else:
            raise HTTPException(status_code=500, detail="JSON export failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Articles JSON export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends/csv")
@cache_response(expire=300)
async def export_trends_csv(
    db: Session = Depends(get_db),
    category: Optional[str] = Query(None),
    trend_type: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    """Export trends to CSV"""
    try:
        query = db.query(Trend)

        if category:
            query = query.filter(Trend.category == category)
        if trend_type:
            query = query.filter(Trend.trend_type == trend_type)
        if start_date:
            query = query.filter(Trend.start_date >= start_date)
        if end_date:
            query = query.filter(Trend.end_date <= end_date)

        trends = query.all()

        if not trends:
            raise HTTPException(status_code=404, detail="No trends found for export")

        trends_data = [trend.to_dict() for trend in trends]

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trends_export_{timestamp}.csv"
        filepath = Path("exports") / filename

        # Export to CSV
        if DataExporter.export_trends_to_csv(trends_data, str(filepath)):
            return FileResponse(str(filepath), media_type="text/csv", filename=filename)
        else:
            raise HTTPException(status_code=500, detail="CSV export failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trends CSV export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends/json")
@cache_response(expire=300)
async def export_trends_json(
    db: Session = Depends(get_db),
    category: Optional[str] = Query(None),
    trend_type: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    """Export trends to JSON"""
    try:
        query = db.query(Trend)

        if category:
            query = query.filter(Trend.category == category)
        if trend_type:
            query = query.filter(Trend.trend_type == trend_type)
        if start_date:
            query = query.filter(Trend.start_date >= start_date)
        if end_date:
            query = query.filter(Trend.end_date <= end_date)

        trends = query.all()

        if not trends:
            raise HTTPException(status_code=404, detail="No trends found for export")

        trends_data = [trend.to_dict() for trend in trends]

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trends_export_{timestamp}.json"
        filepath = Path("exports") / filename

        # Export to JSON
        if DataExporter.export_trends_to_json(trends_data, str(filepath)):
            return FileResponse(
                str(filepath), media_type="application/json", filename=filename
            )
        else:
            raise HTTPException(status_code=500, detail="JSON export failed")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trends JSON export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/complete/{report_date}")
@cache_response(expire=300)
async def export_complete_report(report_date: str, db: Session = Depends(get_db)):
    """Export complete report for a specific date"""
    try:
        from datetime import datetime

        # Parse date
        date_obj = datetime.fromisoformat(report_date)
        start_of_day = date_obj.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = date_obj.replace(hour=23, minute=59, second=59, microsecond=999999)

        # Get articles for the date
        articles = (
            db.query(Article)
            .filter(
                Article.publish_time >= start_of_day, Article.publish_time <= end_of_day
            )
            .all()
        )

        # Get trends for the date
        trends = (
            db.query(Trend)
            .filter(Trend.start_date >= start_of_day, Trend.start_date <= end_of_day)
            .all()
        )

        if not articles and not trends:
            raise HTTPException(
                status_code=404, detail=f"No data found for date {report_date}"
            )

        articles_data = [article.to_dict() for article in articles]
        trends_data = [trend.to_dict() for trend in trends]

        # Create exports directory
        exports_dir = Path("exports")
        exports_dir.mkdir(parents=True, exist_ok=True)

        # Export all formats
        results = DataExporter.export_combined_report(
            articles_data, trends_data, report_date, str(exports_dir)
        )

        if not results:
            raise HTTPException(status_code=500, detail="Export failed")

        # Return JSON with download links
        return {
            "report_date": report_date,
            "total_articles": len(articles),
            "total_trends": len(trends),
            "exported_files": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Complete report export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

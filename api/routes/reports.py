"""
Report API endpoints
"""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import Report
from utils.cache_manager import cache_manager, cache_response
from loguru import logger

router = APIRouter()


@router.get("/")
@cache_response(expire=300)
async def get_reports(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    report_type: Optional[str] = Query(None, regex="^(daily|category|timeline)$"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    is_generated: Optional[bool] = Query(None),
    is_delivered: Optional[bool] = Query(None),
):
    """Get reports with filtering and pagination"""
    try:
        query = db.query(Report)

        # Apply filters
        if report_type:
            query = query.filter(Report.report_type == report_type)

        if start_date:
            query = query.filter(Report.report_date >= start_date)

        if end_date:
            query = query.filter(Report.report_date <= end_date)

        if is_generated is not None:
            query = query.filter(Report.is_generated == is_generated)

        if is_delivered is not None:
            query = query.filter(Report.is_delivered == is_delivered)

        # Apply sorting (newest first)
        query = query.order_by(Report.report_date.desc())

        # Get total count
        total = query.count()

        # Apply pagination
        reports = query.offset(skip).limit(limit).all()

        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "reports": [report.to_dict() for report in reports],
        }

    except Exception as e:
        logger.error(f"Error getting reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{report_id}")
@cache_response(expire=600)
async def get_report(report_id: int, db: Session = Depends(get_db)):
    """Get a single report by ID"""
    try:
        report = db.query(Report).filter(Report.id == report_id).first()

        if not report:
            raise HTTPException(status_code=404, detail="Report not found")

        return report.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report {report_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest/daily")
@cache_response(expire=180)
async def get_latest_daily_report(db: Session = Depends(get_db)):
    """Get the latest daily report"""
    try:
        report = (
            db.query(Report)
            .filter(Report.report_type == "daily", Report.is_generated == True)
            .order_by(Report.report_date.desc())
            .first()
        )

        if not report:
            raise HTTPException(status_code=404, detail="No daily report found")

        return report.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest daily report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/summary")
@cache_response(expire=600)
async def get_reports_summary(db: Session = Depends(get_db)):
    """Get reports summary statistics"""
    try:
        from sqlalchemy import func

        # Total reports by type
        type_stats = (
            db.query(Report.report_type, func.count(Report.id).label("count"))
            .group_by(Report.report_type)
            .all()
        )

        # Reports by generation status
        generation_stats = (
            db.query(Report.is_generated, func.count(Report.id).label("count"))
            .group_by(Report.is_generated)
            .all()
        )

        # Reports by delivery status
        delivery_stats = (
            db.query(Report.is_delivered, func.count(Report.id).label("count"))
            .group_by(Report.is_delivered)
            .all()
        )

        # Date range
        date_range = db.query(
            func.min(Report.report_date).label("earliest"),
            func.max(Report.report_date).label("latest"),
        ).first()

        # Average generation time
        avg_generation_time = (
            db.query(func.avg(Report.generation_time))
            .filter(Report.generation_time.isnot(None))
            .scalar()
        )

        return {
            "total_reports": db.query(Report).count(),
            "by_type": [
                {"type": item.report_type, "count": item.count} for item in type_stats
            ],
            "by_generation_status": [
                {"is_generated": item.is_generated, "count": item.count}
                for item in generation_stats
            ],
            "by_delivery_status": [
                {"is_delivered": item.is_delivered, "count": item.count}
                for item in delivery_stats
            ],
            "date_range": {
                "earliest": date_range.earliest.isoformat()
                if date_range.earliest
                else None,
                "latest": date_range.latest.isoformat() if date_range.latest else None,
            },
            "avg_generation_time": float(avg_generation_time)
            if avg_generation_time
            else 0,
        }

    except Exception as e:
        logger.error(f"Error getting reports summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{report_id}/content")
@cache_response(expire=600)
async def get_report_content(report_id: int, db: Session = Depends(get_db)):
    """Get report content"""
    try:
        report = db.query(Report).filter(Report.id == report_id).first()

        if not report:
            raise HTTPException(status_code=404, detail="Report not found")

        if not report.content:
            raise HTTPException(status_code=404, detail="Report content not available")

        return {
            "report_id": report_id,
            "title": report.title,
            "report_date": report.report_date.isoformat()
            if report.report_date
            else None,
            "format": report.format,
            "content": report.content,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report content {report_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/date/{report_date}")
@cache_response(expire=300)
async def get_report_by_date(report_date: str, db: Session = Depends(get_db)):
    """Get report by date"""
    try:
        report = (
            db.query(Report)
            .filter(Report.report_date == report_date, Report.report_type == "daily")
            .first()
        )

        if not report:
            raise HTTPException(
                status_code=404, detail=f"No report found for date {report_date}"
            )

        return report.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report for date {report_date}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

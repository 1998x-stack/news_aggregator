"""
Health check endpoints
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from db.database import get_db
from utils.cache_manager import cache_manager
from loguru import logger

router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "news-aggregator-api",
        "timestamp": None,  # Will be filled by middleware
    }


@router.get("/details")
async def health_check_details(db: Session = Depends(get_db)):
    """Detailed health check with dependencies"""
    health_status = {
        "status": "healthy",
        "service": "news-aggregator-api",
        "timestamp": None,
        "dependencies": {},
    }

    # Check database
    try:
        db.execute("SELECT 1")
        health_status["dependencies"]["database"] = {
            "status": "healthy",
            "message": "Database connection OK",
        }
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["dependencies"]["database"] = {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}",
        }

    # Check Redis
    try:
        cache_manager.redis_client.ping()
        health_status["dependencies"]["redis"] = {
            "status": "healthy",
            "message": "Redis connection OK",
        }
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["dependencies"]["redis"] = {
            "status": "unhealthy",
            "message": f"Redis connection failed: {str(e)}",
        }

    return health_status


@router.get("/ready")
async def readiness_check(db: Session = Depends(get_db)):
    """Readiness check for Kubernetes"""
    try:
        # Check database
        db.execute("SELECT 1")

        # Check Redis
        cache_manager.redis_client.ping()

        return {"status": "ready", "service": "news-aggregator-api"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {
            "status": "not_ready",
            "service": "news-aggregator-api",
            "error": str(e),
        }


@router.get("/metrics")
async def metrics():
    """Basic metrics endpoint"""
    try:
        # Get Redis info
        redis_info = cache_manager.redis_client.info()

        return {
            "status": "ok",
            "redis": {
                "connected_clients": redis_info.get("connected_clients", 0),
                "used_memory_human": redis_info.get("used_memory_human", "0B"),
                "uptime_in_seconds": redis_info.get("uptime_in_seconds", 0),
            },
        }
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return {"status": "error", "error": str(e)}

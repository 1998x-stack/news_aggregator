"""
Prometheus metrics endpoint
"""

from fastapi import APIRouter, Response
from utils.metrics import metrics_collector

router = APIRouter()


@router.get("/")
async def get_metrics():
    """Get Prometheus metrics"""
    metrics_data = metrics_collector.get_metrics()
    return Response(content=metrics_data, media_type="text/plain")

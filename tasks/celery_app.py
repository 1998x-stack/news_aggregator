"""
Celery application configuration
"""

import os
from celery import Celery
from loguru import logger

# Load configuration from environment
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Create Celery app
celery_app = Celery(
    "news_aggregator",
    broker=f"redis://{':' + REDIS_PASSWORD + '@' if REDIS_PASSWORD else ''}{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
    backend=f"redis://{':' + REDIS_PASSWORD + '@' if REDIS_PASSWORD else ''}{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
    include=["tasks.pipeline_tasks", "tasks.analysis_tasks", "tasks.report_tasks"],
)

# Configure Celery
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    # Task routing
    task_routes={
        "tasks.pipeline_tasks.run_pipeline": {"queue": "pipeline"},
        "tasks.analysis_tasks.*": {"queue": "analysis"},
        "tasks.report_tasks.*": {"queue": "reports"},
    },
    # Beat schedule (for periodic tasks)
    beat_schedule={
        "run-pipeline-daily": {
            "task": "tasks.pipeline_tasks.run_pipeline",
            "schedule": 3600.0,  # Run every hour
        },
        "cleanup-old-data": {
            "task": "tasks.maintenance_tasks.cleanup_old_data",
            "schedule": 86400.0,  # Run daily
        },
    },
    # Result expiration
    result_expires=3600,  # 1 hour
    # Task retry settings
    task_annotations={
        "*": {"max_retries": 3, "default_retry_delay": 60},
    },
)


@celery_app.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery"""
    logger.info("Debug task executed")
    return {"status": "success", "task_id": self.request.id}


@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Setup periodic tasks"""
    logger.info("Setting up periodic tasks")


if __name__ == "__main__":
    celery_app.start()

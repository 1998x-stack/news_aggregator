"""
Prometheus metrics collection for monitoring
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.core import CollectorRegistry
from functools import wraps
from typing import Callable, Optional
import time
from loguru import logger

# Create a custom registry
registry = CollectorRegistry()

# HTTP Request metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
    registry=registry,
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    registry=registry,
)

# Database metrics
db_queries_total = Counter(
    "db_queries_total",
    "Total database queries",
    ["operation", "table"],
    registry=registry,
)

db_query_duration_seconds = Histogram(
    "db_query_duration_seconds",
    "Database query latency",
    ["operation", "table"],
    registry=registry,
)

# Cache metrics
cache_operations_total = Counter(
    "cache_operations_total",
    "Total cache operations",
    ["operation", "result"],
    registry=registry,
)

cache_hit_rate = Gauge("cache_hit_rate", "Cache hit rate (0-1)", registry=registry)

# Pipeline metrics
pipeline_runs_total = Counter(
    "pipeline_runs_total", "Total pipeline runs", ["status"], registry=registry
)

pipeline_articles_processed = Histogram(
    "pipeline_articles_processed",
    "Number of articles processed per pipeline run",
    registry=registry,
)

# LLM metrics
llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM API requests",
    ["model", "operation"],
    registry=registry,
)

llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "LLM request latency",
    ["model", "operation"],
    registry=registry,
)

# System metrics
active_connections = Gauge(
    "active_connections", "Number of active connections", registry=registry
)

total_articles_stored = Gauge(
    "total_articles_stored",
    "Total number of articles stored in database",
    registry=registry,
)

total_trends_identified = Gauge(
    "total_trends_identified", "Total number of trends identified", registry=registry
)


class MetricsCollector:
    """Centralized metrics collection utility"""

    def __init__(self):
        self.cache_hits = 0
        self.cache_misses = 0

    def record_http_request(
        self, method: str, endpoint: str, status_code: int, duration: float
    ):
        """Record HTTP request metrics"""
        http_requests_total.labels(
            method=method, endpoint=endpoint, status_code=status_code
        ).inc()

        http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(
            duration
        )

    def record_db_query(self, operation: str, table: str, duration: float):
        """Record database query metrics"""
        db_queries_total.labels(operation=operation, table=table).inc()

        db_query_duration_seconds.labels(operation=operation, table=table).observe(
            duration
        )

    def record_cache_operation(self, operation: str, hit: bool):
        """Record cache operation metrics"""
        result = "hit" if hit else "miss"
        cache_operations_total.labels(operation=operation, result=result).inc()

        # Update cache hit rate
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        total_ops = self.cache_hits + self.cache_misses
        if total_ops > 0:
            cache_hit_rate.set(self.cache_hits / total_ops)

    def record_pipeline_run(self, status: str, article_count: int):
        """Record pipeline execution metrics"""
        pipeline_runs_total.labels(status=status).inc()
        pipeline_articles_processed.observe(article_count)

    def record_llm_request(self, model: str, operation: str, duration: float):
        """Record LLM request metrics"""
        llm_requests_total.labels(model=model, operation=operation).inc()

        llm_request_duration_seconds.labels(model=model, operation=operation).observe(
            duration
        )

    def update_gauge(self, gauge: Gauge, value: float):
        """Update a gauge metric"""
        gauge.set(value)

    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        return generate_latest(registry).decode("utf-8")


# Global metrics collector instance
metrics_collector = MetricsCollector()


def track_http_metrics(endpoint: str):
    """Decorator to track HTTP request metrics"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            method = kwargs.get("request", {}).get("method", "GET") if kwargs else "GET"

            try:
                result = await func(*args, **kwargs)
                status_code = 200
                return result
            except Exception as e:
                status_code = 500
                raise
            finally:
                duration = time.time() - start_time
                metrics_collector.record_http_request(
                    method=method,
                    endpoint=endpoint,
                    status_code=status_code,
                    duration=duration,
                )

        return wrapper

    return decorator


def track_db_metrics(operation: str, table: str):
    """Decorator to track database query metrics"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metrics_collector.record_db_query(
                    operation=operation, table=table, duration=duration
                )

        return wrapper

    return decorator


def track_llm_metrics(model: str, operation: str):
    """Decorator to track LLM request metrics"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metrics_collector.record_llm_request(
                    model=model, operation=operation, duration=duration
                )

        return wrapper

    return decorator

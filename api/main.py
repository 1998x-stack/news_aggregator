"""
FastAPI application main entry point
"""

import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from api.routes import articles, trends, reports, health, export, metrics, comparative
from api.middleware import logging_middleware, rate_limit_middleware
from utils.cache_manager import cache_manager


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""

    app = FastAPI(
        title="News Aggregator API", description="新闻聚合分析系统API", version="1.0.0"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware
    app.middleware("http")(logging_middleware)
    app.middleware("http")(rate_limit_middleware)

    # Include routers
    app.include_router(articles.router, prefix="/api/v1/articles", tags=["articles"])
    app.include_router(trends.router, prefix="/api/v1/trends", tags=["trends"])
    app.include_router(reports.router, prefix="/api/v1/reports", tags=["reports"])
    app.include_router(health.router, prefix="/api/v1/health", tags=["health"])
    app.include_router(export.router, prefix="/api/v1/export", tags=["export"])
    app.include_router(metrics.router, prefix="/api/v1/metrics", tags=["metrics"])
    app.include_router(
        comparative.router, prefix="/api/v1/comparative", tags=["comparative"]
    )

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "News Aggregator API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    # Startup event
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting News Aggregator API...")

        # Test Redis connection
        try:
            cache_manager.redis_client.ping()
            logger.info("Redis connection verified")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")

        logger.success("API startup completed")

    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down API...")
        # Close Redis connection
        cache_manager.redis_client.close()
        logger.info("API shutdown completed")

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("API_DEBUG", "false").lower() == "true"

    uvicorn.run("api.main:app", host=host, port=port, reload=debug, log_level="info")

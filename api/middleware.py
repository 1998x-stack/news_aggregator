"""
FastAPI middleware components
"""
import time
from typing import Callable
from fastapi import Request, Response
from loguru import logger
from utils.cache_manager import cache_manager, RateLimitExceededError
from utils.metrics import metrics_collector


async def logging_middleware(request: Request, call_next: Callable):
    """请求日志中间件"""
    start_time = time.time()
    
    # 获取客户端信息
    client_host = request.client.host if request.client else "unknown"
    method = request.method
    url = str(request.url)
    path = request.url.path
    
    # 记录请求开始
    logger.info(f"Request started: {method} {url} from {client_host}")
    
    try:
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        status_code = response.status_code
        
        # 记录请求完成
        logger.info(f"Request completed: {method} {url} - Status: {status_code} - Time: {process_time:.3f}s")
        
        # 记录HTTP指标
        metrics_collector.record_http_request(
            method=method,
            endpoint=path,
            status_code=status_code,
            duration=process_time
        )
        
        # 添加处理时间到响应头
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request failed: {method} {url} - Error: {str(e)} - Time: {process_time:.3f}s")
        
        # 记录错误指标
        metrics_collector.record_http_request(
            method=method,
            endpoint=path,
            status_code=500,
            duration=process_time
        )
        raise


async def rate_limit_middleware(request: Request, call_next: Callable):
    """速率限制中间件"""
    # 获取客户端标识
    client_host = request.client.host if request.client else "unknown"
    path = request.url.path
    
    # 为健康检查端点跳过速率限制
    if path.startswith("/api/v1/health"):
        return await call_next(request)
    
    # 生成速率限制键
    rate_key = f"middleware:rate_limit:{client_host}:{path}"
    
    # 检查速率限制 (100 requests per minute)
    allowed, remaining = cache_manager.get_rate_limit(rate_key, limit=100, window=60)
    
    if not allowed:
        logger.warning(f"Rate limit exceeded for {client_host} on {path}")
        return Response(
            content="Rate limit exceeded. Please try again later.",
            status_code=429,
            headers={
                "X-RateLimit-Limit": "100",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": "60",
            }
        )
    
    # 执行请求
    response = await call_next(request)
    
    # 添加速率限制头
    response.headers["X-RateLimit-Limit"] = "100"
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Window"] = "60"
    
    return response


def setup_logging():
    """配置日志"""
    import sys
    import os
    
    # 移除默认处理器
    logger.remove()
    
    # 控制台日志
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>
    )
    
    # 文件日志
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "api.log",
        level="INFO",
        rotation="500 MB",
        retention="10 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {name}:{function}:{line} - {message}"
    )
    
    # 错误日志
    logger.add(
        log_dir / "api_errors.log",
        level="ERROR",
        rotation="100 MB",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {name}:{function}:{line} - {message}"
    )


# 配置日志
setup_logging()

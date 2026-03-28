"""
Redis cache manager for API responses and rate limiting
"""

import json
import hashlib
from typing import Any, Optional, Callable
from functools import wraps
import redis
from loguru import logger

from config.database_config import load_redis_config


class CacheManager:
    """Redis缓存管理器"""

    def __init__(self):
        config = load_redis_config()
        self.redis_client = redis.Redis(
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        self._test_connection()

    def _test_connection(self):
        """测试Redis连接"""
        try:
            self.redis_client.ping()
            logger.info(
                f"Redis connected: {self.redis_client.connection_pool.connection_kwargs['host']}:{self.redis_client.connection_pool.connection_kwargs['port']}"
            )
        except redis.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        try:
            value = self.redis_client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None

    def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """设置缓存值"""
        try:
            serialized = json.dumps(value, ensure_ascii=False)
            return self.redis_client.setex(key, expire, serialized)
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """删除缓存键"""
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """删除匹配模式的键"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache clear pattern error for pattern {pattern}: {e}")
            return 0

    def get_rate_limit(
        self, key: str, limit: int, window: int = 60
    ) -> tuple[bool, int]:
        """
        检查速率限制

        Returns:
            tuple: (allowed, remaining_attempts)
        """
        try:
            current = self.redis_client.get(key)
            if current is None:
                self.redis_client.setex(key, window, 1)
                return True, limit - 1

            current = int(current)
            if current >= limit:
                return False, 0

            self.redis_client.incr(key)
            return True, limit - (current + 1)

        except Exception as e:
            logger.error(f"Rate limit error for key {key}: {e}")
            return True, limit

    def generate_cache_key(self, prefix: str, *args) -> str:
        """生成缓存键"""
        key_parts = [prefix]
        for arg in args:
            if isinstance(arg, dict):
                arg_str = json.dumps(arg, sort_keys=True)
            else:
                arg_str = str(arg)
            key_parts.append(arg_str)

        key = ":".join(key_parts)
        # 如果键太长，使用哈希
        if len(key) > 200:
            key_hash = hashlib.md5(key.encode()).hexdigest()
            key = f"{prefix}:{key_hash[:16]}"

        return key


# 全局缓存管理器实例
cache_manager = CacheManager()


def cache_response(expire: int = 3600, key_prefix: str = "api"):
    """
    API响应缓存装饰器

    Args:
        expire: 缓存过期时间（秒）
        key_prefix: 缓存键前缀
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = cache_manager.generate_cache_key(
                key_prefix, func.__name__, *args, **kwargs
            )

            # 尝试从缓存获取
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_result

            # 执行函数
            result = func(*args, **kwargs)

            # 缓存结果
            cache_manager.set(cache_key, result, expire)
            logger.debug(f"Cache set: {cache_key}")

            return result

        return wrapper

    return decorator


def rate_limit(limit: int, window: int = 60, key_prefix: str = "rate_limit"):
    """
    速率限制装饰器

    Args:
        limit: 时间窗口内允许的最大请求数
        window: 时间窗口（秒）
        key_prefix: 速率限制键前缀
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成速率限制键（使用客户端IP或用户ID）
            client_id = kwargs.get("client_id", "default")
            rate_key = f"{key_prefix}:{func.__name__}:{client_id}"

            allowed, remaining = cache_manager.get_rate_limit(rate_key, limit, window)

            if not allowed:
                raise RateLimitExceededError(
                    f"Rate limit exceeded. Limit: {limit} requests per {window} seconds"
                )

            # 执行函数
            result = func(*args, **kwargs)

            # 添加速率限制信息到响应头
            if isinstance(result, dict):
                result["rate_limit"] = {
                    "limit": limit,
                    "remaining": remaining,
                    "window": window,
                }

            return result

        return wrapper

    return decorator


class RateLimitExceededError(Exception):
    """速率限制超出异常"""

    pass


class CacheKeys:
    """缓存键常量"""

    ARTICLE = "article"
    ARTICLES = "articles"
    TREND = "trend"
    TRENDS = "trends"
    REPORT = "report"
    REPORTS = "reports"
    SEARCH = "search"
    STATS = "stats"

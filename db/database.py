"""
Database connection and session management
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from loguru import logger
from contextlib import contextmanager

from db.models import Base


class DatabaseManager:
    """数据库管理器"""

    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._init_database()

    def _init_database(self):
        """初始化数据库连接"""
        # 从环境变量获取数据库配置
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "news_aggregator")
        db_user = os.getenv("DB_USER", "postgres")
        db_password = os.getenv("DB_PASSWORD", "postgres")

        # 构建数据库URL
        database_url = (
            f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )

        # 创建数据库引擎
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_pre_ping=True,
            echo=False,
        )

        # 创建会话工厂
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
        )

        logger.info(f"Database initialized: {db_host}:{db_port}/{db_name}")

    def create_tables(self):
        """创建数据库表"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise

    def drop_tables(self):
        """删除数据库表（谨慎使用）"""
        try:
            Base.metadata.drop_all(self.engine)
            logger.warning("Database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise

    @contextmanager
    def get_session(self) -> Session:
        """获取数据库会话（上下文管理器）"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_db_session(self) -> Session:
        """获取数据库会话（手动管理）"""
        return self.SessionLocal()


# 全局数据库管理器实例
db_manager = DatabaseManager()


def get_db():
    """FastAPI依赖注入用"""
    db = db_manager.SessionLocal()
    try:
        yield db
    finally:
        db.close()

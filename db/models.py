"""
Database models for news aggregator system
"""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    Boolean,
    JSON,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Article(Base):
    """新闻文章模型"""

    __tablename__ = "articles"

    id = Column(String(255), primary_key=True)  # 原始ID (hn_123, rss_uuid, etc)
    title = Column(String(1000), nullable=False, index=True)
    content = Column(Text)  # 完整内容
    summary = Column(Text)  # 摘要
    url = Column(String(2000))
    source = Column(String(255), index=True)  # HackerNews, RSS, 新浪财经
    author = Column(String(500))

    # 分类信息
    category = Column(String(100), index=True)  # ai_ml, programming, etc
    importance = Column(Integer, default=3)  # 1-5
    classification_confidence = Column(Float, default=0.0)
    classification_reason = Column(Text)

    # 5W2H信息抽取
    extracted_what = Column(Text)
    extracted_who = Column(Text)
    extracted_when = Column(Text)
    extracted_where = Column(Text)
    extracted_why = Column(Text)
    extracted_how = Column(Text)
    extracted_how_much = Column(Text)

    # 元数据
    publish_time = Column(DateTime, index=True)
    collect_time = Column(DateTime, default=func.now())
    extract_time = Column(DateTime)
    classify_time = Column(DateTime)

    # 原始数据
    raw_data = Column(JSON)  # 原始采集数据
    metadata = Column(JSON)  # 额外元数据

    # 分析结果
    sentiment = Column(String(50))  # positive, negative, neutral
    sentiment_score = Column(Float)
    entities = Column(JSON)  # 识别的实体
    keywords = Column(JSON)  # 关键词

    # 统计信息
    score = Column(Integer, default=0)  # 原始分数 (HN score等)
    comments_count = Column(Integer, default=0)

    # 状态
    is_processed = Column(Boolean, default=False)
    is_classified = Column(Boolean, default=False)
    is_extracted = Column(Boolean, default=False)

    # 关系
    trends = relationship("TrendArticle", back_populates="article")

    # 索引
    __table_args__ = (
        Index("idx_articles_category_importance", "category", "importance"),
        Index("idx_articles_publish_time", "publish_time"),
        Index("idx_articles_source", "source"),
    )

    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "url": self.url,
            "source": self.source,
            "author": self.author,
            "category": self.category,
            "importance": self.importance,
            "classification_confidence": self.classification_confidence,
            "classification_reason": self.classification_reason,
            "extracted_what": self.extracted_what,
            "extracted_who": self.extracted_who,
            "extracted_when": self.extracted_when,
            "extracted_where": self.extracted_where,
            "extracted_why": self.extracted_why,
            "extracted_how": self.extracted_how,
            "extracted_how_much": self.extracted_how_much,
            "publish_time": self.publish_time.isoformat()
            if self.publish_time
            else None,
            "collect_time": self.collect_time.isoformat()
            if self.collect_time
            else None,
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
            "entities": self.entities,
            "keywords": self.keywords,
            "score": self.score,
            "comments_count": self.comments_count,
            "is_processed": self.is_processed,
            "is_classified": self.is_classified,
            "is_extracted": self.is_extracted,
        }


class Trend(Base):
    """趋势话题模型"""

    __tablename__ = "trends"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(500), nullable=False, index=True)  # 话题名称
    category = Column(String(100), index=True)  # 类别

    # 热度信息
    heat_score = Column(Float, default=0.0)
    mention_count = Column(Integer, default=0)

    # 时间范围
    start_date = Column(DateTime, index=True)
    end_date = Column(DateTime, index=True)

    # 趋势类型
    trend_type = Column(String(50), index=True)  # hot, emerging, declining

    # 相关实体和关键词
    related_entities = Column(JSON)  # 相关实体
    related_keywords = Column(JSON)  # 相关关键词

    # 摘要和展望
    summary = Column(Text)
    outlook = Column(Text)

    # 元数据
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # 关系
    articles = relationship("TrendArticle", back_populates="trend")

    # 索引
    __table_args__ = (
        Index("idx_trends_heat_score", "heat_score"),
        Index("idx_trends_date_range", "start_date", "end_date"),
    )

    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "heat_score": self.heat_score,
            "mention_count": self.mention_count,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "trend_type": self.trend_type,
            "related_entities": self.related_entities,
            "related_keywords": self.related_keywords,
            "summary": self.summary,
            "outlook": self.outlook,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class TrendArticle(Base):
    """趋势-文章关联表"""

    __tablename__ = "trend_articles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trend_id = Column(Integer, ForeignKey("trends.id"), nullable=False, index=True)
    article_id = Column(
        String(255), ForeignKey("articles.id"), nullable=False, index=True
    )

    # 关联权重
    relevance_score = Column(Float, default=1.0)

    # 关系
    trend = relationship("Trend", back_populates="articles")
    article = relationship("Article", back_populates="trends")

    # 唯一约束
    __table_args__ = (
        Index("idx_trend_articles_unique", "trend_id", "article_id", unique=True),
    )


class Report(Base):
    """报告模型"""

    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_type = Column(
        String(100), nullable=False, index=True
    )  # daily, category, timeline
    report_date = Column(DateTime, nullable=False, index=True)

    # 报告内容
    title = Column(String(500))
    content = Column(JSON)  # 报告内容（结构化数据）
    markdown_content = Column(Text)  # Markdown格式内容
    html_content = Column(Text)  # HTML格式内容

    # 元数据
    format = Column(String(50))  # markdown, json, html
    file_path = Column(String(1000))  # 文件路径
    file_size = Column(Integer)  # 文件大小（字节）

    # 统计信息
    total_articles = Column(Integer, default=0)
    hot_topics_count = Column(Integer, default=0)
    categories_count = Column(Integer, default=0)
    sources_count = Column(Integer, default=0)

    # 生成信息
    generated_at = Column(DateTime, default=func.now())
    generation_time = Column(Float)  # 生成耗时（秒）

    # 状态
    is_generated = Column(Boolean, default=False)
    is_delivered = Column(Boolean, default=False)  # 是否已发送

    # 索引
    __table_args__ = (Index("idx_reports_date_type", "report_date", "report_type"),)

    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "report_type": self.report_type,
            "report_date": self.report_date.isoformat() if self.report_date else None,
            "title": self.title,
            "content": self.content,
            "format": self.format,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "total_articles": self.total_articles,
            "hot_topics_count": self.hot_topics_count,
            "categories_count": self.categories_count,
            "sources_count": self.sources_count,
            "generated_at": self.generated_at.isoformat()
            if self.generated_at
            else None,
            "generation_time": self.generation_time,
            "is_generated": self.is_generated,
            "is_delivered": self.is_delivered,
        }


class Entity(Base):
    """实体模型（公司、人物、产品等）"""

    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(500), nullable=False, index=True)  # 实体名称
    entity_type = Column(
        String(100), index=True
    )  # company, person, product, location, etc

    # 实体信息
    description = Column(Text)
    aliases = Column(JSON)  # 别名

    # 统计信息
    mention_count = Column(Integer, default=0)
    first_mentioned = Column(DateTime)
    last_mentioned = Column(DateTime)

    # 元数据
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # 索引
    __table_args__ = (
        Index("idx_entities_type", "entity_type"),
        Index("idx_entities_mention_count", "mention_count"),
    )

    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "aliases": self.aliases,
            "mention_count": self.mention_count,
            "first_mentioned": self.first_mentioned.isoformat()
            if self.first_mentioned
            else None,
            "last_mentioned": self.last_mentioned.isoformat()
            if self.last_mentioned
            else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

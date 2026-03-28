"""
Tests for database models
"""

import pytest
from datetime import datetime
from db.models import Article, Trend, Report, Entity


class TestArticleModel:
    """Tests for Article model"""

    def test_article_creation(self):
        """Test creating an Article instance"""
        article = Article(
            id="test_123",
            title="Test Article",
            content="Test content",
            url="https://example.com",
            source="HackerNews",
            category="ai_ml",
            importance=4,
            publish_time=datetime(2024, 1, 15, 10, 30),
            collect_time=datetime(2024, 1, 15, 11, 0),
        )

        assert article.id == "test_123"
        assert article.title == "Test Article"
        assert article.content == "Test content"
        assert article.url == "https://example.com"
        assert article.source == "HackerNews"
        assert article.category == "ai_ml"
        assert article.importance == 4
        assert article.publish_time == datetime(2024, 1, 15, 10, 30)
        assert article.collect_time == datetime(2024, 1, 15, 11, 0)
        assert not article.is_processed
        assert not article.is_classified
        assert not article.is_extracted

    def test_article_to_dict(self):
        """Test Article serialization"""
        article = Article(
            id="test_123",
            title="Test Article",
            content="Test content",
            url="https://example.com",
            source="HackerNews",
            category="ai_ml",
            importance=4,
            publish_time=datetime(2024, 1, 15, 10, 30),
            collect_time=datetime(2024, 1, 15, 11, 0),
            sentiment="positive",
            sentiment_score=0.85,
            entities=[{"name": "OpenAI", "type": "company"}],
            keywords=[{"word": "AI", "score": 0.9}],
            score=100,
            comments_count=25,
        )

        article_dict = article.to_dict()

        assert article_dict["id"] == "test_123"
        assert article_dict["title"] == "Test Article"
        assert article_dict["content"] == "Test content"
        assert article_dict["url"] == "https://example.com"
        assert article_dict["source"] == "HackerNews"
        assert article_dict["category"] == "ai_ml"
        assert article_dict["importance"] == 4
        assert article_dict["sentiment"] == "positive"
        assert article_dict["sentiment_score"] == 0.85
        assert article_dict["entities"][0]["name"] == "OpenAI"
        assert article_dict["keywords"][0]["word"] == "AI"
        assert article_dict["score"] == 100
        assert article_dict["comments_count"] == 25
        assert article_dict["publish_time"] == "2024-01-15T10:30:00"
        assert article_dict["collect_time"] == "2024-01-15T11:00:00"
        assert not article_dict["is_processed"]
        assert not article_dict["is_classified"]
        assert not article_dict["is_extracted"]

    def test_article_with_extracted_info(self):
        """Test Article with 5W2H extraction"""
        article = Article(
            id="test_123",
            title="Test Article",
            extracted_what="AI breakthrough",
            extracted_who="Researchers",
            extracted_when="2024-01-15",
            extracted_where="San Francisco",
            extracted_why="To improve AI",
            extracted_how="New architecture",
            extracted_how_much="Significant improvement",
        )

        article_dict = article.to_dict()

        assert article_dict["extracted_what"] == "AI breakthrough"
        assert article_dict["extracted_who"] == "Researchers"
        assert article_dict["extracted_when"] == "2024-01-15"
        assert article_dict["extracted_where"] == "San Francisco"
        assert article_dict["extracted_why"] == "To improve AI"
        assert article_dict["extracted_how"] == "New architecture"
        assert article_dict["extracted_how_much"] == "Significant improvement"


class TestTrendModel:
    """Tests for Trend model"""

    def test_trend_creation(self):
        """Test creating a Trend instance"""
        trend = Trend(
            name="AI Breakthrough",
            category="ai_ml",
            heat_score=85.5,
            mention_count=15,
            start_date=datetime(2024, 1, 15),
            end_date=datetime(2024, 1, 16),
            trend_type="hot",
            related_entities=["OpenAI", "Google"],
            related_keywords=["AI", "machine learning"],
            summary="AI technology advancing rapidly",
            outlook="Expected to continue growing",
        )

        assert trend.name == "AI Breakthrough"
        assert trend.category == "ai_ml"
        assert trend.heat_score == 85.5
        assert trend.mention_count == 15
        assert trend.trend_type == "hot"
        assert trend.related_entities == ["OpenAI", "Google"]
        assert trend.related_keywords == ["AI", "machine learning"]
        assert trend.summary == "AI technology advancing rapidly"
        assert trend.outlook == "Expected to continue growing"

    def test_trend_to_dict(self):
        """Test Trend serialization"""
        trend = Trend(
            id=1,
            name="AI Breakthrough",
            category="ai_ml",
            heat_score=85.5,
            mention_count=15,
            start_date=datetime(2024, 1, 15),
            end_date=datetime(2024, 1, 16),
            trend_type="hot",
        )

        trend_dict = trend.to_dict()

        assert trend_dict["id"] == 1
        assert trend_dict["name"] == "AI Breakthrough"
        assert trend_dict["category"] == "ai_ml"
        assert trend_dict["heat_score"] == 85.5
        assert trend_dict["mention_count"] == 15
        assert trend_dict["start_date"] == "2024-01-15T00:00:00"
        assert trend_dict["end_date"] == "2024-01-16T00:00:00"
        assert trend_dict["trend_type"] == "hot"


class TestReportModel:
    """Tests for Report model"""

    def test_report_creation(self):
        """Test creating a Report instance"""
        report = Report(
            report_type="daily",
            report_date=datetime(2024, 1, 15),
            title="Daily News Report - 2024-01-15",
            format="markdown",
            file_path="/path/to/report.md",
            file_size=1024,
            total_articles=50,
            hot_topics_count=10,
            categories_count=5,
            sources_count=3,
            is_generated=True,
            is_delivered=False,
        )

        assert report.report_type == "daily"
        assert report.report_date == datetime(2024, 1, 15)
        assert report.title == "Daily News Report - 2024-01-15"
        assert report.format == "markdown"
        assert report.file_path == "/path/to/report.md"
        assert report.file_size == 1024
        assert report.total_articles == 50
        assert report.hot_topics_count == 10
        assert report.categories_count == 5
        assert report.sources_count == 3
        assert report.is_generated
        assert not report.is_delivered

    def test_report_to_dict(self):
        """Test Report serialization"""
        report = Report(
            id=1,
            report_type="daily",
            report_date=datetime(2024, 1, 15),
            title="Daily Report",
            total_articles=50,
        )

        report_dict = report.to_dict()

        assert report_dict["id"] == 1
        assert report_dict["report_type"] == "daily"
        assert report_dict["report_date"] == "2024-01-15T00:00:00"
        assert report_dict["title"] == "Daily Report"
        assert report_dict["total_articles"] == 50
        assert report_dict["is_generated"] == False  # Default value


class TestEntityModel:
    """Tests for Entity model"""

    def test_entity_creation(self):
        """Test creating an Entity instance"""
        entity = Entity(
            name="OpenAI",
            entity_type="company",
            description="AI research company",
            aliases=["Open AI", "OpenAI Inc."],
            mention_count=100,
            first_mentioned=datetime(2023, 1, 1),
            last_mentioned=datetime(2024, 1, 15),
        )

        assert entity.name == "OpenAI"
        assert entity.entity_type == "company"
        assert entity.description == "AI research company"
        assert entity.aliases == ["Open AI", "OpenAI Inc."]
        assert entity.mention_count == 100
        assert entity.first_mentioned == datetime(2023, 1, 1)
        assert entity.last_mentioned == datetime(2024, 1, 15)

    def test_entity_to_dict(self):
        """Test Entity serialization"""
        entity = Entity(
            id=1,
            name="OpenAI",
            entity_type="company",
            mention_count=100,
            first_mentioned=datetime(2023, 1, 1),
            last_mentioned=datetime(2024, 1, 15),
        )

        entity_dict = entity.to_dict()

        assert entity_dict["id"] == 1
        assert entity_dict["name"] == "OpenAI"
        assert entity_dict["entity_type"] == "company"
        assert entity_dict["mention_count"] == 100
        assert entity_dict["first_mentioned"] == "2023-01-01T00:00:00"
        assert entity_dict["last_mentioned"] == "2024-01-15T00:00:00"


class TestModelRelationships:
    """Tests for model relationships"""

    def test_article_trend_relationship(self):
        """Test Article-Trend relationship"""
        article = Article(id="article_123", title="Test Article")
        trend = Trend(id=1, name="AI Trend")

        # In a real scenario, these would be connected via TrendArticle
        # For now, we test that the models can be created independently
        assert article.id == "article_123"
        assert trend.id == 1
        assert trend.name == "AI Trend"

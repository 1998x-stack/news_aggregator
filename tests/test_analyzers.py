"""
Tests for analyzer modules (classifier, extractor, trend_analyzer)
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from analyzers.classifier import ContentClassifier, RuleBasedClassifier
from analyzers.extractor import ContentExtractorLLM
from analyzers.trend_analyzer import TrendAnalyzer, TrendReport, TrendItem
from analyzers.executive_summarizer import ExecutiveSummarizer, ExecutiveSummary


class TestContentClassifier:
    """Tests for ContentClassifier"""

    def test_rule_based_classifier_basic(self):
        """Test basic rule-based classification"""
        classifier = RuleBasedClassifier()

        # Test AI/ML classification
        article = {
            "title": "New AI breakthrough in machine learning",
            "content": "Researchers developed a new neural network architecture",
        }

        result = classifier.classify(article)
        assert result.category == "ai_ml"
        assert result.importance >= 3
        assert result.confidence > 0.7

    def test_rule_based_classifier_programming(self):
        """Test programming category classification"""
        classifier = RuleBasedClassifier()

        article = {
            "title": "Python 3.11 performance improvements",
            "content": "New features in Python programming language",
        }

        result = classifier.classify(article)
        assert result.category == "programming"

    def test_rule_based_classifier_default(self):
        """Test default category for unknown content"""
        classifier = RuleBasedClassifier()

        article = {
            "title": "Random article about nothing specific",
            "content": "This is just some random content",
        }

        result = classifier.classify(article)
        assert result.category == "other"
        assert result.importance == 3  # Default importance


class TestContentExtractorLLM:
    """Tests for ContentExtractorLLM"""

    @pytest.fixture
    def mock_ollama_client(self):
        """Mock Ollama client"""
        client = Mock()
        client.generate = Mock(
            return_value={
                "response": json.dumps(
                    {
                        "what": "AI breakthrough",
                        "who": "Researchers at OpenAI",
                        "when": "2024-01-15",
                        "where": "San Francisco",
                        "why": "To improve AI capabilities",
                        "how": "Using new neural network architecture",
                        "how_much": "Significant improvement in performance",
                    }
                )
            }
        )
        return client

    def test_extract_info_success(self, mock_ollama_client):
        """Test successful 5W2H extraction"""
        extractor = ContentExtractorLLM()
        extractor.client = mock_ollama_client

        article = {
            "title": "AI breakthrough announced",
            "content": "Researchers at OpenAI in San Francisco announced a significant AI breakthrough on 2024-01-15 using new neural network architecture to improve AI capabilities, resulting in significant performance improvements.",
        }

        result = extractor.extract_info(article)

        assert result.extracted_what == "AI breakthrough"
        assert result.extracted_who == "Researchers at OpenAI"
        assert result.extracted_when == "2024-01-15"
        assert result.extracted_where == "San Francisco"
        assert result.extracted_why == "To improve AI capabilities"
        assert result.extracted_how == "Using new neural network architecture"
        assert result.extracted_how_much == "Significant improvement in performance"

    def test_extract_info_invalid_json(self, mock_ollama_client):
        """Test handling of invalid JSON from LLM"""
        mock_dashscope_client.generate = Mock(
            return_value={"response": "This is not valid JSON"}
        )

        extractor = ContentExtractorLLM()
        extractor.client = mock_ollama_client

        article = {"title": "Test article", "content": "Test content"}

        result = extractor.extract_info(article)

        # Should handle gracefully and return empty/default values
        assert result.extracted_what is None
        assert result.extracted_who is None


class TestTrendAnalyzer:
    """Tests for TrendAnalyzer"""

    @pytest.fixture
    def sample_articles(self):
        """Sample articles for trend analysis"""
        return [
            {
                "id": f"article_{i}",
                "title": f"AI breakthrough {i}",
                "content": f"Content about AI {i}",
                "category": "ai_ml",
                "importance": 4,
                "keywords": ["AI", "breakthrough", "technology"],
            }
            for i in range(10)
        ]

    def test_identify_hot_topics(self, sample_articles):
        """Test hot topic identification"""
        analyzer = TrendAnalyzer()

        # Add some duplicate keywords to create hot topics
        for article in sample_articles[:5]:
            article["keywords"] = ["AI", "machine learning", "neural networks"]

        trend_report = analyzer.analyze(sample_articles)

        assert len(trend_report.hot_topics) > 0
        assert any(topic.name == "AI" for topic in trend_report.hot_topics)

    def test_category_distribution(self, sample_articles):
        """Test category distribution calculation"""
        analyzer = TrendAnalyzer()

        trend_report = analyzer.analyze(sample_articles)

        assert "ai_ml" in trend_report.category_distribution
        assert trend_report.category_distribution["ai_ml"] == 10

    def test_source_distribution(self, sample_articles):
        """Test source distribution calculation"""
        analyzer = TrendAnalyzer()

        # Add source to articles
        for i, article in enumerate(sample_articles):
            article["source"] = "HackerNews" if i < 5 else "RSS"

        trend_report = analyzer.analyze(sample_articles)

        assert "HackerNews" in trend_report.source_distribution
        assert "RSS" in trend_report.source_distribution


class TestExecutiveSummarizer:
    """Tests for ExecutiveSummarizer"""

    @pytest.fixture
    def mock_trend_report(self):
        """Mock trend report"""
        report = Mock(spec=TrendReport)
        report.total_articles = 10
        report.hot_topics = [
            Mock(
                name="AI Breakthrough",
                heat_score=85.5,
                mention_count=15,
                category="ai_ml",
            ),
            Mock(
                name="Quantum Computing",
                heat_score=72.3,
                mention_count=8,
                category="technology",
            ),
        ]
        report.category_distribution = {"ai_ml": 6, "technology": 4}
        report.source_distribution = {"HackerNews": 7, "RSS": 3}
        report.emerging_trends = []
        report.key_events = []
        report.outlook = "AI continues to advance rapidly"
        report.recommendations = [
            "Invest in AI research",
            "Monitor quantum developments",
        ]
        return report

    @pytest.fixture
    def sample_articles(self):
        """Sample articles for executive summary"""
        return [
            {
                "title": "AI breakthrough announced",
                "source": "HackerNews",
                "category": "ai_ml",
                "importance": 5,
                "sentiment": "positive",
            },
            {
                "title": "Quantum computing milestone",
                "source": "RSS",
                "category": "technology",
                "importance": 4,
                "sentiment": "neutral",
            },
        ]

    def test_generate_summary_success(self, mock_trend_report, sample_articles):
        """Test successful executive summary generation"""
        summarizer = ExecutiveSummarizer()

        # Mock the LLM response
        summarizer.ollama_client = Mock()
        summarizer.dashscope_client.generate = Mock(
            return_value={
                "response": json.dumps(
                    {
                        "key_insights": [
                            "AI is advancing rapidly",
                            "Quantum computing showing promise",
                        ],
                        "market_impact": "Significant disruption expected in tech industry",
                        "recommended_actions": [
                            "Invest in AI",
                            "Monitor quantum developments",
                        ],
                        "risk_factors": [
                            "Regulatory uncertainty",
                            "Technical challenges",
                        ],
                        "opportunities": [
                            "New market opportunities",
                            "Partnership potential",
                        ],
                    }
                )
            }
        )

        summary = summarizer.generate_summary(
            mock_trend_report, sample_articles, "2024-01-15"
        )

        assert len(summary.key_insights) == 2
        assert "AI is advancing rapidly" in summary.key_insights
        assert (
            summary.market_impact == "Significant disruption expected in tech industry"
        )
        assert len(summary.recommended_actions) == 2
        assert len(summary.risk_factors) == 2
        assert len(summary.opportunities) == 2

    def test_generate_summary_llm_error(self, mock_trend_report, sample_articles):
        """Test graceful handling of LLM errors"""
        summarizer = ExecutiveSummarizer()

        # Mock LLM to raise an exception
        summarizer.ollama_client = Mock()
        summarizer.dashscope_client.generate = Mock(
            side_effect=Exception("LLM service unavailable")
        )

        summary = summarizer.generate_summary(
            mock_trend_report, sample_articles, "2024-01-15"
        )

        # Should return default summary
        assert len(summary.key_insights) > 0
        assert "AI" in summary.key_insights[0]
        assert summary.market_impact == "Market impact analysis requires LLM processing"


class TestTrendReport:
    """Tests for TrendReport data structure"""

    def test_trend_report_creation(self):
        """Test creating a TrendReport"""
        from analyzers.trend_analyzer import TrendItem, IndustryDynamic

        hot_topics = [
            TrendItem(
                name="AI Breakthrough",
                category="ai_ml",
                heat_score=85.5,
                mention_count=15,
                related_entities=["OpenAI", "Google"],
                related_keywords=["AI", "machine learning"],
            )
        ]

        report = TrendReport(
            total_articles=10,
            hot_topics=hot_topics,
            emerging_trends=[],
            category_distribution={"ai_ml": 6, "technology": 4},
            source_distribution={"HackerNews": 7, "RSS": 3},
            key_events=[],
            industry_dynamics=[],
            recommendations=["Invest in AI"],
            outlook="AI is advancing rapidly",
        )

        assert report.total_articles == 10
        assert len(report.hot_topics) == 1
        assert report.hot_topics[0].name == "AI Breakthrough"
        assert report.category_distribution["ai_ml"] == 6

    def test_trend_report_to_dict(self):
        """Test TrendReport serialization"""
        from analyzers.trend_analyzer import TrendItem

        report = TrendReport(
            total_articles=5,
            hot_topics=[
                TrendItem(
                    name="Test Trend",
                    category="test",
                    heat_score=50.0,
                    mention_count=10,
                )
            ],
            emerging_trends=[],
            category_distribution={},
            source_distribution={},
            key_events=[],
            industry_dynamics=[],
            recommendations=[],
            outlook="",
        )

        report_dict = report.to_dict()

        assert report_dict["total_articles"] == 5
        assert len(report_dict["hot_topics"]) == 1
        assert report_dict["hot_topics"][0]["name"] == "Test Trend"

"""
Test configuration and fixtures for News Aggregator
"""

import pytest
from unittest.mock import Mock, MagicMock
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response"""

    def _create_response(content, success=True):
        return {"response": content, "success": success, "model": "qwen3:4b"}

    return _create_response


@pytest.fixture
def sample_article_data():
    """Sample article data for testing"""
    return {
        "id": "test_hn_123",
        "title": "Test Article About AI Breakthrough",
        "content": "Researchers have developed a new AI model that achieves state-of-the-art results.",
        "url": "https://example.com/ai-breakthrough",
        "source": "HackerNews",
        "category": "ai_ml",
        "importance": 4,
        "sentiment": None,
        "entities": None,
        "keywords": None,
    }


@pytest.fixture
def sample_trend_data():
    """Sample trend data for testing"""
    return {
        "name": "AI Breakthrough",
        "category": "ai_ml",
        "heat_score": 85.5,
        "mention_count": 15,
        "trend_type": "hot",
    }


@pytest.fixture
def mock_sentiment_response():
    """Mock sentiment analysis response from LLM"""
    return json.dumps(
        {
            "sentiment": "positive",
            "score": 0.85,
            "reason": "Article describes significant positive development in AI",
        }
    )


@pytest.fixture
def mock_entities_response():
    """Mock entity extraction response from LLM"""
    return json.dumps(
        {
            "entities": [
                {"name": "OpenAI", "type": "company", "relevance_score": 0.95},
                {"name": "John Smith", "type": "person", "relevance_score": 0.72},
            ]
        }
    )


@pytest.fixture
def mock_keywords_response():
    """Mock keyword extraction response from LLM"""
    return json.dumps(
        {
            "keywords": [
                {"word": "AI", "relevance_score": 0.92},
                {"word": "breakthrough", "relevance_score": 0.88},
            ]
        }
    )


@pytest.fixture
def mock_executive_summary_response():
    """Mock executive summary response from LLM"""
    return json.dumps(
        {
            "key_insights": [
                "AI technology is advancing rapidly",
                "New models show significant improvements",
            ],
            "market_impact": "This breakthrough could disrupt the AI industry significantly.",
            "recommended_actions": [
                "Monitor OpenAI developments",
                "Invest in AI infrastructure",
            ],
            "risk_factors": ["Regulatory uncertainty", "Competitive pressure"],
            "opportunities": ["New market opportunities", "Technology partnerships"],
        }
    )


@pytest.fixture
def mock_redis_client():
    """Mock Redis client"""
    mock_client = Mock()
    mock_client.get = Mock(return_value=None)
    mock_client.setex = Mock(return_value=True)
    mock_client.incr = Mock(return_value=1)
    mock_client.ping = Mock(return_value=True)
    mock_client.keys = Mock(return_value=[])
    mock_client.delete = Mock(return_value=0)
    return mock_client


@pytest.fixture
def mock_db_session():
    """Mock database session"""
    session = Mock()
    session.query = Mock()
    session.add = Mock()
    session.commit = Mock()
    session.rollback = Mock()
    session.close = Mock()
    return session


@pytest.fixture
def temp_env_vars():
    """Temporary environment variables for testing"""
    original_env = dict(os.environ)

    # Set test environment variables
    os.environ.update(
        {
            "DB_HOST": "localhost",
            "DB_PORT": "5432",
            "DB_NAME": "test_news",
            "DB_USER": "test_user",
            "DB_PASSWORD": "test_pass",
            "REDIS_HOST": "localhost",
            "REDIS_PORT": "6379",
            "REDIS_DB": "0",
            "OLLAMA_HOST": "localhost",
            "OLLAMA_PORT": "11434",
        }
    )

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

"""
Basic API tests for News Aggregator
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app
from db.database import db_manager

client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data


def test_articles_endpoint():
    """Test articles list endpoint"""
    response = client.get("/api/v1/articles?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "articles" in data
    assert "total" in data
    assert isinstance(data["articles"], list)


def test_trends_endpoint():
    """Test trends endpoint"""
    response = client.get("/api/v1/trends?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "trends" in data
    assert "total" in data


def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    content = response.text
    assert "http_requests_total" in content


def test_export_csv_endpoint():
    """Test CSV export endpoint"""
    response = client.get("/api/v1/export/articles/csv?limit=3")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    content = response.text
    assert "id,title,source" in content


def test_export_json_endpoint():
    """Test JSON export endpoint"""
    response = client.get("/api/v1/export/articles/json?limit=3")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    data = response.json()
    assert isinstance(data, list)


def test_comparative_today_vs_yesterday():
    """Test today vs yesterday comparison"""
    response = client.get("/api/v1/comparative/today-vs-yesterday")
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "key_insights" in data


def test_api_docs():
    """Test API documentation endpoint"""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "docs" in data


if __name__ == "__main__":
    # Run tests
    test_health_endpoint()
    test_articles_endpoint()
    test_trends_endpoint()
    test_metrics_endpoint()
    test_export_csv_endpoint()
    test_export_json_endpoint()
    test_comparative_today_vs_yesterday()
    test_api_docs()
    test_root_endpoint()
    print("All tests passed!")

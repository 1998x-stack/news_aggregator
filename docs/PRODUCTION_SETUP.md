# News Aggregator - Production Setup Guide

## Overview

This guide provides instructions for setting up the News Aggregator system in a production environment using Docker Compose.

## System Architecture

The production deployment consists of the following services:

- **PostgreSQL**: Primary database for storing articles, trends, and reports
- **Redis**: Cache layer and Celery message broker
- **Ollama**: LLM service for AI-powered analysis
- **API**: FastAPI application serving REST endpoints
- **Celery Worker**: Background task processing (4 concurrent workers)
- **Celery Beat**: Task scheduler for periodic jobs
- **Flower**: Celery monitoring UI

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM available (for Ollama models)
- Ports 8000, 5432, 6379, 5555, 11434 available

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd news_aggregator

# Copy environment template
cp .env.example .env

# Edit .env with your settings (optional for basic setup)
# The default values work for local development
```

### 2. Build and Start Services

```bash
# Build all Docker images
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

### 3. Initialize Database

```bash
# Create database tables
docker-compose exec api python db/init_db.py

# Run data migration (if you have existing JSON data)
docker-compose exec api python db/migrate.py
```

### 4. Download LLM Models

```bash
# Pull required models (this may take several minutes)
docker-compose exec ollama ollama pull qwen2.5:0.5b
docker-compose exec ollama ollama pull qwen3:4b
```

### 5. Verify Installation

```bash
# Check API health
curl http://localhost:8000/api/v1/health

# View API documentation
open http://localhost:8000/docs

# Access Flower monitoring
open http://localhost:5555

# View Prometheus metrics
curl http://localhost:8000/api/v1/metrics
```

## API Endpoints

### Core Endpoints

- **Health Check**: `GET /api/v1/health`
- **Articles**: `GET /api/v1/articles` (with filtering, search, pagination)
- **Trends**: `GET /api/v1/trends` (hot topics, emerging trends)
- **Reports**: `GET /api/v1/reports` (daily, category reports)
- **Export**: `GET /api/v1/export/articles/csv` (data export)
- **Metrics**: `GET /api/v1/metrics` (Prometheus metrics)
- **Docs**: `/docs` (Swagger UI)

### Example API Calls

```bash
# Get articles with filtering
curl "http://localhost:8000/api/v1/articles?category=ai_ml&importance_min=4"

# Get top hot topics
curl "http://localhost:8000/api/v1/trends/hot/top?limit=10"

# Export articles to CSV
curl -O http://localhost:8000/api/v1/export/articles/csv

# Get daily report
curl "http://localhost:8000/api/v1/reports/latest/daily"
```

## Configuration

### Environment Variables

Key environment variables (see `.env.example` for full list):

```bash
# Database
DB_HOST=postgres
DB_PORT=5432
DB_NAME=news_aggregator
DB_USER=postgres
DB_PASSWORD=postgres

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# Ollama
OLLAMA_HOST=ollama
OLLAMA_PORT=11434

# API
API_HOST=0.0.0.0
API_PORT=8000
```

### Custom Configuration

Edit `docker-compose.yml` to customize:
- Resource limits for each service
- Volume mounts for persistent data
- Network configuration
- Environment variables

## Monitoring

### Metrics Collection

The system includes comprehensive Prometheus metrics:

- **HTTP Requests**: Count, duration, status codes
- **Database Queries**: Operation counts and latency
- **Cache Operations**: Hit/miss rates
- **Pipeline Runs**: Success/failure counts
- **LLM Requests**: Model usage and latency
- **System Metrics**: Active connections, article counts

### Viewing Metrics

```bash
# Get all metrics
curl http://localhost:8000/api/v1/metrics

# In Prometheus, add target:
# http://localhost:8000/api/v1/metrics
```

### Health Monitoring

```bash
# Basic health check
curl http://localhost:8000/api/v1/health

# Detailed health with dependencies
curl http://localhost:8000/api/v1/health/details

# Kubernetes readiness check
curl http://localhost:8000/api/v1/health/ready
```

## Background Tasks

### Celery Tasks

The system uses Celery for background processing:

- **Data Collection**: `tasks.pipeline_tasks.collect_data`
- **Analysis**: `tasks.analysis_tasks.analyze_sentiment`
- **Report Generation**: `tasks.report_tasks.generate_daily_report`
- **Entity Extraction**: `tasks.analysis_tasks.extract_entities`

### Monitoring Tasks

Access Flower UI at http://localhost:5555 to:
- View task status and results
- Monitor worker health
- Retry failed tasks
- View task statistics

## Data Export

### Export Formats

The system supports multiple export formats:

- **CSV**: `/api/v1/export/articles/csv`
- **JSON**: `/api/v1/export/articles/json`
- **Complete Reports**: `/api/v1/export/complete/{date}`

### Example Exports

```bash
# Export all articles
curl -O http://localhost:8000/api/v1/export/articles/csv

# Export filtered articles
curl -O "http://localhost:8000/api/v1/export/articles/csv?category=ai_ml"

# Export complete daily report
curl http://localhost:8000/api/v1/export/complete/2024-12-26
```

## Enhanced Features

### AI-Powered Analysis

The system uses LLMs for:
- **Sentiment Analysis**: Positive/negative/neutral classification
- **Entity Recognition**: Companies, people, products, locations
- **Executive Summaries**: AI-generated key insights and recommendations
- **Keyword Extraction**: Important topics and terms

### Caching

Redis caching provides:
- API response caching (5-minute default)
- Rate limiting (100 requests/minute per client)
- Session storage
- Background task broker

## Troubleshooting

### Common Issues

1. **Ollama models not downloading**
   ```bash
   # Check Ollama logs
   docker-compose logs ollama
   
   # Manual model pull
   docker-compose exec ollama ollama pull qwen3:4b
   ```

2. **Database connection errors**
   ```bash
   # Check PostgreSQL health
   docker-compose exec postgres pg_isready -U postgres
   
   # Restart database
   docker-compose restart postgres
   ```

3. **Celery tasks not processing**
   ```bash
   # Check worker logs
   docker-compose logs celery-worker
   
   # Restart workers
   docker-compose restart celery-worker
   ```

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f celery-worker

# Filter by time
docker-compose logs --since=1h api
```

## Production Best Practices

### Security

1. Change default passwords in `.env`
2. Use strong secrets for database and Redis
3. Enable HTTPS in production (use reverse proxy)
4. Restrict access to Flower UI
5. Regular security updates for base images

### Performance

1. Adjust Celery worker count based on CPU cores
2. Monitor database connection pool usage
3. Scale Redis for larger deployments
4. Use connection pooling for database
5. Optimize Ollama model selection

### Backup

```bash
# Backup PostgreSQL data
docker-compose exec postgres pg_dump -U postgres news_aggregator > backup.sql

# Backup Redis data
docker-compose exec redis redis-cli save
docker cp news-aggregator-redis:/data/dump.rdb ./
```

### Updates

```bash
# Pull latest images
docker-compose pull

# Restart with new images
docker-compose up -d

# Run database migrations if needed
docker-compose exec api python db/init_db.py
```

## Support

For issues and questions:
1. Check logs: `docker-compose logs`
2. Verify health endpoints
3. Check system resources
4. Review API documentation at `/docs`

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       Load Balancer                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼──────┐ ┌─────▼──────┐ ┌────▼──────┐
│     API      │ │   Export   │ │  Metrics  │
│  (FastAPI)   │ │   Service  │ │ (Prom)    │
└───────┬──────┘ └─────┬──────┘ └────┬──────┘
        │              │             │
┌───────▼──────────────▼─────────────▼─────────┐
│           PostgreSQL Database                │
└──────────────────────────────────────────────┘
        │
┌───────▼──────────────┐
│        Redis         │
│ (Cache + Broker)     │
└──────────┬───────────┘
           │
    ┌──────▼────────┐
    │ Celery Worker │
    │  (4 workers)  │
    └──────┬────────┘
           │
    ┌──────▼────────┐
    │   Ollama      │
    │   (LLM)       │
    └───────────────┘
```

## License

MIT License - see LICENSE file for details

# News Aggregator - Monitoring Setup Guide

## 📊 Monitoring Stack

The News Aggregator system includes comprehensive monitoring with Prometheus metrics and Grafana dashboards.

### Components

- **Prometheus**: Metrics collection (port 9090)
- **Grafana**: Visualization and dashboards (port 3000)
- **Custom Metrics**: Application-specific metrics exposed via `/api/v1/metrics`

## 🚀 Quick Start

### 1. Start Monitoring Services

```bash
# Start Prometheus (if not included in docker-compose)
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus:latest

# Start Grafana
docker run -d \
  --name grafana \
  -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=admin123 \
  grafana/grafana:latest
```

### 2. Configure Prometheus

Create `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'news-aggregator'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/api/v1/metrics'
    scrape_interval: 30s
```

### 3. Import Grafana Dashboard

1. Access Grafana at http://localhost:3000
2. Login with admin/admin123
3. Go to Configuration → Data Sources → Add data source
4. Select Prometheus
5. Set URL: http://prometheus:9090
6. Click Save & Test
7. Go to Create → Import
8. Upload `monitoring/grafana_dashboard.json`
9. Select the Prometheus data source
10. Click Import

## 📈 Available Metrics

### HTTP Metrics
- `http_requests_total` - Total HTTP requests by method, endpoint, status
- `http_request_duration_seconds` - Request latency histogram

### Database Metrics
- `db_queries_total` - Total database queries by operation, table
- `db_query_duration_seconds` - Query latency histogram

### Cache Metrics
- `cache_operations_total` - Cache operations by type, result
- `cache_hit_rate` - Cache hit rate gauge

### Pipeline Metrics
- `pipeline_runs_total` - Total pipeline runs by status
- `pipeline_articles_processed` - Articles processed per run

### LLM Metrics
- `llm_requests_total` - LLM requests by model, operation
- `llm_request_duration_seconds` - LLM request latency

### System Metrics
- `active_connections` - Active connections gauge
- `total_articles_stored` - Total articles in database
- `total_trends_identified` - Total trends identified

## 📊 Dashboard Panels

The Grafana dashboard includes:

1. **API Request Rate** - Requests per second by endpoint
2. **API Response Time** - 95th and 50th percentile latencies
3. **HTTP Status Codes** - Status code distribution
4. **Cache Hit Rate** - Cache effectiveness gauge
5. **Database Query Rate** - Queries per second
6. **Pipeline Execution Status** - Success/failure counts
7. **Articles Processed** - Processing rate and totals
8. **LLM Request Rate** - LLM API usage
9. **System Resources** - Connection counts, article/trend totals
10. **Error Rate** - HTTP error percentage

## 🎯 Key Metrics to Monitor

### Performance Indicators
- API response time < 200ms (p95)
- Cache hit rate > 80%
- Database query time < 100ms

### Health Indicators
- Pipeline success rate > 95%
- Error rate < 1%
- Active connections within normal range

### Business Metrics
- Articles processed per hour/day
- Trends identified per time period
- LLM API usage and costs

## 🚨 Alerting Rules

Example Prometheus alerting rules:

```yaml
groups:
  - name: news-aggregator
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status_code=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          
      - alert: SlowAPIResponse
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "API response time is high"
          
      - alert: PipelineFailures
        expr: rate(pipeline_runs_total{status='failed'}[15m]) > 0.1
        for: 15m
        labels:
          severity: critical
        annotations:
          summary: "Pipeline failure rate is high"
```

## 🔧 Custom Metrics

To add custom metrics to your application:

```python
from utils.metrics import metrics_collector

# Record a custom event
metrics_collector.record_custom_metric('my_metric', value=1.0, labels={'type': 'custom'})

# Update a gauge
metrics_collector.update_gauge(total_articles_stored, 1000)
```

## 📱 Accessing Metrics

### Prometheus UI
- URL: http://localhost:9090
- Query metrics directly
- View targets and scraping status

### Grafana UI
- URL: http://localhost:3000
- Dashboard: News Aggregator Monitoring
- Username: admin
- Password: admin123 (change this!)

### API Metrics Endpoint
```bash
# Get raw Prometheus metrics
curl http://localhost:8000/api/v1/metrics
```

## 🐳 Docker Compose Integration

The monitoring stack is included in docker-compose.yml:

```yaml
# Prometheus and Grafana services are already configured
# Just run:
docker-compose up -d prometheus grafana
```

## 🔍 Troubleshooting

### Metrics not appearing
1. Check Prometheus targets: http://localhost:9090/targets
2. Verify API is running: curl http://localhost:8000/api/v1/health
3. Check metrics endpoint: curl http://localhost:8000/api/v1/metrics

### Dashboard not showing data
1. Verify Prometheus data source in Grafana
2. Check time range in dashboard
3. Ensure metrics are being scraped

### High error rates
1. Check logs: docker-compose logs api
2. Verify database connectivity
3. Check Redis connectivity

## 📚 Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [FastAPI Metrics Guide](https://fastapi.tiangolo.com/advanced/middleware/)
- [Production Setup Guide](./PRODUCTION_SETUP.md)

## 🎉 Monitoring Complete

Your News Aggregator system now has comprehensive monitoring with:
- ✅ Real-time metrics collection
- ✅ Grafana visualization dashboard
- ✅ Performance tracking
- ✅ Error alerting capabilities
- ✅ Business metrics monitoring

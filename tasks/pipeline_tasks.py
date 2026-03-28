"""
Pipeline execution tasks
"""

from celery import shared_task
from loguru import logger
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import NewsAggregatorPipeline, PipelineConfig
from db.database import db_manager
from db.models import Article


@shared_task(bind=True, max_retries=3)
def run_pipeline(self, sources: list = None, skip_collect: bool = False):
    """Run the news aggregation pipeline"""
    try:
        logger.info("Starting news aggregation pipeline...")

        # Create pipeline config
        config = PipelineConfig()

        # Configure sources if specified
        if sources:
            config.enable_hackernews = "hn" in sources
            config.enable_rss = "rss" in sources
            config.enable_sina = "sina" in sources

        # Create pipeline
        pipeline = NewsAggregatorPipeline(config)

        # Run pipeline
        result = pipeline.run(skip_collect=skip_collect)

        if result.get("success"):
            logger.success(
                f"Pipeline completed successfully. Articles: {result['stages']['collect']['count']}"
            )

            # Store results in database
            _store_pipeline_results(result, pipeline.raw_items)

            return {"status": "success", "result": result, "task_id": self.request.id}
        else:
            logger.error(f"Pipeline failed: {result.get('error')}")
            raise Exception(f"Pipeline execution failed: {result.get('error')}")

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        # Retry with exponential backoff
        countdown = 2**self.request.retries
        raise self.retry(exc=e, countdown=countdown)


def _store_pipeline_results(result: dict, articles: list):
    """Store pipeline results in database"""
    try:
        with db_manager.get_session() as session:
            stored_count = 0

            for article_data in articles:
                # Check if article already exists
                existing = (
                    session.query(Article).filter_by(id=article_data["id"]).first()
                )
                if existing:
                    continue

                # Create article record
                article = Article(
                    id=article_data["id"],
                    title=article_data.get("title", "")[:1000],
                    content=article_data.get("content", ""),
                    url=article_data.get("url"),
                    source=article_data.get("source", "Unknown"),
                    author=article_data.get("author"),
                    category=article_data.get("category", "other"),
                    importance=article_data.get("importance", 3),
                    score=article_data.get("score", 0),
                    comments_count=article_data.get("comments_count", 0),
                    raw_data=article_data,
                    is_processed=False,
                )

                session.add(article)
                stored_count += 1

            logger.info(f"Stored {stored_count} new articles in database")

    except Exception as e:
        logger.error(f"Error storing pipeline results: {e}")


@shared_task(bind=True, max_retries=3)
def collect_data(self, sources: list = None):
    """Task to collect data only"""
    try:
        logger.info("Starting data collection...")

        config = PipelineConfig()

        if sources:
            config.enable_hackernews = "hn" in sources
            config.enable_rss = "rss" in sources
            config.enable_sina = "sina" in sources

        pipeline = NewsAggregatorPipeline(config)
        articles = pipeline.collect_data()

        logger.success(f"Data collection completed. Collected {len(articles)} articles")

        return {
            "status": "success",
            "articles_count": len(articles),
            "task_id": self.request.id,
        }

    except Exception as e:
        logger.error(f"Data collection error: {e}")
        countdown = 2**self.request.retries
        raise self.retry(exc=e, countdown=countdown)


@shared_task(bind=True, max_retries=3)
def analyze_data(self, article_ids: list = None):
    """Task to analyze collected data"""
    try:
        logger.info("Starting data analysis...")

        config = PipelineConfig()
        pipeline = NewsAggregatorPipeline(config)

        # Load articles from database or cache
        if article_ids:
            with db_manager.get_session() as session:
                articles = (
                    session.query(Article).filter(Article.id.in_(article_ids)).all()
                )
                articles_data = [article.raw_data for article in articles]
        else:
            # Get recent unprocessed articles
            with db_manager.get_session() as session:
                articles = (
                    session.query(Article)
                    .filter(Article.is_processed == False)
                    .limit(100)
                    .all()
                )
                articles_data = [article.raw_data for article in articles]

        if not articles_data:
            logger.info("No articles to analyze")
            return {"status": "success", "message": "No articles to analyze"}

        # Run analysis stages
        pipeline.raw_items = articles_data
        pipeline.extract_content()
        pipeline.classify_items()
        pipeline.extract_info()

        logger.success(f"Data analysis completed for {len(articles_data)} articles")

        return {
            "status": "success",
            "articles_analyzed": len(articles_data),
            "task_id": self.request.id,
        }

    except Exception as e:
        logger.error(f"Data analysis error: {e}")
        countdown = 2**self.request.retries
        raise self.retry(exc=e, countdown=countdown)


@shared_task(bind=True, max_retries=3)
def generate_reports_task(self, report_date: str = None, formats: list = None):
    """Task to generate reports"""
    try:
        logger.info("Starting report generation...")

        config = PipelineConfig()
        pipeline = NewsAggregatorPipeline(config)

        # Load articles for report
        with db_manager.get_session() as session:
            if report_date:
                from datetime import datetime

                date_obj = datetime.fromisoformat(report_date)
                articles = (
                    session.query(Article)
                    .filter(
                        Article.publish_time >= date_obj,
                        Article.publish_time
                        < date_obj.replace(hour=23, minute=59, second=59),
                    )
                    .all()
                )
            else:
                # Get today's articles
                from datetime import datetime, timedelta

                today = datetime.now().date()
                articles = (
                    session.query(Article)
                    .filter(
                        Article.publish_time
                        >= datetime.combine(today, datetime.min.time())
                    )
                    .all()
                )

            articles_data = [article.raw_data for article in articles]

        if not articles_data:
            logger.info("No articles for report generation")
            return {"status": "success", "message": "No articles for report generation"}

        # Run trend analysis and report generation
        pipeline.raw_items = articles_data
        pipeline.analyze_trends()

        if formats:
            config.report_formats = formats

        reports = pipeline.generate_reports()

        logger.success(
            f"Report generation completed. Generated: {list(reports.keys())}"
        )

        return {
            "status": "success",
            "reports_generated": list(reports.keys()),
            "report_files": reports,
            "task_id": self.request.id,
        }

    except Exception as e:
        logger.error(f"Report generation error: {e}")
        countdown = 2**self.request.retries
        raise self.retry(exc=e, countdown=countdown)

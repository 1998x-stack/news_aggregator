"""
Analysis tasks for enhanced features
"""

from celery import shared_task
from loguru import logger
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from db.database import db_manager
from db.models import Article
from utils.ollama_client import OllamaClient
from utils.cache_manager import cache_manager


@shared_task(bind=True, max_retries=3)
def analyze_sentiment(self, article_ids: List[str] = None):
    """Analyze sentiment for articles using LLM"""
    try:
        logger.info("Starting sentiment analysis...")

        ollama_client = OllamaClient()

        with db_manager.get_session() as session:
            # Get articles to analyze
            if article_ids:
                articles = (
                    session.query(Article).filter(Article.id.in_(article_ids)).all()
                )
            else:
                # Get articles without sentiment analysis
                articles = (
                    session.query(Article)
                    .filter(Article.sentiment.is_(None))
                    .limit(50)
                    .all()
                )

            if not articles:
                logger.info("No articles to analyze")
                return {"status": "success", "message": "No articles to analyze"}

            analyzed_count = 0
            for article in articles:
                try:
                    # Prepare text for analysis
                    text = f"{article.title}\n\n{article.content[:2000]}"

                    # Create sentiment analysis prompt
                    prompt = f"""
                    Analyze the sentiment of the following news article.
                    Respond with a JSON object containing:
                    - "sentiment": "positive", "negative", or "neutral"
                    - "score": a float between -1.0 (very negative) and 1.0 (very positive)
                    - "reason": brief explanation
                    
                    Article:
                    {text}
                    
                    Response (JSON only):
                    """

                    # Get response from LLM
                    response = ollama_client.generate(
                        prompt=prompt, model="qwen3:4b", format="json"
                    )

                    # Parse response
                    import json

                    sentiment_data = json.loads(response["response"])

                    article.sentiment = sentiment_data.get("sentiment")
                    article.sentiment_score = sentiment_data.get("score")

                    analyzed_count += 1

                    logger.debug(
                        f"Analyzed sentiment for article {article.id}: {article.sentiment}"
                    )

                except Exception as e:
                    logger.error(
                        f"Error analyzing sentiment for article {article.id}: {e}"
                    )
                    continue

            logger.success(
                f"Sentiment analysis completed for {analyzed_count} articles"
            )

            return {
                "status": "success",
                "articles_analyzed": analyzed_count,
                "task_id": self.request.id,
            }

    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        countdown = 2**self.request.retries
        raise self.retry(exc=e, countdown=countdown)


@shared_task(bind=True, max_retries=3)
def extract_entities(self, article_ids: List[str] = None):
    """Extract entities (companies, people, products) from articles"""
    try:
        logger.info("Starting entity extraction...")

        ollama_client = OllamaClient()

        with db_manager.get_session() as session:
            # Get articles to analyze
            if article_ids:
                articles = (
                    session.query(Article).filter(Article.id.in_(article_ids)).all()
                )
            else:
                # Get articles without entity extraction
                articles = (
                    session.query(Article)
                    .filter(Article.entities.is_(None))
                    .limit(30)
                    .all()
                )

            if not articles:
                logger.info("No articles to analyze")
                return {"status": "success", "message": "No articles to analyze"}

            extracted_count = 0
            for article in articles:
                try:
                    # Prepare text for analysis
                    text = f"{article.title}\n\n{article.content[:1500]}"

                    # Create entity extraction prompt
                    prompt = f"""
                    Extract entities from the following news article.
                    Identify companies, people, products, and locations.
                    Respond with a JSON object containing:
                    - "entities": array of entity objects
                    - Each entity should have: "name", "type" (company/person/product/location), "relevance_score" (0-1)
                    
                    Article:
                    {text}
                    
                    Response (JSON only):
                    """

                    # Get response from LLM
                    response = ollama_client.generate(
                        prompt=prompt, model="qwen3:4b", format="json"
                    )

                    # Parse response
                    import json

                    entities_data = json.loads(response["response"])

                    article.entities = entities_data.get("entities", [])

                    extracted_count += 1

                    logger.debug(
                        f"Extracted entities for article {article.id}: {len(article.entities)} entities"
                    )

                except Exception as e:
                    logger.error(
                        f"Error extracting entities for article {article.id}: {e}"
                    )
                    continue

            logger.success(
                f"Entity extraction completed for {extracted_count} articles"
            )

            return {
                "status": "success",
                "articles_processed": extracted_count,
                "task_id": self.request.id,
            }

    except Exception as e:
        logger.error(f"Entity extraction error: {e}")
        countdown = 2**self.request.retries
        raise self.retry(exc=e, countdown=countdown)


@shared_task(bind=True, max_retries=3)
def extract_keywords(self, article_ids: List[str] = None):
    """Extract keywords from articles"""
    try:
        logger.info("Starting keyword extraction...")

        ollama_client = OllamaClient()

        with db_manager.get_session() as session:
            # Get articles to analyze
            if article_ids:
                articles = (
                    session.query(Article).filter(Article.id.in_(article_ids)).all()
                )
            else:
                # Get articles without keywords
                articles = (
                    session.query(Article)
                    .filter(Article.keywords.is_(None))
                    .limit(50)
                    .all()
                )

            if not articles:
                logger.info("No articles to analyze")
                return {"status": "success", "message": "No articles to analyze"}

            extracted_count = 0
            for article in articles:
                try:
                    # Prepare text for analysis
                    text = f"{article.title}\n\n{article.content[:1000]}"

                    # Create keyword extraction prompt
                    prompt = f"""
                    Extract key topics and keywords from the following news article.
                    Respond with a JSON object containing:
                    - "keywords": array of keyword objects
                    - Each keyword should have: "word", "relevance_score" (0-1)
                    - Include 5-10 most important keywords
                    
                    Article:
                    {text}
                    
                    Response (JSON only):
                    """

                    # Get response from LLM
                    response = ollama_client.generate(
                        prompt=prompt, model="qwen2.5:0.5b", format="json"
                    )

                    # Parse response
                    import json

                    keywords_data = json.loads(response["response"])

                    article.keywords = keywords_data.get("keywords", [])

                    extracted_count += 1

                    logger.debug(
                        f"Extracted keywords for article {article.id}: {len(article.keywords)} keywords"
                    )

                except Exception as e:
                    logger.error(
                        f"Error extracting keywords for article {article.id}: {e}"
                    )
                    continue

            logger.success(
                f"Keyword extraction completed for {extracted_count} articles"
            )

            return {
                "status": "success",
                "articles_processed": extracted_count,
                "task_id": self.request.id,
            }

    except Exception as e:
        logger.error(f"Keyword extraction error: {e}")
        countdown = 2**self.request.retries
        raise self.retry(exc=e, countdown=countdown)


@shared_task(bind=True, max_retries=3)
def batch_analyze_articles(self, batch_size: int = 50):
    """Batch analyze articles for all enhancements"""
    try:
        logger.info(f"Starting batch analysis for {batch_size} articles...")

        # Get articles that need analysis
        with db_manager.get_session() as session:
            articles = (
                session.query(Article)
                .filter(
                    or_(
                        Article.sentiment.is_(None),
                        Article.entities.is_(None),
                        Article.keywords.is_(None),
                    )
                )
                .limit(batch_size)
                .all()
            )

            if not articles:
                logger.info("No articles need analysis")
                return {"status": "success", "message": "No articles need analysis"}

            article_ids = [article.id for article in articles]

        # Trigger analysis tasks
        from celery import group

        job = group(
            [
                analyze_sentiment.si(article_ids),
                extract_entities.si(article_ids),
                extract_keywords.si(article_ids),
            ]
        )

        result = job.apply_async()

        logger.success(f"Batch analysis triggered for {len(article_ids)} articles")

        return {
            "status": "success",
            "articles_queued": len(article_ids),
            "task_id": self.request.id,
            "subtask_ids": [task.id for task in result.children],
        }

    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        countdown = 2**self.request.retries
        raise self.retry(exc=e, countdown=countdown)

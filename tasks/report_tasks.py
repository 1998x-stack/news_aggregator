"""
Report generation and delivery tasks
"""
from celery import shared_task
from loguru import logger
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from db.database import db_manager
from db.models import Report, Article
from analyzers.report_generator import ReportGenerator, ReportConfig
from utils.cache_manager import cache_manager


@shared_task(bind=True, max_retries=3)
def generate_daily_report(self, report_date: str = None, formats: List[str] = None):
    """Generate daily report"""
    try:
        logger.info(f"Generating daily report for {report_date or 'today'}...")
        
        # Parse report date
        if report_date:
            report_date_obj = datetime.fromisoformat(report_date)
        else:
            report_date_obj = datetime.now()
        
        # Get articles for the report date
        with db_manager.get_session() as session:
            start_of_day = report_date_obj.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = report_date_obj.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            articles = session.query(Article).filter(
                Article.publish_time >= start_of_day,
                Article.publish_time <= end_of_day
            ).all()
            
            if not articles:
                logger.info(f"No articles found for {report_date_obj.date()}")
                return {"status": "success", "message": "No articles for report generation"}
            
            articles_data = [article.to_dict() for article in articles]
        
        # Create report generator
        config = ReportConfig(
            title=f"Daily News Report - {report_date_obj.strftime('%Y-%m-%d')}"
        )
        generator = ReportGenerator(config)
        
        # Generate trend report data
        from analyzers.trend_analyzer import TrendAnalyzer, create_trend_analyzer
        trend_analyzer = create_trend_analyzer()
        trend_report = trend_analyzer.analyze(articles_data)
        
        # Generate reports
        if formats is None:
            formats = ["markdown", "json"]
        
        report_files = {}
        for fmt in formats:
            try:
                filepath = generator.generate_daily_report(
                    trend_report,
                    articles_data,
                    report_date_obj.strftime("%Y-%m-%d"),
                    fmt
                )
                report_files[fmt] = filepath
                logger.info(f"Generated {fmt} report: {filepath}")
            except Exception as e:
                logger.error(f"Error generating {fmt} report: {e}")
                continue
        
        # Store report in database
        with db_manager.get_session() as session:
            report = Report(
                report_type="daily",
                report_date=report_date_obj,
                title=f"Daily News Report - {report_date_obj.strftime('%Y-%m-%d')}",
                content={
                    "trend_analysis": trend_report.to_dict(),
                    "articles": articles_data
                },
                format=",".join(formats),
                total_articles=len(articles),
                hot_topics_count=len(trend_report.hot_topics),
                categories_count=len(trend_report.category_distribution),
                sources_count=len(trend_report.source_distribution),
                is_generated=True,
                generated_at=datetime.now()
            )
            session.add(report)
        
        logger.success(f"Daily report generated successfully. Formats: {list(report_files.keys())}")
        
        return {
            "status": "success",
            "report_date": report_date_obj.isoformat(),
            "articles_count": len(articles),
            "report_files": report_files,
            "task_id": self.request.id
        }
    
    except Exception as e:
        logger.error(f"Daily report generation error: {e}")
        countdown = 2 ** self.request.retries
        raise self.retry(exc=e, countdown=countdown)


@shared_task(bind=True, max_retries=3)
def generate_category_report(self, category: str, report_date: str = None):
    """Generate category-specific report"""
    try:
        logger.info(f"Generating category report for {category}...")
        
        # Parse report date
        if report_date:
            report_date_obj = datetime.fromisoformat(report_date)
        else:
            report_date_obj = datetime.now()
        
        # Get articles for the category
        with db_manager.get_session() as session:
            articles = session.query(Article).filter(
                Article.category == category,
                Article.publish_time >= report_date_obj.replace(hour=0, minute=0, second=0),
                Article.publish_time <= report_date_obj.replace(hour=23, minute=59, second=59)
            ).all()
            
            if not articles:
                logger.info(f"No articles found for category {category}")
                return {"status": "success", "message": f"No articles for category {category}"}
            
            articles_data = [article.to_dict() for article in articles]
        
        # Create report generator
        config = ReportConfig(
            title=f"{category.upper()} Category Report - {report_date_obj.strftime('%Y-%m-%d')}"
        )
        generator = ReportGenerator(config)
        
        # Generate category report
        filepath = generator.generate_category_report(
            articles_data,
            category,
            report_date_obj.strftime("%Y-%m-%d")
        )
        
        # Store report in database
        with db_manager.get_session() as session:
            report = Report(
                report_type="category",
                report_date=report_date_obj,
                title=f"{category.upper()} Category Report - {report_date_obj.strftime('%Y-%m-%d')}",
                content={
                    "category": category,
                    "articles": articles_data
                },
                format="markdown",
                file_path=filepath,
                total_articles=len(articles),
                is_generated=True,
                generated_at=datetime.now()
            )
            session.add(report)
        
        logger.success(f"Category report generated for {category}")
        
        return {
            "status": "success",
            "category": category,
            "articles_count": len(articles),
            "report_file": filepath,
            "task_id": self.request.id
        }
    
    except Exception as e:
        logger.error(f"Category report generation error: {e}")
        countdown = 2 ** self.request.retries
        raise self.retry(exc=e, countdown=countdown)


@shared_task(bind=True, max_retries=3)
def deliver_report(self, report_id: int, delivery_method: str = "email", recipients: List[str] = None):
    """Deliver report via specified method"""
    try:
        logger.info(f"Delivering report {report_id} via {delivery_method}...")
        
        # Get report from database
        with db_manager.get_session() as session:
            report = session.query(Report).filter(Report.id == report_id).first()
            
            if not report:
                raise Exception(f"Report {report_id} not found")
            
            if not report.is_generated:
                raise Exception(f"Report {report_id} is not generated yet")
        
        # Deliver based on method
        if delivery_method == "email":
            success = _deliver_via_email(report, recipients)
        elif delivery_method == "slack":
            success = _deliver_via_slack(report, recipients)
        else:
            raise Exception(f"Unsupported delivery method: {delivery_method}")
        
        if success:
                        with db_manager.get_session() as session:
                report = session.query(Report).filter(Report.id == report_id).first()
                report.is_delivered = True
            
            logger.success(f"Report {report_id} delivered successfully via {delivery_method}")
        
        return {
            "status": "success" if success else "failed",
            "report_id": report_id,
            "delivery_method": delivery_method,
            "task_id": self.request.id
        }
    
    except Exception as e:
        logger.error(f"Report delivery error: {e}")
        countdown = 2 ** self.request.retries
        raise self.retry(exc=e, countdown=countdown)


def _deliver_via_email(report: Report, recipients: List[str] = None) -> bool:
    """Deliver report via email"""
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        from config.database_config import load_database_config
        
        config = load_database_config()
        
        # Email configuration from environment
        smtp_host = os.getenv("SMTP_HOST")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        smtp_from = os.getenv("SMTP_FROM")
        
        if not all([smtp_host, smtp_user, smtp_password, smtp_from]):
            logger.error("Email configuration not complete")
            return False
        
        if not recipients:
            logger.error("No recipients specified")
            return False
        
        # Create email
        msg = MIMEMultipart()
        msg['From'] = smtp_from
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = f"[News Aggregator] {report.title}"
        
        # Email body
        body = f"""
        Daily News Report
        
        Report Date: {report.report_date.strftime('%Y-%m-%d')}
        Total Articles: {report.total_articles}
        Hot Topics: {report.hot_topics_count}
        
        This report was automatically generated by the News Aggregator system.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email sent to {recipients}")
        return True
    
    except Exception as e:
        logger.error(f"Email delivery error: {e}")
        return False


def _deliver_via_slack(report: Report, channels: List[str] = None) -> bool:
    """Deliver report via Slack"""
    try:
        import requests
        
        slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        if not slack_webhook:
            logger.error("Slack webhook not configured")
            return False
        
        # Prepare message
        message = {
            "text": f"📰 Daily News Report: {report.title}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"📰 {report.title}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Date:*\n{report.report_date.strftime('%Y-%m-%d')}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Articles:*\n{report.total_article
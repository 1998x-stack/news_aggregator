#!/usr/bin/env python3
"""
Data migration script - Migrate JSON data to PostgreSQL
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from db.database import db_manager
from db.models import Article, Trend, Report
from loguru import logger


class DataMigrator:
    """数据迁移器"""

    def __init__(self):
        self.cache_dir = PROJECT_ROOT / "cache"
        self.outputs_dir = PROJECT_ROOT / "outputs"

    def migrate_articles(self) -> int:
        """迁移文章数据"""
        logger.info("Starting articles migration...")

        migrated_count = 0

        # 查找所有缓存文件
        cache_files = list(self.cache_dir.glob("*_raw_items.json"))
        logger.info(f"Found {len(cache_files)} cache files")

        for cache_file in cache_files:
            logger.info(f"Processing {cache_file.name}")

            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    items = json.load(f)

                with db_manager.get_session() as session:
                    for item in items:
                        article = self._create_article(item)
                        if article:
                            session.add(article)
                            migrated_count += 1

                    # 批量提交
                    logger.info(
                        f"Migrated {len(items)} articles from {cache_file.name}"
                    )

            except Exception as e:
                logger.error(f"Error processing {cache_file}: {e}")
                continue

        logger.success(
            f"Articles migration completed: {migrated_count} articles migrated"
        )
        return migrated_count

    def _create_article(self, item: Dict[str, Any]) -> Article:
        """从JSON项创建Article对象"""
        try:
            article_id = item.get("id")
            if not article_id:
                return None

            # 检查是否已存在
            with db_manager.get_session() as session:
                existing = session.query(Article).filter_by(id=article_id).first()
                if existing:
                    return None

            article = Article(
                id=article_id,
                title=item.get("title", "")[:1000],
                content=item.get("content", ""),
                summary=item.get("summary"),
                url=item.get("url"),
                source=item.get("source", "Unknown"),
                author=item.get("author"),
                category=item.get("category", "other"),
                importance=item.get("importance", 3),
                classification_confidence=item.get("classification_confidence", 0.0),
                classification_reason=item.get("classification_reason"),
                extracted_what=item.get("extracted_what"),
                extracted_who=item.get("extracted_who"),
                extracted_when=item.get("extracted_when"),
                extracted_where=item.get("extracted_where"),
                sentiment=item.get("sentiment"),
                sentiment_score=item.get("sentiment_score"),
                entities=item.get("entities"),
                keywords=item.get("keywords"),
                score=item.get("score", 0),
                comments_count=item.get("comments_count", 0),
                is_processed=item.get("is_processed", False),
                is_classified=item.get("is_classified", False),
                is_extracted=item.get("is_extracted", False),
                raw_data=item,
                metadata=item.get("metadata", {}),
            )

            # 处理时间字段
            publish_time = item.get("publish_time")
            if publish_time:
                try:
                    if isinstance(publish_time, str):
                        article.publish_time = datetime.fromisoformat(
                            publish_time.replace("Z", "+00:00")
                        )
                except:
                    pass

            collect_time = item.get("collect_time")
            if collect_time:
                try:
                    if isinstance(collect_time, str):
                        article.collect_time = datetime.fromisoformat(
                            collect_time.replace("Z", "+00:00")
                        )
                except:
                    pass

            return article

        except Exception as e:
            logger.error(f"Error creating article: {e}")
            return None

    def migrate_reports(self) -> int:
        """迁移报告数据"""
        logger.info("Starting reports migration...")

        migrated_count = 0

        # 查找所有报告文件
        report_files = list(self.outputs_dir.glob("*_daily_report.json"))
        logger.info(f"Found {len(report_files)} report files")

        for report_file in report_files:
            logger.info(f"Processing {report_file.name}")

            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    report_data = json.load(f)

                with db_manager.get_session() as session:
                    report = self._create_report(report_data, str(report_file))
                    if report:
                        session.add(report)
                        migrated_count += 1

            except Exception as e:
                logger.error(f"Error processing {report_file}: {e}")
                continue

        logger.success(
            f"Reports migration completed: {migrated_count} reports migrated"
        )
        return migrated_count

    def _create_report(self, report_data: Dict[str, Any], file_path: str) -> Report:
        """从JSON数据创建Report对象"""
        try:
            metadata = report_data.get("metadata", {})
            trend_analysis = report_data.get("trend_analysis", {})

            # 提取报告日期
            report_date_str = metadata.get("date")
            report_date = None
            if report_date_str:
                try:
                    report_date = datetime.fromisoformat(report_date_str)
                except:
                    report_date = datetime.now()

            report = Report(
                report_type="daily",
                report_date=report_date,
                title=metadata.get("title", "新闻聚合分析报告"),
                content=report_data,
                format="json",
                file_path=file_path,
                total_articles=metadata.get("total_articles", 0),
                hot_topics_count=len(trend_analysis.get("hot_topics", [])),
                categories_count=len(trend_analysis.get("category_distribution", {})),
                sources_count=len(trend_analysis.get("source_distribution", {})),
                is_generated=True,
            )

            return report

        except Exception as e:
            logger.error(f"Error creating report: {e}")
            return None

    def verify_migration(self) -> Dict[str, Any]:
        """验证迁移结果"""
        logger.info("Verifying migration...")

        with db_manager.get_session() as session:
            article_count = session.query(Article).count()
            report_count = session.query(Report).count()

            # 检查数据完整性
            sample_articles = session.query(Article).limit(5).all()
            sample_reports = session.query(Report).limit(5).all()

            result = {
                "article_count": article_count,
                "report_count": report_count,
                "sample_articles": [a.to_dict() for a in sample_articles],
                "sample_reports": [r.to_dict() for r in sample_reports],
            }

            logger.info(
                f"Migration verification: {article_count} articles, {report_count} reports"
            )
            return result


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Data migration script")
    parser.add_argument("--articles", action="store_true", help="Migrate articles only")
    parser.add_argument("--reports", action="store_true", help="Migrate reports only")
    parser.add_argument("--verify", action="store_true", help="Verify migration only")

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:7}</level> | {message}",
    )

    migrator = DataMigrator()

    if args.verify:
        result = migrator.verify_migration()
        print("\nMigration verification result:")
        print(f"Articles: {result['article_count']}")
        print(f"Reports: {result['report_count']}")
        return

    # Default: migrate everything
    migrate_all = not (args.articles or args.reports)

    if args.articles or migrate_all:
        migrator.migrate_articles()

    if args.reports or migrate_all:
        migrator.migrate_reports()

    # Verify
    result = migrator.verify_migration()
    print("\nMigration completed!")
    print(f"Total articles: {result['article_count']}")
    print(f"Total reports: {result['report_count']}")


if __name__ == "__main__":
    main()

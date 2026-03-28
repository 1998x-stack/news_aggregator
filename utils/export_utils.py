"""
Data export utilities for CSV and JSON formats
"""

import csv
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from loguru import logger


class DataExporter:
    """Utility class for exporting data to various formats"""

    @staticmethod
    def export_to_csv(
        data: List[Dict[str, Any]],
        filepath: str,
        fieldnames: Optional[List[str]] = None,
    ) -> bool:
        """
        Export data to CSV file

        Args:
            data: List of dictionaries to export
            filepath: Output file path
            fieldnames: Specific fieldnames to use (optional)

        Returns:
            bool: Success status
        """
        try:
            if not data:
                logger.warning("No data to export to CSV")
                return False

            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # Determine fieldnames from first item if not provided
            if fieldnames is None:
                fieldnames = list(data[0].keys())

            with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for row in data:
                    # Filter row to only include fieldnames and handle nested data
                    filtered_row = {}
                    for field in fieldnames:
                        value = row.get(field, "")
                        # Convert complex objects to JSON strings
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value, ensure_ascii=False)
                        filtered_row[field] = value

                    writer.writerow(filtered_row)

            logger.success(f"CSV export completed: {filepath}")
            return True

        except Exception as e:
            logger.error(f"CSV export error: {e}")
            return False

    @staticmethod
    def export_to_json(
        data: List[Dict[str, Any]], filepath: str, pretty: bool = True
    ) -> bool:
        """
        Export data to JSON file

        Args:
            data: List of dictionaries to export
            filepath: Output file path
            pretty: Whether to format JSON with indentation

        Returns:
            bool: Success status
        """
        try:
            if not data:
                logger.warning("No data to export to JSON")
                return False

            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as jsonfile:
                if pretty:
                    json.dump(data, jsonfile, ensure_ascii=False, indent=2, default=str)
                else:
                    json.dump(data, jsonfile, ensure_ascii=False, default=str)

            logger.success(f"JSON export completed: {filepath}")
            return True

        except Exception as e:
            logger.error(f"JSON export error: {e}")
            return False

    @staticmethod
    def export_articles_to_csv(articles: List[Dict[str, Any]], filepath: str) -> bool:
        """Export articles to CSV with standard fields"""
        fieldnames = [
            "id",
            "title",
            "source",
            "category",
            "importance",
            "sentiment",
            "sentiment_score",
            "publish_time",
            "url",
            "author",
            "score",
            "comments_count",
        ]

        return DataExporter.export_to_csv(articles, filepath, fieldnames)

    @staticmethod
    def export_articles_to_json(articles: List[Dict[str, Any]], filepath: str) -> bool:
        """Export articles to JSON"""
        return DataExporter.export_to_json(articles, filepath)

    @staticmethod
    def export_trends_to_csv(trends: List[Dict[str, Any]], filepath: str) -> bool:
        """Export trends to CSV with standard fields"""
        fieldnames = [
            "id",
            "name",
            "category",
            "heat_score",
            "mention_count",
            "trend_type",
            "start_date",
            "end_date",
        ]

        return DataExporter.export_to_csv(trends, filepath, fieldnames)

    @staticmethod
    def export_trends_to_json(trends: List[Dict[str, Any]], filepath: str) -> bool:
        """Export trends to JSON"""
        return DataExporter.export_to_json(trends, filepath)

    @staticmethod
    def export_combined_report(
        articles: List[Dict[str, Any]],
        trends: List[Dict[str, Any]],
        report_date: str,
        output_dir: str,
    ) -> Dict[str, str]:
        """
        Export complete report with both articles and trends

        Returns:
            Dict with filepaths for each exported file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        # Export articles
        articles_csv = output_path / f"{report_date}_articles.csv"
        if DataExporter.export_articles_to_csv(articles, str(articles_csv)):
            results["articles_csv"] = str(articles_csv)

        articles_json = output_path / f"{report_date}_articles.json"
        if DataExporter.export_articles_to_json(articles, str(articles_json)):
            results["articles_json"] = str(articles_json)

        # Export trends
        trends_csv = output_path / f"{report_date}_trends.csv"
        if DataExporter.export_trends_to_csv(trends, str(trends_csv)):
            results["trends_csv"] = str(trends_csv)

        trends_json = output_path / f"{report_date}_trends.json"
        if DataExporter.export_trends_to_json(trends, str(trends_json)):
            results["trends_json"] = str(trends_json)

        # Export combined data
        combined_data = {
            "metadata": {
                "report_date": report_date,
                "generated_at": datetime.now().isoformat(),
                "total_articles": len(articles),
                "total_trends": len(trends),
            },
            "articles": articles,
            "trends": trends,
        }

        combined_json = output_path / f"{report_date}_complete_report.json"
        if DataExporter.export_to_json(
            [combined_data], str(combined_json), pretty=True
        ):
            results["combined_json"] = str(combined_json)

        return results

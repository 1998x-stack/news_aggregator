#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析模块"""

from .classifier import (
    ContentClassifier,
    ClassificationResult,
    create_classifier
)
from .extractor import (
    ContentExtractorLLM,
    ExtractedInfo,
    create_extractor
)
from .trend_analyzer import (
    TrendAnalyzer,
    TrendReport,
    TrendItem,
    IndustryDynamic,
    create_trend_analyzer
)
from .report_generator import (
    ReportGenerator,
    ReportConfig,
    create_report_generator
)

__all__ = [
    # Classifier
    "ContentClassifier",
    "ClassificationResult",
    "create_classifier",
    # Extractor
    "ContentExtractorLLM",
    "ExtractedInfo",
    "create_extractor",
    # Trend Analyzer
    "TrendAnalyzer",
    "TrendReport",
    "TrendItem",
    "IndustryDynamic",
    "create_trend_analyzer",
    # Report Generator
    "ReportGenerator",
    "ReportConfig",
    "create_report_generator"
]

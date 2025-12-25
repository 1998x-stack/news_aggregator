#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""内容抽取模块"""

from .content_extractor import (
    ContentExtractor,
    ExtractedContent,
    SimpleHTMLExtractor
)

__all__ = [
    "ContentExtractor",
    "ExtractedContent",
    "SimpleHTMLExtractor"
]

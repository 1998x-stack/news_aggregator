#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM提示词模块"""

from .llm_prompts import (
    PromptType,
    PromptTemplate,
    PROMPT_TEMPLATES,
    get_prompt,
    get_prompt_template,
    format_prompt,
    format_classification_prompt,
    format_extraction_prompt,
    PromptGenerator,
    # 原始prompt字符串
    CLASSIFICATION_SYSTEM_PROMPT,
    CLASSIFICATION_USER_PROMPT,
    EXTRACTION_5W2H_SYSTEM_PROMPT,
    EXTRACTION_5W2H_USER_PROMPT,
    TREND_ANALYSIS_SYSTEM_PROMPT,
    TREND_ANALYSIS_USER_PROMPT,
)

__all__ = [
    "PromptType",
    "PromptTemplate",
    "PROMPT_TEMPLATES",
    "get_prompt",
    "get_prompt_template",
    "format_prompt",
    "format_classification_prompt",
    "format_extraction_prompt",
    "PromptGenerator",
    "CLASSIFICATION_SYSTEM_PROMPT",
    "CLASSIFICATION_USER_PROMPT",
    "EXTRACTION_5W2H_SYSTEM_PROMPT",
    "EXTRACTION_5W2H_USER_PROMPT",
    "TREND_ANALYSIS_SYSTEM_PROMPT",
    "TREND_ANALYSIS_USER_PROMPT",
]
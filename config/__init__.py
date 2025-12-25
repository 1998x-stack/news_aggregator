#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配置模块"""

from .settings import (
    ContentCategory,
    ImportanceLevel,
    IMPORTANCE_LEVELS,
    CLASSIFICATION_KEYWORDS,
    CATEGORY_KEYWORDS,
    IMPORTANCE_KEYWORDS,
    IMPORTANCE_BOOST_KEYWORDS,
    IMPORTANCE_PENALTY_KEYWORDS,
    IMPORTANCE_RULES,
    LLMSettings,
    DATA_SOURCES,
    OLLAMA_MODELS,
    OllamaModelConfig,
    PathConfig,
    SystemConfig,
    config,
    init_system,
    validate_config
)

__all__ = [
    "ContentCategory",
    "ImportanceLevel",
    "IMPORTANCE_LEVELS",
    "CLASSIFICATION_KEYWORDS",
    "CATEGORY_KEYWORDS",
    "IMPORTANCE_KEYWORDS",
    "IMPORTANCE_BOOST_KEYWORDS",
    "IMPORTANCE_PENALTY_KEYWORDS",
    "IMPORTANCE_RULES",
    "LLMSettings",
    "DATA_SOURCES",
    "OLLAMA_MODELS",
    "OllamaModelConfig",
    "PathConfig",
    "SystemConfig",
    "config",
    "init_system",
    "validate_config"
]
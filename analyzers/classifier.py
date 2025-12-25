#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内容分类器模块

使用qwen2.5:0.5b轻量模型进行:
- 内容分类（14个预设类别）
- 重要性评估（5级评分）
- 批量处理与缓存

依赖:
    - loguru: 日志记录
    - utils.ollama_client: Ollama客户端

作者: News Aggregator System
创建日期: 2025-12-25
"""

import json
import hashlib
import traceback
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from enum import Enum

from loguru import logger

# 导入本地模块
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ollama_client import OllamaClient, OllamaConfig, OllamaModel
from config.settings import (
    ContentCategory, 
    ImportanceLevel, 
    CATEGORY_KEYWORDS,
    IMPORTANCE_BOOST_KEYWORDS,
    IMPORTANCE_PENALTY_KEYWORDS,
    OllamaModelConfig
)
from prompts.llm_prompts import get_prompt, PromptType


@dataclass
class ClassificationResult:
    """分类结果数据类"""
    category: ContentCategory
    importance: ImportanceLevel
    confidence: float  # 置信度 0-1
    reason: str  # 分类理由
    keywords_matched: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    model_used: str = ""
    raw_response: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "category": self.category.value,
            "importance": self.importance.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "keywords_matched": self.keywords_matched,
            "processing_time_ms": self.processing_time_ms,
            "model_used": self.model_used
        }


@dataclass
class ContentItem:
    """待分类内容项"""
    id: str
    title: str
    source: str
    summary: Optional[str] = None
    content: Optional[str] = None
    url: Optional[str] = None
    published_at: Optional[datetime] = None
    
    def to_prompt_text(self) -> str:
        """转换为用于提示词的文本"""
        parts = [f"标题: {self.title}", f"来源: {self.source}"]
        if self.summary:
            parts.append(f"摘要: {self.summary[:500]}")
        elif self.content:
            parts.append(f"内容: {self.content[:500]}")
        return "\n".join(parts)
    
    def get_hash(self) -> str:
        """获取内容哈希（用于缓存）"""
        text = f"{self.title}|{self.source}|{self.summary or ''}"
        return hashlib.md5(text.encode()).hexdigest()[:16]


class RuleBasedClassifier:
    """
    基于规则的快速分类器
    
    用于预分类和LLM结果验证
    """
    
    def __init__(self):
        """初始化规则分类器"""
        self.category_keywords = CATEGORY_KEYWORDS
        self.boost_keywords = IMPORTANCE_BOOST_KEYWORDS
        self.penalty_keywords = IMPORTANCE_PENALTY_KEYWORDS
    
    def classify(self, item: ContentItem) -> Tuple[Optional[ContentCategory], List[str]]:
        """
        基于关键词进行分类
        
        Args:
            item: 待分类内容
            
        Returns:
            Tuple[category, matched_keywords]: 分类结果和匹配的关键词
        """
        text = f"{item.title} {item.summary or ''} {item.content or ''}".lower()
        
        category_scores: Dict[ContentCategory, int] = {}
        matched_keywords: Dict[ContentCategory, List[str]] = {}
        
        for category, keywords in self.category_keywords.items():
            score = 0
            matches = []
            for keyword in keywords:
                if keyword.lower() in text:
                    score += 1
                    matches.append(keyword)
            
            if score > 0:
                category_scores[category] = score
                matched_keywords[category] = matches
        
        if not category_scores:
            return None, []
        
        # 返回得分最高的分类
        best_category = max(category_scores, key=category_scores.get)
        return best_category, matched_keywords.get(best_category, [])
    
    def evaluate_importance(self, item: ContentItem, base_importance: int = 3) -> int:
        """
        评估内容重要性
        
        Args:
            item: 待评估内容
            base_importance: 基础重要性分数
            
        Returns:
            int: 调整后的重要性分数（1-5）
        """
        text = f"{item.title} {item.summary or ''}".lower()
        importance = base_importance
        
        # 提升词加分
        for keyword in self.boost_keywords:
            if keyword.lower() in text:
                importance += 1
        
        # 惩罚词减分
        for keyword in self.penalty_keywords:
            if keyword.lower() in text:
                importance -= 1
        
        # 限制范围
        return max(1, min(5, importance))


class ContentClassifier:
    """
    内容分类器
    
    结合规则引擎和LLM进行智能分类
    
    使用示例:
        >>> classifier = ContentClassifier()
        >>> item = ContentItem(id="1", title="OpenAI发布GPT-5", source="TechCrunch")
        >>> result = classifier.classify(item)
        >>> print(result.category, result.importance)
    """
    
    # 分类模型配置
    DEFAULT_MODEL = OllamaModel.QWEN_0_5B.value
    DEFAULT_TEMPERATURE = 0.1  # 低温度保证一致性
    DEFAULT_MAX_TOKENS = 512
    
    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        use_rules_first: bool = True,
        cache_enabled: bool = True
    ):
        """
        初始化分类器
        
        Args:
            ollama_client: Ollama客户端实例
            use_rules_first: 是否优先使用规则分类
            cache_enabled: 是否启用结果缓存
        """
        self.client = ollama_client or self._create_client()
        self.rule_classifier = RuleBasedClassifier()
        self.use_rules_first = use_rules_first
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, ClassificationResult] = {}
        
        # 加载提示词
        self._system_prompt = self._build_system_prompt()
        
        logger.info(f"ContentClassifier初始化完成，模型: {self.DEFAULT_MODEL}")
    
    def _create_client(self) -> OllamaClient:
        """创建Ollama客户端"""
        config = OllamaConfig(
            timeout=60,
            max_retries=2,
            default_model=self.DEFAULT_MODEL,
            default_temperature=self.DEFAULT_TEMPERATURE,
            default_max_tokens=self.DEFAULT_MAX_TOKENS
        )
        return OllamaClient(config)
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        # 从prompts模块获取分类提示词
        prompt_template = get_prompt_template(PromptType.CLASSIFICATION)
        return prompt_template.system_prompt
        
        # # 构建分类列表
        # categories_text = "\n".join([
        #     f"- {cat.value}: {cat.name}" for cat in ContentCategory
        # ])
        
        # return prompt_template.format(categories=categories_text)
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        解析LLM响应
        
        Args:
            response_text: LLM原始响应文本
            
        Returns:
            Dict: 解析后的结果
        """
        try:
            # 尝试直接解析JSON
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取JSON块
        text = response_text
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start_idx = text.find(start_char)
            end_idx = text.rfind(end_char)
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                try:
                    return json.loads(text[start_idx:end_idx + 1])
                except json.JSONDecodeError:
                    continue
        
        logger.warning(f"无法解析LLM响应: {response_text[:200]}")
        return {}
    
    def _map_category(self, category_str: str) -> ContentCategory:
        """
        映射分类字符串到枚举
        
        Args:
            category_str: 分类字符串
            
        Returns:
            ContentCategory: 分类枚举
        """
        category_str = category_str.lower().strip()
        
        # 直接匹配
        for cat in ContentCategory:
            if cat.value == category_str or cat.name.lower() == category_str:
                return cat
        
        # 模糊匹配
        category_mapping = {
            "ai": ContentCategory.AI_ML,
            "ml": ContentCategory.AI_ML,
            "artificial intelligence": ContentCategory.AI_ML,
            "machine learning": ContentCategory.AI_ML,
            "人工智能": ContentCategory.AI_ML,
            "大模型": ContentCategory.AI_ML,
            "llm": ContentCategory.AI_ML,
            "programming": ContentCategory.PROGRAMMING,
            "coding": ContentCategory.PROGRAMMING,
            "编程": ContentCategory.PROGRAMMING,
            "cloud": ContentCategory.CLOUD_INFRA,
            "infrastructure": ContentCategory.CLOUD_INFRA,
            "security": ContentCategory.SECURITY,
            "cybersecurity": ContentCategory.SECURITY,
            "安全": ContentCategory.SECURITY,
            "data": ContentCategory.DATA_SCIENCE,
            "analytics": ContentCategory.DATA_SCIENCE,
            "blockchain": ContentCategory.BLOCKCHAIN,
            "crypto": ContentCategory.BLOCKCHAIN,
            "hardware": ContentCategory.HARDWARE,
            "chip": ContentCategory.HARDWARE,
            "startup": ContentCategory.STARTUP,
            "funding": ContentCategory.STARTUP,
            "big tech": ContentCategory.BIG_TECH,
            "faang": ContentCategory.BIG_TECH,
            "market": ContentCategory.MARKET,
            "stock": ContentCategory.MARKET,
            "policy": ContentCategory.POLICY,
            "regulation": ContentCategory.POLICY,
            "research": ContentCategory.RESEARCH,
            "paper": ContentCategory.RESEARCH,
            "product": ContentCategory.PRODUCT,
            "launch": ContentCategory.PRODUCT,
        }
        
        for key, cat in category_mapping.items():
            if key in category_str:
                return cat
        
        return ContentCategory.OTHER
    
    def _map_importance(self, importance_value: Any) -> ImportanceLevel:
        """
        映射重要性值到枚举
        
        Args:
            importance_value: 重要性值（可能是int或str）
            
        Returns:
            ImportanceLevel: 重要性枚举
        """
        try:
            level = int(importance_value)
            level = max(1, min(5, level))
            return ImportanceLevel(level)
        except (ValueError, TypeError):
            return ImportanceLevel.MEDIUM
    
    def classify(
        self,
        item: ContentItem,
        use_llm: bool = True
    ) -> ClassificationResult:
        """
        对单个内容进行分类
        
        Args:
            item: 待分类内容
            use_llm: 是否使用LLM（False时仅用规则）
            
        Returns:
            ClassificationResult: 分类结果
        """
        start_time = datetime.now()
        
        # 检查缓存
        cache_key = item.get_hash()
        if self.cache_enabled and cache_key in self._cache:
            logger.debug(f"缓存命中: {cache_key}")
            return self._cache[cache_key]
        
        # 规则预分类
        rule_category, matched_keywords = self.rule_classifier.classify(item)
        rule_importance = self.rule_classifier.evaluate_importance(item)
        
        # 如果规则分类置信度高或不使用LLM
        if (self.use_rules_first and rule_category and 
            len(matched_keywords) >= 3) or not use_llm:
            
            result = ClassificationResult(
                category=rule_category or ContentCategory.OTHER,
                importance=ImportanceLevel(rule_importance),
                confidence=min(0.9, 0.3 + 0.1 * len(matched_keywords)),
                reason=f"规则匹配: {', '.join(matched_keywords[:5])}",
                keywords_matched=matched_keywords,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                model_used="rule_based"
            )
            
            if self.cache_enabled:
                self._cache[cache_key] = result
            
            return result
        
        # LLM分类
        try:
            prompt = f"""请分析以下内容并进行分类：

{item.to_prompt_text()}

请以JSON格式返回结果，包含以下字段：
- category: 分类（从以下选择：{', '.join([c.value for c in ContentCategory])}）
- importance: 重要性（1-5，5最重要）
- confidence: 置信度（0-1）
- reason: 分类理由（简短说明）

仅返回JSON，不要其他内容。"""

            llm_result = self.client.generate(
                prompt=prompt,
                model=self.DEFAULT_MODEL,
                system=self._system_prompt,
                temperature=self.DEFAULT_TEMPERATURE,
                max_tokens=self.DEFAULT_MAX_TOKENS,
                format_json=True
            )
            
            if llm_result.success:
                parsed = self._parse_llm_response(llm_result.text)
                
                if parsed:
                    category = self._map_category(parsed.get("category", "other"))
                    importance = self._map_importance(parsed.get("importance", 3))
                    
                    # 如果规则和LLM结果冲突，取置信度高的
                    llm_confidence = float(parsed.get("confidence", 0.5))
                    
                    if rule_category and len(matched_keywords) >= 2:
                        rule_confidence = 0.3 + 0.15 * len(matched_keywords)
                        if rule_confidence > llm_confidence:
                            category = rule_category
                    
                    # 综合评估重要性
                    llm_importance_val = importance.value
                    final_importance = round((llm_importance_val + rule_importance) / 2)
                    final_importance = max(1, min(5, final_importance))
                    
                    result = ClassificationResult(
                        category=category,
                        importance=ImportanceLevel(final_importance),
                        confidence=llm_confidence,
                        reason=parsed.get("reason", "LLM分类"),
                        keywords_matched=matched_keywords,
                        processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                        model_used=self.DEFAULT_MODEL,
                        raw_response=parsed
                    )
                    
                    if self.cache_enabled:
                        self._cache[cache_key] = result
                    
                    return result
        
        except Exception as e:
            logger.warning(f"LLM分类失败: {e}")
            traceback.print_exc(file=sys.stderr)
        
        # 回退到规则分类
        result = ClassificationResult(
            category=rule_category or ContentCategory.OTHER,
            importance=ImportanceLevel(rule_importance),
            confidence=0.3,
            reason="LLM失败，使用规则分类",
            keywords_matched=matched_keywords,
            processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            model_used="rule_fallback"
        )
        
        if self.cache_enabled:
            self._cache[cache_key] = result
        
        return result
    
    def classify_batch(
        self,
        items: List[ContentItem],
        use_llm: bool = True,
        progress_callback: Optional[callable] = None
    ) -> List[ClassificationResult]:
        """
        批量分类
        
        Args:
            items: 待分类内容列表
            use_llm: 是否使用LLM
            progress_callback: 进度回调函数
            
        Returns:
            List[ClassificationResult]: 分类结果列表
        """
        results = []
        total = len(items)
        
        logger.info(f"开始批量分类，共 {total} 项")
        
        for i, item in enumerate(items):
            result = self.classify(item, use_llm=use_llm)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total, item, result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"分类进度: {i + 1}/{total}")
        
        # 统计分类分布
        category_dist = {}
        importance_dist = {}
        
        for r in results:
            cat = r.category.value
            imp = r.importance.value
            category_dist[cat] = category_dist.get(cat, 0) + 1
            importance_dist[imp] = importance_dist.get(imp, 0) + 1
        
        logger.info(f"分类完成，分布: {category_dist}")
        logger.info(f"重要性分布: {importance_dist}")
        
        return results
    
    def filter_by_category(
        self,
        items: List[ContentItem],
        categories: List[ContentCategory],
        use_llm: bool = True
    ) -> List[Tuple[ContentItem, ClassificationResult]]:
        """
        过滤指定分类的内容
        
        Args:
            items: 待过滤内容列表
            categories: 目标分类列表
            use_llm: 是否使用LLM
            
        Returns:
            List[Tuple]: 匹配的(内容, 分类结果)元组列表
        """
        results = self.classify_batch(items, use_llm=use_llm)
        
        filtered = []
        for item, result in zip(items, results):
            if result.category in categories:
                filtered.append((item, result))
        
        return filtered
    
    def filter_by_importance(
        self,
        items: List[ContentItem],
        min_importance: ImportanceLevel = ImportanceLevel.MEDIUM,
        use_llm: bool = True
    ) -> List[Tuple[ContentItem, ClassificationResult]]:
        """
        过滤高重要性内容
        
        Args:
            items: 待过滤内容列表
            min_importance: 最低重要性级别
            use_llm: 是否使用LLM
            
        Returns:
            List[Tuple]: 匹配的(内容, 分类结果)元组列表
        """
        results = self.classify_batch(items, use_llm=use_llm)
        
        filtered = []
        for item, result in zip(items, results):
            if result.importance.value >= min_importance.value:
                filtered.append((item, result))
        
        return filtered
    
    def get_category_stats(
        self,
        results: List[ClassificationResult]
    ) -> Dict[str, Any]:
        """
        获取分类统计信息
        
        Args:
            results: 分类结果列表
            
        Returns:
            Dict: 统计信息
        """
        if not results:
            return {}
        
        category_counts = {}
        importance_counts = {}
        confidence_sum = 0.0
        processing_time_sum = 0.0
        
        for r in results:
            cat = r.category.value
            imp = r.importance.value
            
            category_counts[cat] = category_counts.get(cat, 0) + 1
            importance_counts[imp] = importance_counts.get(imp, 0) + 1
            confidence_sum += r.confidence
            processing_time_sum += r.processing_time_ms
        
        return {
            "total": len(results),
            "category_distribution": category_counts,
            "importance_distribution": importance_counts,
            "avg_confidence": confidence_sum / len(results),
            "avg_processing_time_ms": processing_time_sum / len(results),
            "high_importance_count": sum(1 for r in results if r.importance.value >= 4),
            "low_confidence_count": sum(1 for r in results if r.confidence < 0.5)
        }
    
    def clear_cache(self):
        """清空分类缓存"""
        self._cache.clear()
        logger.info("分类缓存已清空")
    
    def close(self):
        """关闭分类器"""
        if self.client:
            self.client.close()
        logger.info("ContentClassifier已关闭")


def create_classifier(
    ollama_base_url: str = "http://localhost:11434",
    use_rules_first: bool = True,
    cache_enabled: bool = True
) -> ContentClassifier:
    """
    工厂函数：创建分类器
    
    Args:
        ollama_base_url: Ollama服务地址
        use_rules_first: 是否优先使用规则分类
        cache_enabled: 是否启用缓存
        
    Returns:
        ContentClassifier: 分类器实例
    """
    config = OllamaConfig(
        base_url=ollama_base_url,
        timeout=60,
        max_retries=2,
        default_model=OllamaModel.QWEN_0_5B.value,
        default_temperature=0.1,
        default_max_tokens=512
    )
    client = OllamaClient(config)
    
    return ContentClassifier(
        ollama_client=client,
        use_rules_first=use_rules_first,
        cache_enabled=cache_enabled
    )


if __name__ == "__main__":
    # 测试代码
    logger.add(sys.stderr, level="DEBUG")
    
    # 测试数据
    test_items = [
        ContentItem(
            id="1",
            title="OpenAI发布GPT-5：性能提升显著",
            source="TechCrunch",
            summary="OpenAI今日宣布发布GPT-5，新模型在推理和创造性任务上表现出色"
        ),
        ContentItem(
            id="2",
            title="Python 3.14发布，性能提升20%",
            source="Python.org",
            summary="Python发布新版本，引入新的JIT编译器"
        ),
        ContentItem(
            id="3",
            title="AWS宣布新一代EC2实例",
            source="AWS Blog",
            summary="Amazon推出基于自研芯片的新EC2实例"
        ),
        ContentItem(
            id="4",
            title="比特币突破10万美元",
            source="CoinDesk",
            summary="比特币价格创历史新高"
        ),
        ContentItem(
            id="5",
            title="某创业公司获得A轮融资5000万",
            source="36Kr",
            summary="AI医疗创业公司完成融资"
        )
    ]
    
    # 仅使用规则分类器测试（不需要Ollama服务）
    print("=== 规则分类器测试 ===")
    rule_classifier = RuleBasedClassifier()
    
    for item in test_items:
        category, keywords = rule_classifier.classify(item)
        importance = rule_classifier.evaluate_importance(item)
        print(f"\n标题: {item.title}")
        print(f"分类: {category.value if category else 'other'}")
        print(f"重要性: {importance}")
        print(f"匹配关键词: {keywords}")
    
    # 完整分类器测试（需要Ollama服务）
    print("\n=== 完整分类器测试 ===")
    try:
        classifier = create_classifier()
        
        if not classifier.client.check_health():
            print("Ollama服务不可用，跳过LLM测试")
        else:
            results = classifier.classify_batch(test_items, use_llm=True)
            
            for item, result in zip(test_items, results):
                print(f"\n标题: {item.title}")
                print(f"分类: {result.category.value}")
                print(f"重要性: {result.importance.value}")
                print(f"置信度: {result.confidence:.2f}")
                print(f"理由: {result.reason}")
                print(f"耗时: {result.processing_time_ms:.0f}ms")
            
            # 统计信息
            stats = classifier.get_category_stats(results)
            print(f"\n=== 统计信息 ===")
            print(json.dumps(stats, indent=2, ensure_ascii=False))
            
            classifier.close()
            
    except Exception as e:
        print(f"分类器测试失败: {e}")
        traceback.print_exc(file=sys.stderr)

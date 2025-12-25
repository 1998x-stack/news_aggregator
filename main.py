#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻聚合分析系统 - 主入口

整合数据采集、内容抽取、分析、报告生成的完整流水线

使用方法:
    python main.py                    # 运行完整流水线
    python main.py --collect-only     # 仅采集数据
    python main.py --analyze-only     # 仅分析(使用缓存数据)
    python main.py --sources hn,rss   # 指定数据源

作者: Claude
日期: 2024-12
"""

import sys
import os
import json
import argparse
import traceback
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入项目模块
from config.settings import (
    DATA_SOURCES, ContentCategory, IMPORTANCE_LEVELS,
    LLMSettings, init_system, validate_config
)

# 数据采集
from collectors.hackernews_collector import HackerNewsCollector
from collectors.rss_collector import RSSCollector
from collectors.sina_zhibo_collector import SinaZhiboCollector

# 内容抽取
from extractors.content_extractor import ContentExtractor

# 分析模块
from analyzers.classifier import ContentClassifier, create_classifier
from analyzers.extractor import ContentExtractorLLM, create_extractor
from analyzers.trend_analyzer import TrendAnalyzer, create_trend_analyzer
from analyzers.report_generator import ReportGenerator, ReportConfig, create_report_generator

# 工具
from utils.ollama_client import OllamaClient, check_ollama_available
from utils.file_utils import save_json, load_json, ensure_dir


def get_err_message():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    error_message = repr(
        traceback.format_exception(exc_type, exc_value, exc_traceback)
    )
    return error_message
            
@dataclass
class PipelineConfig:
    """流水线配置"""
    # 数据源开关
    enable_hackernews: bool = True
    enable_rss: bool = True
    enable_sina: bool = True
    
    # 采集参数
    hn_max_stories: int = 30
    rss_max_items: int = 50
    sina_max_items: int = 50
    
    # 内容抽取
    extract_full_content: bool = True
    max_extract_workers: int = 5
    
    # 分析参数
    enable_classification: bool = True
    enable_extraction: bool = True
    enable_trend_analysis: bool = True
    
    # 报告参数
    report_formats: List[str] = field(default_factory=lambda: ["markdown", "json"])
    
    # 缓存
    use_cache: bool = True
    cache_dir: str = "cache"
    
    # 输出
    output_dir: str = "outputs"


class NewsAggregatorPipeline:
    """
    新闻聚合分析流水线
    
    完整流程:
    1. 数据采集 - 从多个源获取新闻
    2. 内容抽取 - 获取文章全文
    3. 分类标注 - 类别和重要性
    4. 信息抽取 - 5W2H结构化
    5. 趋势分析 - 识别热点和趋势
    6. 报告生成 - 输出分析报告
    
    Attributes:
        config: 流水线配置
        date: 运行日期
    """
    
    def __init__(self, config: PipelineConfig = None):
        """
        初始化流水线
        
        Args:
            config: 流水线配置
        """
        self.config = config or PipelineConfig()
        self.date = datetime.now().strftime("%Y-%m-%d")
        
        # 确保目录存在
        ensure_dir(self.config.cache_dir)
        ensure_dir(self.config.output_dir)
        
        # 初始化组件（延迟加载）
        self._collectors = {}
        self._content_extractor = None
        self._classifier = None
        self._info_extractor = None
        self._trend_analyzer = None
        self._report_generator = None
        
        # 运行时数据
        self.raw_items = []  # 原始采集数据
        self.articles = []   # 处理后的文章
        self.trend_report = None  # 趋势分析报告
        
        logger.info("NewsAggregatorPipeline 初始化完成")
    
    # ==================== 组件懒加载 ====================
    
    def _get_hn_collector(self) -> HackerNewsCollector:
        if "hn" not in self._collectors:
            self._collectors["hn"] = HackerNewsCollector()
        return self._collectors["hn"]
    
    def _get_rss_collector(self) -> RSSCollector:
        if "rss" not in self._collectors:
            self._collectors["rss"] = RSSCollector()
        return self._collectors["rss"]
    
    def _get_sina_collector(self) -> SinaZhiboCollector:
        if "sina" not in self._collectors:
            self._collectors["sina"] = SinaZhiboCollector()
        return self._collectors["sina"]
    
    def _get_content_extractor(self) -> ContentExtractor:
        if self._content_extractor is None:
            self._content_extractor = ContentExtractor()
        return self._content_extractor
    
    def _get_classifier(self) -> ContentClassifier:
        if self._classifier is None:
            self._classifier = create_classifier()
        return self._classifier
    
    def _get_info_extractor(self) -> ContentExtractorLLM:
        if self._info_extractor is None:
            self._info_extractor = create_extractor()
        return self._info_extractor
    
    def _get_trend_analyzer(self) -> TrendAnalyzer:
        if self._trend_analyzer is None:
            self._trend_analyzer = create_trend_analyzer()
        return self._trend_analyzer
    
    def _get_report_generator(self) -> ReportGenerator:
        if self._report_generator is None:
            report_config = ReportConfig(output_dir=self.config.output_dir)
            self._report_generator = create_report_generator(report_config)
        return self._report_generator
    
    # ==================== 数据采集 ====================
    
    def collect_data(self) -> List[Dict[str, Any]]:
        """
        从所有启用的数据源采集数据
        
        Returns:
            采集到的原始数据列表
        """
        logger.info("=" * 50)
        logger.info("开始数据采集阶段")
        logger.info("=" * 50)
        
        all_items = []
        
        # HackerNews
        if self.config.enable_hackernews:
            try:
                logger.info("采集 HackerNews...")
                collector = self._get_hn_collector()
                stories = collector.fetch_top_stories(limit=self.config.hn_max_stories)
                
                for story in stories:
                    item = {
                        "id": f"hn_{story.id}",
                        "source": "HackerNews",
                        "title": story.title,
                        "url": story.url,
                        "content": story.text or "",
                        "author": story.by,
                        "score": story.score,
                        "publish_time": datetime.fromtimestamp(story.time).isoformat() if story.time else "",
                        "comments_count": story.descendants or 0
                    }
                    all_items.append(item)
                
                logger.info(f"HackerNews: 采集 {len(stories)} 条")
            except Exception as e:
                err_message = get_err_message()
                logger.error(f"HackerNews采集失败: {err_message}")
        
        # RSS订阅
        if self.config.enable_rss:
            try:
                logger.info("采集 RSS订阅...")
                collector = self._get_rss_collector()
                feeds = collector.fetch_all(parallel=True)
                
                for feed_item in feeds[:self.config.rss_max_items]:
                    item = {
                        "id": feed_item.id,
                        "source": feed_item.source_name,
                        "title": feed_item.title,
                        "url": feed_item.link,
                        "content": feed_item.content or feed_item.description or "",
                        "author": feed_item.author,
                        "publish_time": feed_item.published_at or "",
                        "categories": feed_item.categories
                    }
                    all_items.append(item)
                
                logger.info(f"RSS: 采集 {len(feeds)} 条")
            except Exception as e:
                err_message = get_err_message()
                logger.error(f"RSS采集失败: {err_message}")
        
        # 新浪财经直播
        if self.config.enable_sina:
            try:
                logger.info("采集 新浪财经直播...")
                collector = self._get_sina_collector()
                zhibo_items = collector.fetch_all_channels(pages_per_channel=2)
                
                for zhibo_item_list in zhibo_items.values():
                    for zhibo in zhibo_item_list[:self.config.sina_max_items]:
                        item = {
                            "id": zhibo.id,
                            "source": f"新浪财经-{zhibo.channel_name}",
                            "title": zhibo.content[:50] if zhibo.content else "",
                            "url": zhibo.doc_url or "",
                            "content": zhibo.content,
                            "publish_time": str(zhibo.create_time),
                            "tags": zhibo.tags,
                            "stocks": zhibo.stocks
                        }
                        all_items.append(item)
                
                logger.info(f"新浪财经: 采集 {len(all_items)} 条")
            except Exception as e:
                err_message = get_err_message()
                logger.error(f"新浪财经采集失败: {err_message}")
        
        self.raw_items = all_items
        logger.info(f"数据采集完成, 总计: {len(all_items)} 条")
        
        # 缓存原始数据
        if self.config.use_cache:
            cache_file = os.path.join(self.config.cache_dir, f"{self.date}_raw_items.json")
            save_json(all_items, cache_file)
            logger.info(f"原始数据已缓存: {cache_file}")
        
        return all_items
    
    # ==================== 内容抽取 ====================
    
    def extract_content(self, items: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        抽取文章全文内容
        
        Args:
            items: 待处理的条目列表
            
        Returns:
            包含全文内容的条目列表
        """
        items = items or self.raw_items
        
        if not items:
            logger.warning("无数据需要抽取内容")
            return []
        
        if not self.config.extract_full_content:
            logger.info("跳过内容抽取(已禁用)")
            return items
        
        logger.info("=" * 50)
        logger.info("开始内容抽取阶段")
        logger.info("=" * 50)
        
        extractor = self._get_content_extractor()
        
        # 需要抽取的URL
        urls_to_extract = []
        url_to_idx = {}
        
        for i, item in enumerate(items):
            url = item.get("url", "")
            # 只抽取有URL且内容较短的条目
            if url and len(item.get("content", "")) < 500:
                urls_to_extract.append(url)
                url_to_idx[url] = i
        
        logger.info(f"需要抽取全文: {len(urls_to_extract)} 条")
        
        if urls_to_extract:
            # 批量抽取
            extracted = extractor.extract_batch(
                urls_to_extract[:50],  # 限制数量
                # max_workers=self.config.max_extract_workers
            )
            
            # 更新内容
            success_count = 0
            for content in extracted:
                if content.url in url_to_idx and content.text:
                    idx = url_to_idx[content.url]
                    items[idx]["content"] = content.text
                    items[idx]["extracted_title"] = content.title
                    items[idx]["extracted_author"] = content.author
                    success_count += 1
            
            logger.info(f"成功抽取全文: {success_count} 条")
        
        return items
    
    # ==================== 分类标注 ====================
    
    def classify_items(self, items: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        对条目进行分类和重要性标注
        
        Args:
            items: 待分类的条目列表
            
        Returns:
            包含分类结果的条目列表
        """
        items = items or self.raw_items
        
        if not items:
            logger.warning("无数据需要分类")
            return []
        
        if not self.config.enable_classification:
            logger.info("跳过分类(已禁用)")
            return items
        
        logger.info("=" * 50)
        logger.info("开始分类标注阶段")
        logger.info("=" * 50)
        
        # 检查Ollama是否可用
        if not check_ollama_available():
            logger.warning("Ollama不可用，使用规则分类")
            return self._rule_based_classify(items)
        
        classifier = self._get_classifier()
        
        # 批量分类
        results = classifier.classify_batch(
            items,
            max_workers=3,
            show_progress=True
        )
        
        # 更新条目
        for i, result in enumerate(results):
            items[i]["category"] = result.category.value if hasattr(result.category, 'value') else result.category
            items[i]["importance"] = result.importance
            items[i]["classification_confidence"] = result.confidence
            items[i]["classification_reason"] = result.reason
        
        logger.info(f"分类完成: {len(items)} 条")
        
        return items
    
    def _rule_based_classify(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于规则的分类（后备方案）"""
        from config.settings import CLASSIFICATION_KEYWORDS, IMPORTANCE_KEYWORDS
        
        for item in items:
            text = f"{item.get('title', '')} {item.get('content', '')[:200]}".lower()
            
            # 类别匹配
            best_category = "other"
            best_score = 0
            
            for category, keywords in CLASSIFICATION_KEYWORDS.items():
                score = sum(1 for kw in keywords if kw.lower() in text)
                if score > best_score:
                    best_score = score
                    best_category = category
            
            item["category"] = best_category
            
            # 重要性评估
            importance = 3  # 默认中等
            for keyword, delta in IMPORTANCE_KEYWORDS.items():
                if keyword.lower() in text:
                    importance += delta
            
            item["importance"] = max(1, min(5, importance))
        
        return items
    
    # ==================== 信息抽取 ====================
    
    def extract_info(self, items: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        抽取5W2H结构化信息
        
        Args:
            items: 待抽取的条目列表
            
        Returns:
            包含结构化信息的条目列表
        """
        items = items or self.raw_items
        
        if not items:
            logger.warning("无数据需要信息抽取")
            return []
        
        if not self.config.enable_extraction:
            logger.info("跳过信息抽取(已禁用)")
            return items
        
        logger.info("=" * 50)
        logger.info("开始信息抽取阶段")
        logger.info("=" * 50)
        
        # 检查Ollama是否可用
        if not check_ollama_available():
            logger.warning("Ollama不可用，跳过LLM信息抽取")
            return items
        
        extractor = self._get_info_extractor()
        
        # 只对重要文章进行深度抽取
        important_items = [item for item in items if item.get("importance", 3) >= 4]
        logger.info(f"高重要性文章: {len(important_items)} 条")
        
        if important_items:
            results = extractor.extract_batch(
                important_items,
                max_workers=2,
                show_progress=True
            )
            
            # 更新条目
            for item, result in zip(important_items, results):
                if result.is_valid():
                    item["extracted_what"] = result.what
                    item["extracted_who"] = result.who
                    item["extracted_when"] = result.when
                    item["extracted_where"] = result.where
                    item["summary"] = result.summary or item.get("title", "")[:50]
                    item["key_points"] = result.key_points
        
        logger.info(f"信息抽取完成")
        
        return items
    
    # ==================== 趋势分析 ====================
    
    def analyze_trends(self, items: List[Dict[str, Any]] = None):
        """
        进行趋势分析
        
        Args:
            items: 待分析的条目列表
            
        Returns:
            TrendReport: 趋势分析报告
        """
        items = items or self.raw_items
        
        if not items:
            logger.warning("无数据需要趋势分析")
            return None
        
        if not self.config.enable_trend_analysis:
            logger.info("跳过趋势分析(已禁用)")
            return None
        
        logger.info("=" * 50)
        logger.info("开始趋势分析阶段")
        logger.info("=" * 50)
        
        analyzer = self._get_trend_analyzer()
        
        self.trend_report = analyzer.analyze(
            items,
            top_n=10
        )
        
        logger.info(f"趋势分析完成, 热点: {len(self.trend_report.hot_topics)}")
        
        return self.trend_report
    
    # ==================== 报告生成 ====================
    
    def generate_reports(
        self,
        items: List[Dict[str, Any]] = None,
        trend_report = None
    ) -> Dict[str, str]:
        """
        生成分析报告
        
        Args:
            items: 文章列表
            trend_report: 趋势报告
            
        Returns:
            格式到文件路径的映射
        """
        items = items or self.raw_items
        trend_report = trend_report or self.trend_report
        
        if not items:
            logger.warning("无数据生成报告")
            return {}
        
        logger.info("=" * 50)
        logger.info("开始报告生成阶段")
        logger.info("=" * 50)
        
        generator = self._get_report_generator()
        
        # 如果没有趋势报告，创建一个基础的
        if not trend_report:
            from analyzers.trend_analyzer import TrendReport
            trend_report = TrendReport(
                report_time=datetime.now().isoformat(),
                total_articles=len(items)
            )
        
        results = {}
        
        for fmt in self.config.report_formats:
            try:
                filepath = generator.generate_daily_report(
                    trend_report,
                    items,
                    self.date,
                    fmt
                )
                results[fmt] = filepath
                logger.info(f"生成{fmt}报告: {filepath}")
            except Exception as e:
                err_message = get_err_message()
                logger.error(f"生成{fmt}报告失败: {err_message}")
        
        return results
    
    # ==================== 完整流水线 ====================
    
    def run(self, skip_collect: bool = False) -> Dict[str, Any]:
        """
        运行完整流水线
        
        Args:
            skip_collect: 是否跳过采集阶段（使用缓存数据）
            
        Returns:
            运行结果摘要
        """
        start_time = datetime.now()
        
        logger.info("=" * 60)
        logger.info(f"新闻聚合分析流水线启动 - {self.date}")
        logger.info("=" * 60)
        
        result = {
            "date": self.date,
            "start_time": start_time.isoformat(),
            "stages": {},
            "reports": {},
            "success": False
        }
        
        try:
            # 阶段1: 数据采集
            if skip_collect and self.config.use_cache:
                cache_file = os.path.join(self.config.cache_dir, f"{self.date}_raw_items.json")
                if os.path.exists(cache_file):
                    self.raw_items = load_json(cache_file)
                    logger.info(f"从缓存加载数据: {len(self.raw_items)} 条")
                    result["stages"]["collect"] = {"status": "cached", "count": len(self.raw_items)}
                else:
                    self.collect_data()
                    result["stages"]["collect"] = {"status": "done", "count": len(self.raw_items)}
            else:
                self.collect_data()
                result["stages"]["collect"] = {"status": "done", "count": len(self.raw_items)}
            
            if not self.raw_items:
                logger.error("无数据，流水线终止")
                result["error"] = "No data collected"
                return result
            
            # 阶段2: 内容抽取
            self.extract_content()
            result["stages"]["extract_content"] = {"status": "done"}
            
            # 阶段3: 分类标注
            self.classify_items()
            result["stages"]["classify"] = {"status": "done"}
            
            # 阶段4: 信息抽取
            self.extract_info()
            result["stages"]["extract_info"] = {"status": "done"}
            
            # 阶段5: 趋势分析
            self.analyze_trends()
            result["stages"]["trend_analysis"] = {"status": "done"}
            
            # 阶段6: 报告生成
            reports = self.generate_reports()
            result["reports"] = reports
            result["stages"]["report"] = {"status": "done", "files": list(reports.values())}
            
            result["success"] = True
            
        except Exception as e:
            err_message = get_err_message()
            logger.error(f"流水线异常: {err_message}")
            result["error"] = str(err_message)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result["end_time"] = end_time.isoformat()
        result["duration_seconds"] = duration
        
        logger.info("=" * 60)
        logger.info(f"流水线完成 - 耗时: {duration:.1f}秒")
        logger.info(f"结果: {'成功' if result['success'] else '失败'}")
        logger.info("=" * 60)
        
        # 保存运行结果
        result_file = os.path.join(self.config.output_dir, f"{self.date}_pipeline_result.json")
        save_json(result, result_file)
        
        return result


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="新闻聚合分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="仅采集数据，不进行分析"
    )
    
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="仅分析（使用缓存数据）"
    )
    
    parser.add_argument(
        "--sources",
        type=str,
        default="all",
        help="数据源，逗号分隔 (hn,rss,sina) 或 'all'"
    )
    
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="禁用LLM分析（使用规则分类）"
    )
    
    parser.add_argument(
        "--formats",
        type=str,
        default="markdown,json",
        help="报告格式，逗号分隔 (markdown,json,html)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="输出目录"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 配置日志
    logger.remove()
    log_level = "DEBUG" if args.debug else "INFO"
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:7}</level> | <cyan>{function}</cyan> - {message}"
    )
    
    # 日志文件
    log_file = f"logs/{datetime.now().strftime('%Y-%m-%d')}.log"
    ensure_dir(os.path.dirname(log_file))
    logger.add(log_file, level="DEBUG", rotation="1 day")
    
    # 初始化系统
    init_system()
    
    # 验证配置
    if not validate_config():
        logger.error("配置验证失败")
        sys.exit(1)
    
    # 构建流水线配置
    config = PipelineConfig(output_dir=args.output_dir)
    
    # 解析数据源
    if args.sources != "all":
        sources = args.sources.lower().split(",")
        config.enable_hackernews = "hn" in sources
        config.enable_rss = "rss" in sources
        config.enable_sina = "sina" in sources
    
    # LLM设置
    if args.no_llm:
        config.enable_classification = False  # 使用规则分类
        config.enable_extraction = False
    
    # 报告格式
    config.report_formats = args.formats.split(",")
    
    # 创建流水线
    pipeline = NewsAggregatorPipeline(config)
    
    # 运行
    if args.collect_only:
        pipeline.collect_data()
        logger.info("数据采集完成")
    elif args.analyze_only:
        result = pipeline.run(skip_collect=True)
        logger.info(f"分析完成: {result}")
    else:
        result = pipeline.run()
        logger.info(f"流水线完成: {result}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

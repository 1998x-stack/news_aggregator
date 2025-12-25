#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势分析模块

分析新闻数据集，识别:
- 热点话题
- 新兴趋势
- 行业动态
- 关键事件

作者: Claude
日期: 2024-12
"""

import sys
import json
import traceback
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, 'news_aggregator')

from utils.ollama_client import OllamaClient, OllamaConfig
from prompts.llm_prompts import get_prompt, PromptType
from config.settings import LLMSettings, ContentCategory


@dataclass
class TrendItem:
    """趋势条目"""
    name: str  # 趋势名称
    category: str  # 所属类别
    heat_score: float  # 热度评分 (0-100)
    mention_count: int  # 提及次数
    sources: List[str] = field(default_factory=list)  # 来源列表
    keywords: List[str] = field(default_factory=list)  # 相关关键词
    summary: str = ""  # 趋势描述
    first_seen: str = ""  # 首次出现时间
    last_seen: str = ""  # 最后出现时间
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "heat_score": self.heat_score,
            "mention_count": self.mention_count,
            "sources": self.sources,
            "keywords": self.keywords,
            "summary": self.summary,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen
        }


@dataclass
class IndustryDynamic:
    """行业动态"""
    industry: str  # 行业名称
    sentiment: str  # 整体情绪: positive/negative/neutral
    key_events: List[str] = field(default_factory=list)  # 关键事件
    major_players: List[str] = field(default_factory=list)  # 主要参与者
    outlook: str = ""  # 展望
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "industry": self.industry,
            "sentiment": self.sentiment,
            "key_events": self.key_events,
            "major_players": self.major_players,
            "outlook": self.outlook
        }


@dataclass
class TrendReport:
    """趋势分析报告"""
    
    # 报告元数据
    report_time: str = ""
    analysis_period: str = ""
    total_articles: int = 0
    
    # 核心内容
    hot_topics: List[TrendItem] = field(default_factory=list)  # 热点话题 Top10
    emerging_trends: List[TrendItem] = field(default_factory=list)  # 新兴趋势
    industry_dynamics: List[IndustryDynamic] = field(default_factory=list)  # 行业动态
    key_events: List[Dict[str, str]] = field(default_factory=list)  # 关键事件
    
    # 统计信息
    category_distribution: Dict[str, int] = field(default_factory=dict)  # 类别分布
    source_distribution: Dict[str, int] = field(default_factory=dict)  # 来源分布
    sentiment_distribution: Dict[str, int] = field(default_factory=dict)  # 情绪分布
    
    # AI分析
    outlook: str = ""  # 趋势展望
    recommendations: List[str] = field(default_factory=list)  # 建议关注
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_time": self.report_time,
            "analysis_period": self.analysis_period,
            "total_articles": self.total_articles,
            "hot_topics": [t.to_dict() for t in self.hot_topics],
            "emerging_trends": [t.to_dict() for t in self.emerging_trends],
            "industry_dynamics": [d.to_dict() for d in self.industry_dynamics],
            "key_events": self.key_events,
            "category_distribution": self.category_distribution,
            "source_distribution": self.source_distribution,
            "sentiment_distribution": self.sentiment_distribution,
            "outlook": self.outlook,
            "recommendations": self.recommendations
        }


class TrendAnalyzer:
    """
    趋势分析器
    
    结合统计分析和LLM分析，识别新闻数据中的趋势和热点
    
    Attributes:
        client: Ollama客户端
        model: 使用的模型名称
    """
    
    def __init__(
        self,
        model: str = None,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0
    ):
        """
        初始化趋势分析器
        
        Args:
            model: 模型名称
            base_url: Ollama服务地址
            timeout: 请求超时时间
        """
        self.model = model or LLMSettings.EXTRACTOR_MODEL  # 使用较大模型
        
        config = OllamaConfig(
            base_url=base_url,
            timeout=timeout,
            default_model=self.model
        )
        self.client = OllamaClient(config)
        
        # 关键词权重配置
        self.keyword_weights = {
            "突破": 2.0, "首次": 2.0, "重大": 1.8, "发布": 1.5,
            "融资": 1.5, "收购": 1.8, "上市": 1.8, "破产": 2.0,
            "breakthrough": 2.0, "first": 1.8, "major": 1.5,
            "launch": 1.5, "funding": 1.5, "acquisition": 1.8
        }
        
        logger.info(f"TrendAnalyzer初始化完成, 模型: {self.model}")
    
    def analyze(
        self,
        articles: List[Dict[str, Any]],
        time_range: Tuple[datetime, datetime] = None,
        top_n: int = 10
    ) -> TrendReport:
        """
        分析文章集合，生成趋势报告
        
        Args:
            articles: 文章列表，每篇应包含title, content, category, source, publish_time等字段
            time_range: 分析时间范围
            top_n: 返回的热点数量
            
        Returns:
            TrendReport: 趋势分析报告
        """
        report = TrendReport(
            report_time=datetime.now().isoformat(),
            total_articles=len(articles)
        )
        
        if not articles:
            logger.warning("无文章数据，返回空报告")
            return report
        
        # 设置分析时间范围
        if time_range:
            report.analysis_period = f"{time_range[0].date()} 至 {time_range[1].date()}"
        else:
            report.analysis_period = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"开始趋势分析, 文章数: {len(articles)}")
        
        # 1. 统计分析
        self._compute_distributions(articles, report)
        
        # 2. 关键词提取和热点识别
        hot_topics = self._identify_hot_topics(articles, top_n)
        report.hot_topics = hot_topics
        
        # 3. 识别新兴趋势
        emerging = self._identify_emerging_trends(articles)
        report.emerging_trends = emerging
        
        # 4. 提取关键事件
        key_events = self._extract_key_events(articles)
        report.key_events = key_events
        
        # 5. LLM深度分析
        if len(articles) >= 5:
            self._llm_deep_analysis(articles, report)
        
        logger.info(f"趋势分析完成, 热点: {len(report.hot_topics)}, 新兴趋势: {len(report.emerging_trends)}")
        
        return report
    
    def _compute_distributions(
        self,
        articles: List[Dict[str, Any]],
        report: TrendReport
    ) -> None:
        """计算各类分布统计"""
        category_counter = Counter()
        source_counter = Counter()
        sentiment_counter = Counter()
        
        for article in articles:
            # 类别分布
            category = article.get("category", "other")
            if isinstance(category, ContentCategory):
                category = category.value
            category_counter[category] += 1
            
            # 来源分布
            source = article.get("source", "unknown")
            source_counter[source] += 1
            
            # 情绪分布
            sentiment = article.get("sentiment", "neutral")
            sentiment_counter[sentiment] += 1
        
        report.category_distribution = dict(category_counter.most_common(20))
        report.source_distribution = dict(source_counter.most_common(20))
        report.sentiment_distribution = dict(sentiment_counter)
    
    def _identify_hot_topics(
        self,
        articles: List[Dict[str, Any]],
        top_n: int = 10
    ) -> List[TrendItem]:
        """
        识别热点话题
        
        使用TF-IDF类似的方法计算关键词热度
        """
        # 收集所有标题和摘要中的关键词
        keyword_articles = defaultdict(list)
        keyword_sources = defaultdict(set)
        keyword_times = defaultdict(list)
        
        for article in articles:
            title = article.get("title", "")
            summary = article.get("summary", article.get("content", ""))[:200]
            source = article.get("source", "")
            pub_time = article.get("publish_time", "")
            
            # 简单的关键词提取（实际应用中可使用jieba等分词工具）
            text = f"{title} {summary}".lower()
            words = self._extract_keywords(text)
            
            for word in words:
                keyword_articles[word].append(article)
                keyword_sources[word].add(source)
                if pub_time:
                    keyword_times[word].append(pub_time)
        
        # 计算热度评分
        hot_topics = []
        
        for keyword, related_articles in keyword_articles.items():
            if len(related_articles) < 2:  # 至少出现2次
                continue
            
            # 热度计算: 文章数 * 来源多样性 * 关键词权重
            mention_count = len(related_articles)
            source_diversity = len(keyword_sources[keyword])
            weight = self.keyword_weights.get(keyword, 1.0)
            
            heat_score = min(100, mention_count * 10 * (1 + source_diversity * 0.2) * weight)
            
            # 获取时间范围
            times = keyword_times[keyword]
            first_seen = min(times) if times else ""
            last_seen = max(times) if times else ""
            
            # 确定类别（取最常见的）
            categories = [a.get("category", "other") for a in related_articles]
            most_common_cat = Counter(categories).most_common(1)[0][0]
            if isinstance(most_common_cat, ContentCategory):
                most_common_cat = most_common_cat.value
            
            topic = TrendItem(
                name=keyword,
                category=most_common_cat,
                heat_score=round(heat_score, 1),
                mention_count=mention_count,
                sources=list(keyword_sources[keyword])[:5],
                keywords=[keyword],
                first_seen=first_seen,
                last_seen=last_seen
            )
            hot_topics.append(topic)
        
        # 按热度排序
        hot_topics.sort(key=lambda x: x.heat_score, reverse=True)
        
        return hot_topics[:top_n]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        从文本中提取关键词
        
        简化实现，实际应用建议使用jieba或其他NLP工具
        """
        import re
        
        # 停用词列表
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall",
            "的", "了", "是", "在", "和", "与", "或", "但", "而", "且",
            "为", "以", "及", "等", "到", "从", "向", "对", "于", "中"
        }
        
        # 分词
        words = re.findall(r'\b[a-zA-Z]{3,15}\b|[\u4e00-\u9fa5]{2,8}', text.lower())
        
        # 过滤
        keywords = [w for w in words if w not in stopwords and len(w) >= 2]
        
        return keywords
    
    def _identify_emerging_trends(
        self,
        articles: List[Dict[str, Any]],
        threshold: int = 3
    ) -> List[TrendItem]:
        """
        识别新兴趋势
        
        查找最近出现频率增加的话题
        """
        # 按时间分组
        time_groups = defaultdict(list)
        
        for article in articles:
            pub_time = article.get("publish_time", "")
            if pub_time:
                # 简化：按日期分组
                date_key = pub_time[:10] if len(pub_time) >= 10 else pub_time
                time_groups[date_key].append(article)
        
        if len(time_groups) < 2:
            return []
        
        # 比较最近和之前的关键词频率
        sorted_dates = sorted(time_groups.keys())
        recent_date = sorted_dates[-1]
        
        recent_keywords = Counter()
        old_keywords = Counter()
        
        for date, group in time_groups.items():
            for article in group:
                text = f"{article.get('title', '')} {article.get('summary', '')[:100]}"
                keywords = self._extract_keywords(text)
                
                if date == recent_date:
                    recent_keywords.update(keywords)
                else:
                    old_keywords.update(keywords)
        
        # 找出新增或大幅增长的关键词
        emerging = []
        
        for keyword, count in recent_keywords.most_common(50):
            if count < threshold:
                continue
            
            old_count = old_keywords.get(keyword, 0)
            
            # 新出现或增长超过2倍
            if old_count == 0 or count / max(old_count, 1) > 2:
                trend = TrendItem(
                    name=keyword,
                    category="emerging",
                    heat_score=min(100, count * 15),
                    mention_count=count,
                    summary=f"新兴关键词，近期提及{count}次",
                    first_seen=recent_date
                )
                emerging.append(trend)
        
        emerging.sort(key=lambda x: x.heat_score, reverse=True)
        
        return emerging[:5]
    
    def _extract_key_events(
        self,
        articles: List[Dict[str, Any]],
        max_events: int = 10
    ) -> List[Dict[str, str]]:
        """
        提取关键事件
        
        基于重要性评分和关键词匹配
        """
        event_keywords = {
            "发布", "宣布", "推出", "上线", "开源",
            "融资", "收购", "合并", "上市", "退市",
            "突破", "首次", "创新", "里程碑",
            "launch", "announce", "release", "acquire",
            "funding", "ipo", "breakthrough"
        }
        
        key_events = []
        
        for article in articles:
            title = article.get("title", "")
            importance = article.get("importance", 3)
            
            # 检查是否包含事件关键词
            title_lower = title.lower()
            has_event_keyword = any(kw in title_lower for kw in event_keywords)
            
            if importance >= 4 or has_event_keyword:
                event = {
                    "title": title,
                    "source": article.get("source", ""),
                    "time": article.get("publish_time", ""),
                    "category": str(article.get("category", "other")),
                    "importance": str(importance)
                }
                key_events.append(event)
        
        # 按重要性排序
        key_events.sort(key=lambda x: int(x.get("importance", 0)), reverse=True)
        
        return key_events[:max_events]
    
    def _llm_deep_analysis(
        self,
        articles: List[Dict[str, Any]],
        report: TrendReport
    ) -> None:
        """
        使用LLM进行深度趋势分析
        """
        # 准备输入数据
        titles = [a.get("title", "") for a in articles[:30]]  # 限制数量
        categories = list(report.category_distribution.keys())[:10]
        
        analysis_input = {
            "titles": titles,
            "categories": categories,
            "hot_topics": [t.name for t in report.hot_topics[:5]],
            "total_count": len(articles)
        }
        
        # 构建提示词
        prompt_template = get_prompt(PromptType.TREND_ANALYSIS)
        user_prompt = prompt_template.format(
            titles=json.dumps(analysis_input["titles"], ensure_ascii=False),
            categories=json.dumps(analysis_input["categories"], ensure_ascii=False),
            hot_topics=json.dumps(analysis_input["hot_topics"], ensure_ascii=False)
        )
        
        try:
            response = self.client.generate(
                prompt=user_prompt,
                model=self.model,
                temperature=0.4,
                max_tokens=2048
            )
            
            if response.success:
                parsed = self._parse_llm_response(response.text)
                if parsed:
                    # 更新报告
                    report.outlook = parsed.get("outlook", "")
                    report.recommendations = parsed.get("recommendations", [])
                    
                    # 行业动态
                    dynamics = parsed.get("industry_dynamics", {})
                    for industry, info in dynamics.items():
                        dynamic = IndustryDynamic(
                            industry=industry,
                            sentiment=info.get("sentiment", "neutral"),
                            key_events=info.get("events", []),
                            major_players=info.get("players", []),
                            outlook=info.get("outlook", "")
                        )
                        report.industry_dynamics.append(dynamic)
                    
                    logger.info("LLM深度分析完成")
            else:
                logger.warning(f"LLM分析失败: {response.error}")
                
        except Exception as e:
            logger.error(f"LLM深度分析异常: {e}")
            logger.debug(traceback.format_exc())
    
    def _parse_llm_response(self, text: str) -> Optional[Dict[str, Any]]:
        """解析LLM响应"""
        import re
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取JSON
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    clean = match.strip()
                    if clean.startswith('{'):
                        return json.loads(clean)
                except:
                    continue
        
        return None
    
    def compare_periods(
        self,
        current_articles: List[Dict[str, Any]],
        previous_articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        比较两个时期的趋势变化
        
        Args:
            current_articles: 当前时期文章
            previous_articles: 前一时期文章
            
        Returns:
            趋势比较结果
        """
        current_report = self.analyze(current_articles, top_n=10)
        previous_report = self.analyze(previous_articles, top_n=10)
        
        # 计算变化
        current_topics = {t.name: t for t in current_report.hot_topics}
        previous_topics = {t.name: t for t in previous_report.hot_topics}
        
        rising_topics = []
        falling_topics = []
        new_topics = []
        
        for name, topic in current_topics.items():
            if name in previous_topics:
                old_heat = previous_topics[name].heat_score
                change = topic.heat_score - old_heat
                if change > 10:
                    rising_topics.append({"name": name, "change": change})
                elif change < -10:
                    falling_topics.append({"name": name, "change": change})
            else:
                new_topics.append({"name": name, "heat": topic.heat_score})
        
        return {
            "current_period": {
                "article_count": current_report.total_articles,
                "top_topics": [t.name for t in current_report.hot_topics[:5]]
            },
            "previous_period": {
                "article_count": previous_report.total_articles,
                "top_topics": [t.name for t in previous_report.hot_topics[:5]]
            },
            "changes": {
                "rising_topics": rising_topics,
                "falling_topics": falling_topics,
                "new_topics": new_topics
            }
        }


def create_trend_analyzer(
    model: str = None,
    base_url: str = "http://localhost:11434"
) -> TrendAnalyzer:
    """工厂函数: 创建趋势分析器"""
    return TrendAnalyzer(model=model, base_url=base_url)


# 测试代码
if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    
    # 测试数据
    test_articles = [
        {"title": "OpenAI发布GPT-5模型", "category": "ai_ml", "source": "TechCrunch", "importance": 5, "publish_time": "2024-12-20"},
        {"title": "谷歌推出Gemini 2.0", "category": "ai_ml", "source": "The Verge", "importance": 5, "publish_time": "2024-12-19"},
        {"title": "AI芯片市场竞争加剧", "category": "hardware", "source": "Wired", "importance": 4, "publish_time": "2024-12-18"},
        {"title": "微软Azure云服务更新", "category": "cloud_infra", "source": "TechCrunch", "importance": 3, "publish_time": "2024-12-17"},
        {"title": "字节跳动开源大模型", "category": "ai_ml", "source": "36氪", "importance": 4, "publish_time": "2024-12-20"},
    ]
    
    print("=" * 60)
    print("趋势分析测试")
    print("=" * 60)
    
    analyzer = create_trend_analyzer()
    report = analyzer.analyze(test_articles)
    
    print(f"\n分析报告:")
    print(f"  总文章数: {report.total_articles}")
    print(f"  类别分布: {report.category_distribution}")
    print(f"  热点话题: {[t.name for t in report.hot_topics]}")
    print(f"  关键事件数: {len(report.key_events)}")

"""
RSS Feed 数据采集模块
RSS Feed Data Collector Module

支持多种 RSS/Atom 格式的科技新闻源
"""
import feedparser
import hashlib
import time
import traceback
import sys
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger


@dataclass
class FeedItem:
    """RSS Feed 条目数据类"""
    id: str                              # 唯一标识符
    title: str                           # 标题
    link: str                            # 链接
    description: Optional[str] = None    # 摘要/描述
    content: Optional[str] = None        # 完整内容
    author: Optional[str] = None         # 作者
    published_at: Optional[datetime] = None  # 发布时间
    updated_at: Optional[datetime] = None    # 更新时间
    categories: List[str] = field(default_factory=list)  # 分类标签
    source_name: str = ""                # 来源名称
    source_url: str = ""                 # 来源 URL
    media_url: Optional[str] = None      # 媒体 URL (图片/视频)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        data['published_at'] = self.published_at.isoformat() if self.published_at else None
        data['updated_at'] = self.updated_at.isoformat() if self.updated_at else None
        return data


@dataclass
class FeedSource:
    """RSS 源配置"""
    name: str                  # 源名称
    url: str                   # RSS URL
    category: str = "general"  # 分类
    language: str = "en"       # 语言
    priority: int = 5          # 优先级 1-10
    enabled: bool = True       # 是否启用
    fetch_interval: int = 600  # 获取间隔(秒)
    
    def __hash__(self):
        return hash(self.url)


class RSSCollector:
    """RSS 内容采集器"""
    
    # 预定义科技类 RSS 源
    DEFAULT_SOURCES = [
        # === 英文科技媒体 ===
        FeedSource("TechCrunch", "https://techcrunch.com/feed/", "tech_news", "en", 9),
        FeedSource("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/", "ai", "en", 9),
        FeedSource("The Verge", "https://www.theverge.com/rss/index.xml", "tech_news", "en", 9),
        FeedSource("The Verge AI", "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml", "ai", "en", 9),
        FeedSource("Wired", "https://www.wired.com/feed/rss", "tech_news", "en", 8),
        FeedSource("Wired AI", "https://www.wired.com/feed/tag/ai/latest/rss", "ai", "en", 8),
        FeedSource("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index", "tech_news", "en", 8),
        FeedSource("VentureBeat", "https://venturebeat.com/feed/", "tech_news", "en", 7),
        FeedSource("VentureBeat AI", "https://venturebeat.com/category/ai/feed/", "ai", "en", 8),
        FeedSource("Engadget", "https://www.engadget.com/rss.xml", "tech_news", "en", 7),
        
        # === HN 相关 ===
        FeedSource("HN Frontpage", "https://hnrss.org/frontpage", "hacker_news", "en", 10),
        FeedSource("HN Best", "https://hnrss.org/best", "hacker_news", "en", 9),
        FeedSource("HN Newest", "https://hnrss.org/newest", "hacker_news", "en", 7),
        
        # === 学术/研究类 ===
        FeedSource("arXiv CS.AI", "https://export.arxiv.org/rss/cs.AI", "research", "en", 8),
        FeedSource("arXiv CS.LG", "https://export.arxiv.org/rss/cs.LG", "research", "en", 8),
        FeedSource("ScienceDaily AI", "https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml", "research", "en", 7),
        
        # === 技术博客 ===
        FeedSource("Towards Data Science", "https://towardsdatascience.com/feed", "tech_blog", "en", 7),
        FeedSource("Towards AI", "https://pub.towardsai.net/feed", "ai", "en", 7),
        
        # === 中文科技媒体 ===
        FeedSource("36氪", "https://36kr.com/feed", "tech_news", "zh", 8),
        FeedSource("少数派", "https://sspai.com/feed", "tech_news", "zh", 7),
    ]
    
    def __init__(
        self,
        timeout: int = 30,
        max_workers: int = 5,
        user_agent: str = "NewsAggregator/1.0"
    ):
        """
        初始化 RSS 采集器
        
        Args:
            timeout: 请求超时时间(秒)
            max_workers: 最大并发工作线程数
            user_agent: User-Agent 字符串
        """
        self.timeout = timeout
        self.max_workers = max_workers
        self.user_agent = user_agent
        self._sources: List[FeedSource] = []
        
    def add_source(self, source: FeedSource) -> None:
        """添加 RSS 源"""
        if source not in self._sources:
            self._sources.append(source)
            logger.debug(f"添加 RSS 源: {source.name}")
    
    def add_sources(self, sources: List[FeedSource]) -> None:
        """批量添加 RSS 源"""
        for source in sources:
            self.add_source(source)
    
    def add_default_sources(self) -> None:
        """添加默认科技 RSS 源"""
        self.add_sources(self.DEFAULT_SOURCES)
        logger.info(f"已添加 {len(self.DEFAULT_SOURCES)} 个默认 RSS 源")
    
    def _parse_datetime(self, time_struct) -> Optional[datetime]:
        """解析时间结构为 datetime"""
        if time_struct:
            try:
                return datetime(*time_struct[:6])
            except Exception:
                pass
        return None
    
    def _generate_id(self, entry: Dict, source_name: str) -> str:
        """生成条目唯一ID"""
        # 优先使用 guid 或 id
        if entry.get('id'):
            return str(entry['id'])
        if entry.get('guid'):
            return str(entry['guid'])
        # 否则基于链接和标题生成
        content = f"{source_name}:{entry.get('link', '')}:{entry.get('title', '')}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _extract_content(self, entry: Dict) -> Optional[str]:
        """提取完整内容"""
        # 尝试从 content 字段提取
        if entry.get('content'):
            for content in entry['content']:
                if content.get('type') in ['text/html', 'text/plain', None]:
                    return content.get('value', '')
        
        # 尝试从 summary_detail 提取
        if entry.get('summary_detail'):
            return entry['summary_detail'].get('value', '')
        
        return None
    
    def _extract_media(self, entry: Dict) -> Optional[str]:
        """提取媒体 URL"""
        # 尝试从 media_content 提取
        if entry.get('media_content'):
            for media in entry['media_content']:
                if media.get('url'):
                    return media['url']
        
        # 尝试从 media_thumbnail 提取
        if entry.get('media_thumbnail'):
            for thumb in entry['media_thumbnail']:
                if thumb.get('url'):
                    return thumb['url']
        
        # 尝试从 enclosures 提取
        if entry.get('enclosures'):
            for enc in entry['enclosures']:
                if enc.get('href'):
                    return enc['href']
        
        return None
    
    def _parse_entry(self, entry: Dict, source: FeedSource) -> FeedItem:
        """解析单个 feed entry"""
        # 提取分类标签
        categories = []
        if entry.get('tags'):
            categories = [t.get('term', '') for t in entry['tags'] if t.get('term')]
        
        return FeedItem(
            id=self._generate_id(entry, source.name),
            title=entry.get('title', ''),
            link=entry.get('link', ''),
            description=entry.get('summary', ''),
            content=self._extract_content(entry),
            author=entry.get('author', ''),
            published_at=self._parse_datetime(entry.get('published_parsed')),
            updated_at=self._parse_datetime(entry.get('updated_parsed')),
            categories=categories,
            source_name=source.name,
            source_url=source.url,
            media_url=self._extract_media(entry)
        )
    
    def fetch_feed(self, source: FeedSource) -> List[FeedItem]:
        """
        获取单个 RSS 源的内容
        
        Args:
            source: RSS 源配置
            
        Returns:
            FeedItem 列表
        """
        if not source.enabled:
            logger.debug(f"源 {source.name} 已禁用，跳过")
            return []
        
        logger.info(f"获取 RSS 源: {source.name}")
        
        try:
            # 设置请求头
            feedparser.USER_AGENT = self.user_agent
            
            # 解析 feed
            feed = feedparser.parse(source.url)
            
            # 检查解析状态
            if feed.bozo and feed.bozo_exception:
                logger.warning(f"RSS 解析警告 {source.name}: {feed.bozo_exception}")
            
            # 检查 HTTP 状态
            if hasattr(feed, 'status') and feed.status >= 400:
                logger.error(f"RSS 获取失败 {source.name}: HTTP {feed.status}")
                return []
            
            # 解析条目
            items = []
            for entry in feed.entries:
                try:
                    item = self._parse_entry(entry, source)
                    items.append(item)
                except Exception as e:
                    logger.error(f"解析条目失败: {e}")
                    continue
            
            logger.info(f"从 {source.name} 获取 {len(items)} 条内容")
            return items
            
        except Exception as e:
            logger.error(f"获取 RSS 源 {source.name} 失败: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def fetch_all(self, parallel: bool = True) -> List[FeedItem]:
        """
        获取所有启用源的内容
        
        Args:
            parallel: 是否并行获取
            
        Returns:
            FeedItem 列表
        """
        all_items = []
        enabled_sources = [s for s in self._sources if s.enabled]
        
        if not enabled_sources:
            logger.warning("没有启用的 RSS 源")
            return []
        
        logger.info(f"开始获取 {len(enabled_sources)} 个 RSS 源")
        
        if parallel:
            # 并行获取
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_source = {
                    executor.submit(self.fetch_feed, source): source
                    for source in enabled_sources
                }
                
                for future in as_completed(future_to_source):
                    source = future_to_source[future]
                    try:
                        items = future.result()
                        all_items.extend(items)
                    except Exception as e:
                        logger.error(f"获取 {source.name} 失败: {e}")
        else:
            # 串行获取
            for source in enabled_sources:
                items = self.fetch_feed(source)
                all_items.extend(items)
                time.sleep(0.5)  # 添加延迟避免过快请求
        
        logger.info(f"共获取 {len(all_items)} 条内容")
        return all_items
    
    def fetch_by_category(self, category: str) -> List[FeedItem]:
        """按分类获取内容"""
        items = []
        for source in self._sources:
            if source.enabled and source.category == category:
                items.extend(self.fetch_feed(source))
        return items
    
    def fetch_by_language(self, language: str) -> List[FeedItem]:
        """按语言获取内容"""
        items = []
        for source in self._sources:
            if source.enabled and source.language == language:
                items.extend(self.fetch_feed(source))
        return items
    
    def get_sources(self) -> List[FeedSource]:
        """获取所有源配置"""
        return self._sources.copy()
    
    def get_enabled_sources(self) -> List[FeedSource]:
        """获取启用的源配置"""
        return [s for s in self._sources if s.enabled]


def create_rss_collector(
    include_defaults: bool = True,
    custom_sources: Optional[List[Dict]] = None
) -> RSSCollector:
    """
    创建 RSS 采集器工厂方法
    
    Args:
        include_defaults: 是否包含默认源
        custom_sources: 自定义源配置列表
        
    Returns:
        配置好的 RSSCollector 实例
    """
    collector = RSSCollector()
    
    if include_defaults:
        collector.add_default_sources()
    
    if custom_sources:
        for source_config in custom_sources:
            source = FeedSource(**source_config)
            collector.add_source(source)
    
    return collector


if __name__ == "__main__":
    # 配置 loguru
    logger.remove()
    logger.add(
        sys.stderr,
        format="{time:HH:mm:ss} | {level: <8} | {message}",
        level="INFO"
    )
    
    # 测试采集器
    collector = create_rss_collector(include_defaults=True)
    
    # 只测试少量源
    test_sources = [
        FeedSource("HN Frontpage", "https://hnrss.org/frontpage", "hacker_news", "en", 10),
    ]
    
    test_collector = RSSCollector()
    test_collector.add_sources(test_sources)
    
    print("\n=== 测试 RSS 采集 ===")
    items = test_collector.fetch_all()
    
    for item in items[:5]:
        print(f"\n标题: {item.title}")
        print(f"来源: {item.source_name}")
        print(f"链接: {item.link}")
        print(f"时间: {item.published_at}")

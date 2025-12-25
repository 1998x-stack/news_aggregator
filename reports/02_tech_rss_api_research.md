# 科技类 RSS 信息源 API 详细调研报告

## 1. 概述

RSS (Really Simple Syndication) 是一种标准化的内容分发格式，广泛用于新闻聚合和内容订阅。本报告详细调研科技类 RSS 信息源及其 API 接口。

---

## 2. RSS 标准格式

### 2.1 RSS 2.0 结构

```xml
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>频道标题</title>
    <link>网站链接</link>
    <description>频道描述</description>
    <language>en-us</language>
    <lastBuildDate>Thu, 25 Dec 2025 10:00:00 GMT</lastBuildDate>
    
    <item>
      <title>文章标题</title>
      <link>文章链接</link>
      <description>文章摘要</description>
      <pubDate>Thu, 25 Dec 2025 09:00:00 GMT</pubDate>
      <guid>唯一标识符</guid>
      <author>作者</author>
      <category>分类</category>
    </item>
  </channel>
</rss>
```

### 2.2 Atom 格式

```xml
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>频道标题</title>
  <link href="网站链接"/>
  <updated>2025-12-25T10:00:00Z</updated>
  
  <entry>
    <title>文章标题</title>
    <link href="文章链接"/>
    <id>唯一标识符</id>
    <updated>2025-12-25T09:00:00Z</updated>
    <summary>文章摘要</summary>
    <author>
      <name>作者名</name>
    </author>
  </entry>
</feed>
```

---

## 3. 主流科技媒体 RSS 源

### 3.1 TechCrunch

**主要 RSS 端点**:

| 类别 | URL | 说明 |
|------|-----|------|
| 全站 | `https://techcrunch.com/feed/` | 所有文章 |
| 创业公司 | `https://techcrunch.com/category/startups/feed/` | 创业相关 |
| AI | `https://techcrunch.com/category/artificial-intelligence/feed/` | 人工智能 |
| 安全 | `https://techcrunch.com/category/security/feed/` | 安全相关 |
| 融资 | `https://techcrunch.com/tag/funding/feed` | 融资新闻 |
| 硬件 | `https://techcrunch.com/category/gadgets/feed/` | 硬件设备 |

**特点**:
- 更新频率: 高 (每天多篇)
- 内容质量: 高
- 聚焦: 创业、融资、科技产品

### 3.2 The Verge

**主要 RSS 端点**:

| 类别 | URL | 说明 |
|------|-----|------|
| 全站 | `https://www.theverge.com/rss/index.xml` | 所有文章 |
| 科技 | `https://www.theverge.com/tech/rss/index.xml` | 科技新闻 |
| 评测 | `https://www.theverge.com/reviews/rss/index.xml` | 产品评测 |
| AI | `https://www.theverge.com/ai-artificial-intelligence/rss/index.xml` | AI 相关 |

**特点**:
- 更新频率: 高
- 内容质量: 高
- 聚焦: 消费科技、数码产品

### 3.3 Wired

**主要 RSS 端点**:

| 类别 | URL | 说明 |
|------|-----|------|
| 全站 | `https://www.wired.com/feed/rss` | 所有文章 |
| AI | `https://www.wired.com/feed/tag/ai/latest/rss` | 人工智能 |
| 科技 | `https://www.wired.com/feed/category/gear/latest/rss` | 硬件设备 |
| 商业 | `https://www.wired.com/feed/category/business/latest/rss` | 商业科技 |
| 安全 | `https://www.wired.com/feed/category/security/latest/rss` | 网络安全 |

**特点**:
- 更新频率: 中高
- 内容质量: 高
- 聚焦: 深度报道、长篇分析

### 3.4 Ars Technica

**主要 RSS 端点**:

| 类别 | URL | 说明 |
|------|-----|------|
| 全站 | `https://feeds.arstechnica.com/arstechnica/index` | 所有文章 |
| 科技 | `https://feeds.arstechnica.com/arstechnica/technology-lab` | 技术实验室 |
| 科学 | `https://feeds.arstechnica.com/arstechnica/science` | 科学新闻 |
| 游戏 | `https://feeds.arstechnica.com/arstechnica/gaming` | 游戏相关 |

### 3.5 Engadget

**主要 RSS 端点**:

| 类别 | URL | 说明 |
|------|-----|------|
| 全站 | `https://www.engadget.com/rss.xml` | 所有文章 |

### 3.6 VentureBeat

**主要 RSS 端点**:

| 类别 | URL | 说明 |
|------|-----|------|
| 全站 | `https://venturebeat.com/feed/` | 所有文章 |
| AI | `https://venturebeat.com/category/ai/feed/` | AI 相关 |

### 3.7 MIT Technology Review

**主要 RSS 端点**:

| 类别 | URL | 说明 |
|------|-----|------|
| 全站 | `https://www.technologyreview.com/feed/` | 所有文章 |
| AI | `https://www.technologyreview.com/topic/artificial-intelligence/feed/` | AI |

### 3.8 Hacker News (非官方 RSS)

虽然 HN 不提供官方 RSS，但有第三方服务:

| 类别 | URL | 说明 |
|------|-----|------|
| 首页 | `https://hnrss.org/frontpage` | 首页文章 |
| 最新 | `https://hnrss.org/newest` | 最新提交 |
| 最佳 | `https://hnrss.org/best` | 最佳文章 |
| Ask HN | `https://hnrss.org/ask` | Ask HN |
| Show HN | `https://hnrss.org/show` | Show HN |
| 招聘 | `https://hnrss.org/jobs` | 招聘信息 |

**自定义参数**:
- `?points=100` - 最低分数
- `?comments=25` - 最低评论数
- `?q=python` - 关键词过滤

### 3.9 科学与学术类

| 来源 | URL | 说明 |
|------|-----|------|
| ScienceDaily AI | `https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml` | AI 科学新闻 |
| arXiv CS.AI | `https://export.arxiv.org/rss/cs.AI` | AI 论文 |
| arXiv CS.LG | `https://export.arxiv.org/rss/cs.LG` | 机器学习论文 |

### 3.10 Medium 科技类

| 类别 | URL | 说明 |
|------|-----|------|
| Towards Data Science | `https://towardsdatascience.com/feed` | 数据科学 |
| Towards AI | `https://pub.towardsai.net/feed` | AI 教程 |
| Better Programming | `https://betterprogramming.pub/feed` | 编程 |

---

## 4. 中文科技 RSS 源

### 4.1 36氪

| 类别 | URL | 说明 |
|------|-----|------|
| 全站 | `https://36kr.com/feed` | 所有文章 |

### 4.2 少数派

| 类别 | URL | 说明 |
|------|-----|------|
| 全站 | `https://sspai.com/feed` | 所有文章 |

### 4.3 InfoQ 中国

| 类别 | URL | 说明 |
|------|-----|------|
| 全站 | `https://www.infoq.cn/feed` | 技术文章 |

---

## 5. RSS 解析 Python 实现

### 5.1 使用 feedparser 库

```python
"""
RSS Feed 解析模块
使用 feedparser 解析各类 RSS/Atom 源
"""
import feedparser
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from loguru import logger
import time
import hashlib


@dataclass
class FeedItem:
    """RSS Feed 条目数据类"""
    id: str                          # 唯一标识符
    title: str                       # 标题
    link: str                        # 链接
    description: Optional[str] = None  # 摘要/描述
    content: Optional[str] = None    # 完整内容
    author: Optional[str] = None     # 作者
    published: Optional[datetime] = None  # 发布时间
    updated: Optional[datetime] = None    # 更新时间
    categories: List[str] = field(default_factory=list)  # 分类标签
    source: str = ""                 # 来源名称
    raw_data: Dict[str, Any] = field(default_factory=dict)  # 原始数据


@dataclass
class FeedSource:
    """RSS 源配置"""
    name: str           # 源名称
    url: str            # RSS URL
    category: str       # 分类
    language: str = "en"  # 语言
    priority: int = 5   # 优先级 1-10
    enabled: bool = True  # 是否启用


class RSSCollector:
    """RSS 内容采集器"""
    
    def __init__(self, timeout: int = 30):
        """
        初始化 RSS 采集器
        
        Args:
            timeout: 请求超时时间(秒)
        """
        self.timeout = timeout
        self._sources: List[FeedSource] = []
        
    def add_source(self, source: FeedSource) -> None:
        """添加 RSS 源"""
        self._sources.append(source)
        logger.info(f"添加 RSS 源: {source.name} - {source.url}")
    
    def add_sources_from_config(self, sources: List[Dict]) -> None:
        """从配置列表添加多个源"""
        for src in sources:
            self.add_source(FeedSource(**src))
    
    def _parse_datetime(self, time_struct) -> Optional[datetime]:
        """解析时间结构为 datetime"""
        if time_struct:
            try:
                return datetime(*time_struct[:6])
            except Exception:
                pass
        return None
    
    def _generate_id(self, item: Dict, source_name: str) -> str:
        """生成条目唯一ID"""
        # 优先使用 guid 或 id
        if item.get('id'):
            return item['id']
        if item.get('guid'):
            return item['guid']
        # 否则基于链接和标题生成
        content = f"{source_name}:{item.get('link', '')}:{item.get('title', '')}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _parse_entry(self, entry: Dict, source: FeedSource) -> FeedItem:
        """解析单个 feed entry"""
        # 提取内容
        content = None
        if entry.get('content'):
            content = entry['content'][0].get('value', '')
        elif entry.get('summary_detail'):
            content = entry['summary_detail'].get('value', '')
        
        # 提取分类
        categories = []
        if entry.get('tags'):
            categories = [t.get('term', '') for t in entry['tags'] if t.get('term')]
        
        return FeedItem(
            id=self._generate_id(entry, source.name),
            title=entry.get('title', ''),
            link=entry.get('link', ''),
            description=entry.get('summary', ''),
            content=content,
            author=entry.get('author', ''),
            published=self._parse_datetime(entry.get('published_parsed')),
            updated=self._parse_datetime(entry.get('updated_parsed')),
            categories=categories,
            source=source.name,
            raw_data=dict(entry)
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
        
        logger.info(f"正在获取 RSS 源: {source.name}")
        
        try:
            # 解析 feed
            feed = feedparser.parse(
                source.url,
                request_headers={'User-Agent': 'Mozilla/5.0 NewsAggregator/1.0'}
            )
            
            # 检查解析状态
            if feed.bozo and feed.bozo_exception:
                logger.warning(f"RSS 解析警告 {source.name}: {feed.bozo_exception}")
            
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
            return []
    
    def fetch_all(self) -> List[FeedItem]:
        """获取所有启用源的内容"""
        all_items = []
        
        for source in self._sources:
            items = self.fetch_feed(source)
            all_items.extend(items)
            # 添加短暂延迟，避免请求过快
            time.sleep(0.5)
        
        logger.info(f"共获取 {len(all_items)} 条内容")
        return all_items
    
    def fetch_by_category(self, category: str) -> List[FeedItem]:
        """按分类获取内容"""
        items = []
        for source in self._sources:
            if source.category == category:
                items.extend(self.fetch_feed(source))
                time.sleep(0.5)
        return items


# 预定义的科技类 RSS 源配置
TECH_RSS_SOURCES = [
    # 英文科技媒体
    {"name": "TechCrunch", "url": "https://techcrunch.com/feed/", "category": "tech_news", "language": "en", "priority": 9},
    {"name": "TechCrunch AI", "url": "https://techcrunch.com/category/artificial-intelligence/feed/", "category": "ai", "language": "en", "priority": 9},
    {"name": "The Verge", "url": "https://www.theverge.com/rss/index.xml", "category": "tech_news", "language": "en", "priority": 9},
    {"name": "The Verge AI", "url": "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml", "category": "ai", "language": "en", "priority": 9},
    {"name": "Wired", "url": "https://www.wired.com/feed/rss", "category": "tech_news", "language": "en", "priority": 8},
    {"name": "Wired AI", "url": "https://www.wired.com/feed/tag/ai/latest/rss", "category": "ai", "language": "en", "priority": 8},
    {"name": "Ars Technica", "url": "https://feeds.arstechnica.com/arstechnica/index", "category": "tech_news", "language": "en", "priority": 8},
    {"name": "VentureBeat", "url": "https://venturebeat.com/feed/", "category": "tech_news", "language": "en", "priority": 7},
    {"name": "VentureBeat AI", "url": "https://venturebeat.com/category/ai/feed/", "category": "ai", "language": "en", "priority": 8},
    {"name": "Engadget", "url": "https://www.engadget.com/rss.xml", "category": "tech_news", "language": "en", "priority": 7},
    
    # HN 相关
    {"name": "HN Frontpage", "url": "https://hnrss.org/frontpage", "category": "hacker_news", "language": "en", "priority": 10},
    {"name": "HN Best", "url": "https://hnrss.org/best", "category": "hacker_news", "language": "en", "priority": 9},
    
    # 学术/研究类
    {"name": "arXiv CS.AI", "url": "https://export.arxiv.org/rss/cs.AI", "category": "research", "language": "en", "priority": 8},
    {"name": "arXiv CS.LG", "url": "https://export.arxiv.org/rss/cs.LG", "category": "research", "language": "en", "priority": 8},
    {"name": "ScienceDaily AI", "url": "https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml", "category": "research", "language": "en", "priority": 7},
    
    # Medium 技术博客
    {"name": "Towards Data Science", "url": "https://towardsdatascience.com/feed", "category": "tech_blog", "language": "en", "priority": 7},
    {"name": "Towards AI", "url": "https://pub.towardsai.net/feed", "category": "ai", "language": "en", "priority": 7},
    
    # 中文科技媒体
    {"name": "36氪", "url": "https://36kr.com/feed", "category": "tech_news", "language": "zh", "priority": 8},
    {"name": "少数派", "url": "https://sspai.com/feed", "category": "tech_news", "language": "zh", "priority": 7},
]
```

---

## 6. RSS 源质量评估

### 6.1 评估维度

| 维度 | 说明 | 权重 |
|------|------|------|
| 更新频率 | 每天更新文章数量 | 20% |
| 内容质量 | 原创性、深度、准确性 | 30% |
| 技术相关度 | 与科技主题的相关程度 | 25% |
| 数据完整性 | RSS 字段完整程度 | 15% |
| 稳定性 | 服务可用性和可靠性 | 10% |

### 6.2 推荐优先级

**Tier 1 (最高优先级)**:
- TechCrunch
- The Verge
- Hacker News (via hnrss.org)
- arXiv (学术研究)

**Tier 2 (高优先级)**:
- Wired
- Ars Technica
- VentureBeat

**Tier 3 (中等优先级)**:
- Engadget
- MIT Technology Review
- Medium 技术博客

---

## 7. 总结

本报告详细调研了主流科技类 RSS 信息源，包括:

1. **RSS 标准格式**: RSS 2.0 和 Atom 的结构说明
2. **主流科技媒体**: TechCrunch、The Verge、Wired 等 20+ 个源
3. **中文科技源**: 36氪、少数派等
4. **Python 实现**: 完整的 RSS 采集器代码
5. **质量评估**: RSS 源的评估维度和优先级推荐

建议根据实际需求选择合适的 RSS 源组合，并实现去重、过滤和优先级排序机制。

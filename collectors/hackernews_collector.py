"""
HackerNews 数据采集模块
HackerNews Data Collector Module

支持 Official Firebase API 和 Algolia Search API
"""
import requests
import time
import traceback
import sys
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger


@dataclass
class HNStory:
    """HackerNews 故事数据类"""
    id: int
    title: str
    url: Optional[str] = None
    text: Optional[str] = None
    by: Optional[str] = None
    score: int = 0
    time: Optional[int] = None
    descendants: int = 0
    kids: List[int] = field(default_factory=list)
    type: str = "story"
    source: str = "hackernews"
    
    @property
    def published_at(self) -> Optional[datetime]:
        """获取发布时间"""
        if self.time:
            return datetime.fromtimestamp(self.time)
        return None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        data['published_at'] = self.published_at.isoformat() if self.published_at else None
        return data


@dataclass
class HNComment:
    """HackerNews 评论数据类"""
    id: int
    text: Optional[str] = None
    by: Optional[str] = None
    time: Optional[int] = None
    parent: Optional[int] = None
    kids: List[int] = field(default_factory=list)
    type: str = "comment"
    deleted: bool = False
    dead: bool = False
    
    @property
    def published_at(self) -> Optional[datetime]:
        if self.time:
            return datetime.fromtimestamp(self.time)
        return None


class HackerNewsOfficialAPI:
    """HackerNews Official Firebase API 封装"""
    
    BASE_URL = "https://hacker-news.firebaseio.com/v0"
    
    def __init__(self, timeout: int = 30, max_workers: int = 10):
        """
        初始化
        
        Args:
            timeout: 请求超时时间(秒)
            max_workers: 最大并发工作线程数
        """
        self.timeout = timeout
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "NewsAggregator/1.0"
        })
    
    def _get_json(self, endpoint: str) -> Optional[Dict]:
        """发送 GET 请求获取 JSON"""
        url = f"{self.BASE_URL}/{endpoint}.json"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"请求失败 {endpoint}: {e}")
            return None
        except Exception as e:
            logger.error(f"解析失败 {endpoint}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def get_item(self, item_id: int) -> Optional[Dict]:
        """获取单个 Item"""
        return self._get_json(f"item/{item_id}")
    
    def get_user(self, username: str) -> Optional[Dict]:
        """获取用户信息"""
        return self._get_json(f"user/{username}")
    
    def get_max_item_id(self) -> Optional[int]:
        """获取最大 Item ID"""
        return self._get_json("maxitem")
    
    def get_top_stories(self, limit: int = 30) -> List[int]:
        """获取热门故事 ID 列表"""
        ids = self._get_json("topstories") or []
        return ids[:limit]
    
    def get_new_stories(self, limit: int = 30) -> List[int]:
        """获取最新故事 ID 列表"""
        ids = self._get_json("newstories") or []
        return ids[:limit]
    
    def get_best_stories(self, limit: int = 30) -> List[int]:
        """获取最佳故事 ID 列表"""
        ids = self._get_json("beststories") or []
        return ids[:limit]
    
    def get_ask_stories(self, limit: int = 30) -> List[int]:
        """获取 Ask HN 故事 ID 列表"""
        ids = self._get_json("askstories") or []
        return ids[:limit]
    
    def get_show_stories(self, limit: int = 30) -> List[int]:
        """获取 Show HN 故事 ID 列表"""
        ids = self._get_json("showstories") or []
        return ids[:limit]
    
    def get_job_stories(self, limit: int = 30) -> List[int]:
        """获取招聘信息 ID 列表"""
        ids = self._get_json("jobstories") or []
        return ids[:limit]
    
    def get_updates(self) -> Dict:
        """获取最近更新"""
        return self._get_json("updates") or {"items": [], "profiles": []}
    
    def fetch_stories(self, story_ids: List[int]) -> List[HNStory]:
        """
        批量获取故事详情
        
        Args:
            story_ids: 故事 ID 列表
            
        Returns:
            HNStory 列表
        """
        stories = []
        
        def fetch_one(story_id: int) -> Optional[HNStory]:
            data = self.get_item(story_id)
            if data and data.get('type') in ['story', 'job', 'poll']:
                return HNStory(
                    id=data.get('id', story_id),
                    title=data.get('title', ''),
                    url=data.get('url'),
                    text=data.get('text'),
                    by=data.get('by'),
                    score=data.get('score', 0),
                    time=data.get('time'),
                    descendants=data.get('descendants', 0),
                    kids=data.get('kids', []),
                    type=data.get('type', 'story')
                )
            return None
        
        # 并发获取
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_id = {executor.submit(fetch_one, sid): sid for sid in story_ids}
            
            for future in as_completed(future_to_id):
                try:
                    story = future.result()
                    if story:
                        stories.append(story)
                except Exception as e:
                    logger.error(f"获取故事失败: {e}")
        
        logger.info(f"成功获取 {len(stories)}/{len(story_ids)} 个故事")
        return stories
    
    def fetch_comments(
        self,
        comment_ids: List[int],
        max_depth: int = 3,
        max_comments: int = 100
    ) -> List[HNComment]:
        """
        批量获取评论(递归获取子评论)
        
        Args:
            comment_ids: 评论 ID 列表
            max_depth: 最大递归深度
            max_comments: 最大评论数量
            
        Returns:
            HNComment 列表
        """
        comments = []
        to_fetch = [(cid, 0) for cid in comment_ids]  # (id, depth)
        fetched_ids = set()
        
        while to_fetch and len(comments) < max_comments:
            current_id, depth = to_fetch.pop(0)
            
            if current_id in fetched_ids:
                continue
            fetched_ids.add(current_id)
            
            data = self.get_item(current_id)
            if not data:
                continue
            
            if data.get('type') == 'comment' and not data.get('deleted'):
                comment = HNComment(
                    id=data.get('id', current_id),
                    text=data.get('text'),
                    by=data.get('by'),
                    time=data.get('time'),
                    parent=data.get('parent'),
                    kids=data.get('kids', []),
                    deleted=data.get('deleted', False),
                    dead=data.get('dead', False)
                )
                comments.append(comment)
                
                # 添加子评论到待获取列表
                if depth < max_depth and comment.kids:
                    for kid_id in comment.kids[:10]:  # 每条评论最多10个子评论
                        to_fetch.append((kid_id, depth + 1))
        
        logger.info(f"获取 {len(comments)} 条评论")
        return comments


class AlgoliaHNAPI:
    """Algolia HN Search API 封装"""
    
    BASE_URL = "https://hn.algolia.com/api/v1"
    
    def __init__(self, timeout: int = 30):
        """
        初始化
        
        Args:
            timeout: 请求超时时间(秒)
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "NewsAggregator/1.0"
        })
    
    def search(
        self,
        query: str,
        tags: Optional[str] = None,
        page: int = 0,
        hits_per_page: int = 20,
        numeric_filters: Optional[str] = None
    ) -> Dict:
        """
        搜索 HN 内容
        
        Args:
            query: 搜索关键词
            tags: 标签过滤(如 story, comment, author_xxx)
            page: 页码(从0开始)
            hits_per_page: 每页数量
            numeric_filters: 数值过滤(如 points>100)
            
        Returns:
            搜索结果字典
        """
        params = {
            "query": query,
            "page": page,
            "hitsPerPage": hits_per_page
        }
        if tags:
            params["tags"] = tags
        if numeric_filters:
            params["numericFilters"] = numeric_filters
        
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/search",
                params=params,
                timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Algolia 搜索失败: {e}")
            logger.error(traceback.format_exc())
            return {"hits": [], "nbHits": 0}
    
    def search_by_date(
        self,
        query: str = "",
        tags: Optional[str] = None,
        page: int = 0,
        hits_per_page: int = 20
    ) -> Dict:
        """按日期搜索(最新优先)"""
        params = {
            "page": page,
            "hitsPerPage": hits_per_page
        }
        if query:
            params["query"] = query
        if tags:
            params["tags"] = tags
        
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/search_by_date",
                params=params,
                timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Algolia 日期搜索失败: {e}")
            return {"hits": [], "nbHits": 0}
    
    def get_item(self, item_id: int) -> Optional[Dict]:
        """获取 Item 详情(包含完整评论树)"""
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/items/{item_id}",
                timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"获取 Item {item_id} 失败: {e}")
            return None
    
    def get_user(self, username: str) -> Optional[Dict]:
        """获取用户信息"""
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/users/{username}",
                timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"获取用户 {username} 失败: {e}")
            return None
    
    def get_story_comments(
        self,
        story_id: int,
        max_comments: int = 100
    ) -> List[Dict]:
        """
        获取故事的所有评论
        
        Args:
            story_id: 故事 ID
            max_comments: 最大评论数
            
        Returns:
            评论列表
        """
        hits_per_page = min(max_comments, 1000)
        result = self.search_by_date(
            tags=f"comment,story_{story_id}",
            hits_per_page=hits_per_page
        )
        return result.get("hits", [])[:max_comments]


class HackerNewsCollector:
    """HackerNews 统一采集器"""
    
    def __init__(
        self,
        timeout: int = 30,
        max_workers: int = 10,
        use_algolia: bool = True
    ):
        """
        初始化
        
        Args:
            timeout: 请求超时时间
            max_workers: 并发工作线程数
            use_algolia: 是否使用 Algolia API
        """
        self.official_api = HackerNewsOfficialAPI(timeout, max_workers)
        self.algolia_api = AlgoliaHNAPI(timeout) if use_algolia else None
    
    def fetch_top_stories(
        self,
        limit: int = 30,
        include_comments: bool = False,
        max_comments_per_story: int = 50
    ) -> List[HNStory]:
        """
        获取热门故事
        
        Args:
            limit: 故事数量
            include_comments: 是否包含评论
            max_comments_per_story: 每个故事最大评论数
            
        Returns:
            HNStory 列表
        """
        logger.info(f"获取 HN 热门故事 (limit={limit})")
        
        # 获取故事 ID
        story_ids = self.official_api.get_top_stories(limit)
        
        # 获取故事详情
        stories = self.official_api.fetch_stories(story_ids)
        
        # 如果需要评论，使用 Algolia API 获取
        if include_comments and self.algolia_api:
            for story in stories:
                if story.descendants > 0:
                    comments = self.algolia_api.get_story_comments(
                        story.id,
                        max_comments_per_story
                    )
                    story.comments = comments
        
        return stories
    
    def search_stories(
        self,
        query: str,
        limit: int = 30,
        min_points: int = 0
    ) -> List[Dict]:
        """
        搜索故事
        
        Args:
            query: 搜索关键词
            limit: 结果数量
            min_points: 最低分数
            
        Returns:
            搜索结果列表
        """
        if not self.algolia_api:
            logger.warning("Algolia API 未启用，无法搜索")
            return []
        
        numeric_filters = f"points>{min_points}" if min_points > 0 else None
        
        result = self.algolia_api.search(
            query=query,
            tags="story",
            hits_per_page=limit,
            numeric_filters=numeric_filters
        )
        
        return result.get("hits", [])
    
    def fetch_recent_stories(
        self,
        hours: int = 24,
        limit: int = 100
    ) -> List[Dict]:
        """
        获取最近的故事
        
        Args:
            hours: 时间范围(小时)
            limit: 结果数量
            
        Returns:
            故事列表
        """
        if not self.algolia_api:
            # 使用 Official API
            story_ids = self.official_api.get_new_stories(limit)
            return [s.to_dict() for s in self.official_api.fetch_stories(story_ids)]
        
        # 使用 Algolia API
        import time
        timestamp = int(time.time()) - hours * 3600
        
        result = self.algolia_api.search(
            query="",
            tags="story",
            hits_per_page=limit,
            numeric_filters=f"created_at_i>{timestamp}"
        )
        
        return result.get("hits", [])


if __name__ == "__main__":
    # 配置 loguru
    logger.remove()
    logger.add(
        sys.stderr,
        format="{time:HH:mm:ss} | {level: <8} | {message}",
        level="INFO"
    )
    
    # 测试采集器
    collector = HackerNewsCollector()
    
    # 获取热门故事
    print("\n=== 热门故事 ===")
    stories = collector.fetch_top_stories(limit=5)
    for story in stories:
        print(f"[{story.score}分] {story.title}")
        print(f"  URL: {story.url}")
        print(f"  评论数: {story.descendants}")
        print()
    
    # 搜索故事
    print("\n=== 搜索 'AI' ===")
    results = collector.search_stories("AI", limit=5, min_points=50)
    for hit in results:
        print(f"[{hit.get('points', 0)}分] {hit.get('title', '')}")

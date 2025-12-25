# HackerNews API 详细调研报告

## 1. 概述

HackerNews 提供两个主要的 API 系统：
1. **Official Firebase API** - Y Combinator 官方提供
2. **Algolia Search API** - 第三方搜索增强 API

---

## 2. Official HackerNews Firebase API

### 2.1 基础信息
- **Base URL**: `https://hacker-news.firebaseio.com/v0/`
- **认证**: 无需认证，完全公开
- **限流**: 目前无速率限制
- **数据格式**: JSON
- **协议**: HTTPS

### 2.2 核心端点

#### 2.2.1 Item 端点 (Stories, Comments, Jobs, Polls)
```
GET /v0/item/{id}.json
```

**Item 类型**:
- `story`: 故事/文章
- `comment`: 评论
- `job`: 招聘信息
- `poll`: 投票
- `pollopt`: 投票选项

**Item 字段说明**:
| 字段 | 类型 | 说明 | 必需 |
|------|------|------|------|
| id | int | 唯一标识符 | ✓ |
| type | string | item类型 | |
| by | string | 作者用户名 | |
| time | int | Unix时间戳 | |
| text | string | 内容(HTML) | |
| url | string | 文章URL | |
| title | string | 标题(HTML) | |
| score | int | 得分/投票数 | |
| kids | array[int] | 子评论ID列表 | |
| parent | int | 父级item ID | |
| descendants | int | 评论总数 | |
| deleted | bool | 是否已删除 | |
| dead | bool | 是否已死亡 | |

**示例请求 - Story**:
```bash
curl "https://hacker-news.firebaseio.com/v0/item/8863.json?print=pretty"
```

**响应示例**:
```json
{
  "by": "dhouston",
  "descendants": 71,
  "id": 8863,
  "kids": [8952, 9224, 8917, ...],
  "score": 111,
  "time": 1175714200,
  "title": "My YC app: Dropbox - Throw away your USB drive",
  "type": "story",
  "url": "http://www.getdropbox.com/u/2/screencast.html"
}
```

**示例请求 - Comment**:
```bash
curl "https://hacker-news.firebaseio.com/v0/item/2921983.json?print=pretty"
```

**响应示例**:
```json
{
  "by": "norvig",
  "id": 2921983,
  "kids": [2922097, 2922429, ...],
  "parent": 2921506,
  "text": "Aw shucks, guys...",
  "time": 1314211127,
  "type": "comment"
}
```

#### 2.2.2 User 端点
```
GET /v0/user/{username}.json
```

**User 字段**:
| 字段 | 类型 | 说明 |
|------|------|------|
| id | string | 用户名(大小写敏感) |
| created | int | 创建时间(Unix) |
| karma | int | karma值 |
| about | string | 个人简介(HTML) |
| submitted | array[int] | 提交的item ID列表 |

**示例**:
```bash
curl "https://hacker-news.firebaseio.com/v0/user/jl.json?print=pretty"
```

#### 2.2.3 Stories 列表端点

| 端点 | 说明 | 数量限制 |
|------|------|----------|
| `/v0/topstories.json` | 热门故事 | 最多500 |
| `/v0/newstories.json` | 最新故事 | 最多500 |
| `/v0/beststories.json` | 最佳故事 | 最多500 |
| `/v0/askstories.json` | Ask HN | 最多200 |
| `/v0/showstories.json` | Show HN | 最多200 |
| `/v0/jobstories.json` | 招聘信息 | 最多200 |

**返回格式**: ID数组
```json
[9129911, 9129199, 9127761, ...]
```

#### 2.2.4 实时数据端点

**Max Item ID**:
```
GET /v0/maxitem.json
```
返回当前最大item ID，可用于向后遍历发现所有items。

**Updates (变更通知)**:
```
GET /v0/updates.json
```
返回最近变更的items和profiles:
```json
{
  "items": [8423305, 8420805, ...],
  "profiles": ["thefox", "mdda", ...]
}
```

### 2.3 API 使用注意事项

1. **评论树遍历**: 
   - API 不返回完整评论树
   - 需要递归获取每个评论的 `kids` 字段
   - 这是设计限制，非bug

2. **性能优化建议**:
   - 使用本地缓存减少请求
   - 批量请求时考虑并发限制
   - 使用 Firebase 实时订阅功能监听变化

3. **数据处理**:
   - HTML 内容需要转义处理
   - Unix时间戳需要转换

---

## 3. Algolia HackerNews Search API

### 3.1 基础信息
- **Base URL**: `https://hn.algolia.com/api/v1/`
- **认证**: 无需认证
- **特点**: 提供搜索功能，可获取完整评论树

### 3.2 核心端点

#### 3.2.1 搜索端点

**按相关性搜索**:
```
GET /search?query={query}
```

**按日期搜索(最新优先)**:
```
GET /search_by_date?query={query}
```

#### 3.2.2 搜索参数

| 参数 | 说明 | 示例 |
|------|------|------|
| query | 搜索关键词 | `query=react` |
| tags | 过滤标签 | `tags=story`, `tags=comment` |
| numericFilters | 数值过滤 | `numericFilters=points>100` |
| page | 分页 | `page=0` |
| hitsPerPage | 每页数量 | `hitsPerPage=50` (最大1000) |

**Tags 类型**:
- `story`: 故事
- `comment`: 评论
- `poll`: 投票
- `pollopt`: 投票选项
- `show_hn`: Show HN
- `ask_hn`: Ask HN
- `front_page`: 首页
- `author_{username}`: 特定作者
- `story_{id}`: 特定故事的评论

**Tags 组合逻辑**:
- 默认 AND 逻辑
- 括号内为 OR 逻辑
- 示例: `tags=author_pg,(story,poll)` = author=pg AND (type=story OR type=poll)

#### 3.2.3 Item 详情端点
```
GET /items/{id}
```
返回完整的嵌套评论树结构。

#### 3.2.4 User 详情端点
```
GET /users/{username}
```

### 3.3 响应结构

```json
{
  "hits": [
    {
      "objectID": "123456",
      "title": "Article Title",
      "url": "https://example.com",
      "author": "username",
      "points": 150,
      "num_comments": 45,
      "created_at": "2025-01-15T10:30:00.000Z",
      "created_at_i": 1736934600,
      "_tags": ["story", "author_username"],
      "_highlightResult": {...}
    }
  ],
  "nbHits": 1000,
  "page": 0,
  "nbPages": 50,
  "hitsPerPage": 20,
  "query": "search term"
}
```

### 3.4 高级用法示例

**获取特定故事的所有评论(按时间排序)**:
```
GET /search_by_date?tags=comment,story_35111646&hitsPerPage=1000
```

**搜索过去一周的热门AI文章**:
```
GET /search?query=artificial+intelligence&tags=story&numericFilters=points>100,created_at_i>1736329200
```

---

## 4. API 对比与选择建议

| 特性 | Official API | Algolia API |
|------|-------------|-------------|
| 搜索功能 | ❌ | ✓ |
| 完整评论树 | ❌ (需递归) | ✓ |
| 实时更新 | ✓ (Firebase) | ❌ |
| 数据完整性 | ✓ 官方数据 | ✓ 近实时同步 |
| 历史数据 | ✓ | ✓ |
| 评论排序 | ✓ 原始排序 | ❌ 需手动排序 |

**推荐策略**:
1. 使用 **Algolia API** 进行搜索和批量获取评论
2. 使用 **Official API** 获取实时更新和确保数据准确性
3. 组合使用两个 API 以获得最佳体验

---

## 5. Python 实现示例

```python
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger

@dataclass
class HNItem:
    """HackerNews Item 数据类"""
    id: int
    type: str
    title: Optional[str] = None
    url: Optional[str] = None
    text: Optional[str] = None
    by: Optional[str] = None
    time: Optional[int] = None
    score: Optional[int] = None
    descendants: Optional[int] = None
    kids: Optional[List[int]] = None

class HackerNewsAPI:
    """HackerNews Official API 封装"""
    
    BASE_URL = "https://hacker-news.firebaseio.com/v0"
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
    
    def get_item(self, item_id: int) -> Optional[Dict]:
        """获取单个 item"""
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/item/{item_id}.json",
                timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"获取 item {item_id} 失败: {e}")
            return None
    
    def get_top_stories(self, limit: int = 30) -> List[int]:
        """获取热门故事ID列表"""
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/topstories.json",
                timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()[:limit]
        except Exception as e:
            logger.error(f"获取热门故事失败: {e}")
            return []

class AlgoliaHNAPI:
    """Algolia HN Search API 封装"""
    
    BASE_URL = "https://hn.algolia.com/api/v1"
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
    
    def search(
        self,
        query: str,
        tags: Optional[str] = None,
        page: int = 0,
        hits_per_page: int = 20
    ) -> Dict:
        """搜索 HN 内容"""
        params = {
            "query": query,
            "page": page,
            "hitsPerPage": hits_per_page
        }
        if tags:
            params["tags"] = tags
        
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/search",
                params=params,
                timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return {"hits": []}
    
    def get_item_with_comments(self, item_id: int) -> Optional[Dict]:
        """获取 item 及完整评论树"""
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/items/{item_id}",
                timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"获取 item {item_id} 失败: {e}")
            return None
```

---

## 6. 总结

HackerNews API 系统提供了丰富的数据访问接口：

1. **Official Firebase API** 适合:
   - 获取实时更新
   - 确保数据准确性
   - 构建实时应用

2. **Algolia Search API** 适合:
   - 关键词搜索
   - 批量获取评论
   - 数据分析和挖掘

建议根据具体需求组合使用两个 API，以获得最佳的数据获取效果。

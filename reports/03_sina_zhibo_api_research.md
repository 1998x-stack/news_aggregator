# 新浪财经直播 (Zhibo) API 详细调研报告

## 1. 概述

新浪财经直播 API 提供 7x24 小时财经快讯和实时直播内容。本报告基于对 `zhibo.sina.com.cn` API 的逆向分析。

---

## 2. API 基础信息

### 2.1 基础端点

```
https://zhibo.sina.com.cn/api/zhibo/feed
```

### 2.2 请求方式
- **方法**: GET
- **格式**: JSONP (支持 callback 参数)
- **编码**: UTF-8

---

## 3. 核心 API 端点

### 3.1 Feed 获取接口

```
GET https://zhibo.sina.com.cn/api/zhibo/feed
```

**请求参数**:

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| callback | string | 否 | JSONP 回调函数名 |
| page | int | 是 | 页码，从 1 开始 |
| page_size | int | 是 | 每页数量 |
| zhibo_id | int | 是 | 直播间/频道 ID |
| tag_id | int | 否 | 标签 ID，0 表示全部 |
| dire | string | 否 | 方向，'f' 表示向前 |
| dpc | int | 否 | 数据处理参数 |

**示例请求**:
```
https://zhibo.sina.com.cn/api/zhibo/feed?page=1&page_size=20&zhibo_id=152&tag_id=0&dire=f&dpc=1
```

### 3.2 响应结构

```json
{
  "result": {
    "status": {
      "code": 0,
      "msg": "OK"
    },
    "timestamp": "Thu Dec 25 22:46:48 +0800 2025",
    "数据": {
      "feed": {
        "列表": [
          {
            "id": 4576925,
            "zhibo_id": 152,
            "类型": 0,
            "rich_text": "【新闻标题】新闻内容...",
            "multimedia": "",
            "commentid": "live:finance-152-4576925:0",
            "compere_id": 0,
            "creator": "user@staff.sina.com",
            "mender": "user@staff.sina.com",
            "create_time": "2025-12-25 22:43:50",
            "update_time": "2025-12-25 22:43:57",
            "is_need_check": "0",
            "check_status": "1",
            "is_delete": 0,
            "top_value": 0,
            "is_focus": 0,
            "ext": "{\"stocks\":[...],\"docurl\":\"...\",\"docid\":\"...\"}",
            "tag": [{"id": "1", "name": "宏观"}],
            "like_nums": 0,
            "docurl": "https://finance.sina.cn/7x24/..."
          }
        ],
        "page_info": {
          "totalPage": 900,
          "pageSize": 1,
          "page": 1,
          "totalNum": 900
        },
        "max_id": 4576925,
        "min_id": 4576925
      }
    }
  }
}
```

---

## 4. zhibo_id 频道分类

根据分析，zhibo_id 代表不同的直播频道/内容类型:

### 4.1 已知 zhibo_id 列表

| zhibo_id | 频道名称 | 说明 |
|----------|----------|------|
| 152 | 7x24 财经直播 | 24小时财经快讯，最主要的财经新闻源 |
| 153 | 期货直播 | 期货市场相关资讯 |
| 155 | 外汇直播 | 外汇市场相关资讯 |
| 156 | 美股直播 | 美股市场相关资讯 |
| 157 | 港股直播 | 港股市场相关资讯 |
| 158 | 债券直播 | 债券市场相关资讯 |
| 159 | 基金直播 | 基金相关资讯 |
| 160 | 保险直播 | 保险相关资讯 |
| 161 | 银行直播 | 银行相关资讯 |

### 4.2 tag_id 标签分类

| tag_id | 标签名称 | 说明 |
|--------|----------|------|
| 0 | 全部 | 所有标签 |
| 1 | 宏观 | 宏观经济 |
| 2 | 公司 | 公司新闻 |
| 3 | 行业 | 行业动态 |
| 4 | 市场 | 市场资讯 |
| 5 | 政策 | 政策法规 |

---

## 5. 数据字段详解

### 5.1 Feed Item 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| id | int | 唯一标识符 |
| zhibo_id | int | 所属频道 ID |
| 类型 | int | 内容类型 (0=文字, 1=图片, 2=视频) |
| rich_text | string | 富文本内容 (Unicode 编码) |
| multimedia | string | 多媒体附件 |
| commentid | string | 评论 ID |
| compere_id | int | 主播 ID |
| creator | string | 创建者邮箱 |
| create_time | string | 创建时间 |
| update_time | string | 更新时间 |
| is_delete | int | 是否删除 |
| top_value | int | 置顶权重 |
| is_focus | int | 是否焦点 |
| ext | string | 扩展信息 (JSON 字符串) |
| tag | array | 标签列表 |
| like_nums | int | 点赞数 |
| docurl | string | 详情页 URL |

### 5.2 扩展字段 (ext) 结构

```json
{
  "stocks": [
    {
      "market": "cn",
      "symbol": "si931775",
      "key": "房地产"
    }
  ],
  "needPushWB": false,
  "needCMSLink": true,
  "docurl": "https://finance.sina.com.cn/...",
  "docid": "nhcztfz3112144"
}
```

---

## 6. 其他新浪财经 API

### 6.1 股票实时数据 API

**端点**: `https://hq.sinajs.cn/list={codes}`

**注意**: 2022年后需要添加 Referer 头

```python
headers = {
    "Referer": "https://finance.sina.com.cn"
}
```

**示例**:
```bash
curl -H "Referer: https://finance.sina.com.cn" \
     "https://hq.sinajs.cn/list=sh601006"
```

### 6.2 K线数据 API

**端点**:
```
https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData
```

**参数**:
- `symbol`: 股票代码 (如 sh600519)
- `scale`: 时间周期 (5/15/30/60/240)
- `ma`: 均线周期
- `datalen`: 数据长度

---

## 7. Python 实现

```python
"""
新浪财经直播 API 采集模块
"""
import requests
import json
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger


@dataclass
class SinaZhiboItem:
    """新浪财经直播条目"""
    id: int
    zhibo_id: int
    content: str
    create_time: datetime
    update_time: datetime
    tags: List[str] = field(default_factory=list)
    doc_url: Optional[str] = None
    doc_id: Optional[str] = None
    stocks: List[Dict] = field(default_factory=list)
    is_focus: bool = False
    like_nums: int = 0


class SinaZhiboCollector:
    """新浪财经直播采集器"""
    
    BASE_URL = "https://zhibo.sina.com.cn/api/zhibo/feed"
    
    # 频道 ID 映射
    CHANNELS = {
        152: "7x24财经",
        153: "期货",
        155: "外汇",
        156: "美股",
        157: "港股",
        158: "债券",
        159: "基金",
        160: "保险",
        161: "银行",
    }
    
    # 标签 ID 映射
    TAGS = {
        0: "全部",
        1: "宏观",
        2: "公司",
        3: "行业",
        4: "市场",
        5: "政策",
    }
    
    def __init__(self, timeout: int = 30):
        """
        初始化采集器
        
        Args:
            timeout: 请求超时时间
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://finance.sina.com.cn"
        })
    
    def _parse_jsonp(self, jsonp_str: str) -> Optional[Dict]:
        """解析 JSONP 响应"""
        try:
            # 提取 JSON 部分
            match = re.search(r'\{.*\}', jsonp_str, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.error(f"解析 JSONP 失败: {e}")
        return None
    
    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """解析时间字符串"""
        try:
            return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None
    
    def _decode_unicode(self, text: str) -> str:
        """解码 Unicode 转义字符"""
        try:
            return text.encode().decode('unicode_escape')
        except Exception:
            return text
    
    def _parse_item(self, item: Dict) -> SinaZhiboItem:
        """解析单个条目"""
        # 解析扩展字段
        ext = {}
        if item.get('ext'):
            try:
                ext = json.loads(item['ext'])
            except Exception:
                pass
        
        # 解析标签
        tags = []
        if item.get('tag'):
            tags = [t.get('name', '') for t in item['tag'] if t.get('name')]
        
        # 解码富文本内容
        content = self._decode_unicode(item.get('rich_text', ''))
        
        return SinaZhiboItem(
            id=item.get('id', 0),
            zhibo_id=item.get('zhibo_id', 0),
            content=content,
            create_time=self._parse_datetime(item.get('create_time', '')),
            update_time=self._parse_datetime(item.get('update_time', '')),
            tags=tags,
            doc_url=item.get('docurl') or ext.get('docurl'),
            doc_id=ext.get('docid'),
            stocks=ext.get('stocks', []),
            is_focus=bool(item.get('is_focus', 0)),
            like_nums=item.get('like_nums', 0)
        )
    
    def fetch_feed(
        self,
        zhibo_id: int = 152,
        tag_id: int = 0,
        page: int = 1,
        page_size: int = 20
    ) -> List[SinaZhiboItem]:
        """
        获取直播流
        
        Args:
            zhibo_id: 频道 ID，默认 152 (7x24财经)
            tag_id: 标签 ID，默认 0 (全部)
            page: 页码
            page_size: 每页数量
            
        Returns:
            SinaZhiboItem 列表
        """
        params = {
            "page": page,
            "page_size": page_size,
            "zhibo_id": zhibo_id,
            "tag_id": tag_id,
            "dire": "f",
            "dpc": 1
        }
        
        channel_name = self.CHANNELS.get(zhibo_id, f"频道{zhibo_id}")
        logger.info(f"获取新浪财经直播: {channel_name}, 页码: {page}")
        
        try:
            resp = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=self.timeout
            )
            resp.raise_for_status()
            
            # 解析响应
            data = self._parse_jsonp(resp.text) or resp.json()
            
            # 提取条目
            items = []
            feed_data = data.get('result', {}).get('数据', {}).get('feed', {})
            item_list = feed_data.get('列表', [])
            
            for item in item_list:
                try:
                    parsed = self._parse_item(item)
                    items.append(parsed)
                except Exception as e:
                    logger.error(f"解析条目失败: {e}")
                    continue
            
            logger.info(f"获取 {len(items)} 条内容")
            return items
            
        except Exception as e:
            logger.error(f"获取直播流失败: {e}")
            return []
    
    def fetch_multiple_pages(
        self,
        zhibo_id: int = 152,
        tag_id: int = 0,
        pages: int = 5,
        page_size: int = 20
    ) -> List[SinaZhiboItem]:
        """获取多页内容"""
        all_items = []
        for page in range(1, pages + 1):
            items = self.fetch_feed(zhibo_id, tag_id, page, page_size)
            all_items.extend(items)
            if not items:
                break
        return all_items
    
    def fetch_all_channels(
        self,
        pages_per_channel: int = 2
    ) -> Dict[str, List[SinaZhiboItem]]:
        """获取所有频道内容"""
        result = {}
        for zhibo_id, name in self.CHANNELS.items():
            items = self.fetch_multiple_pages(zhibo_id, pages=pages_per_channel)
            result[name] = items
        return result


# 使用示例
if __name__ == "__main__":
    collector = SinaZhiboCollector()
    
    # 获取 7x24 财经快讯
    items = collector.fetch_feed(zhibo_id=152, page_size=10)
    for item in items:
        print(f"[{item.create_time}] {item.content[:100]}...")
```

---

## 8. API 调用注意事项

### 8.1 请求限制
- 建议请求间隔 >= 1 秒
- 单次请求 page_size 建议 <= 50
- 避免并发大量请求

### 8.2 数据处理
- rich_text 字段包含 Unicode 转义，需要解码
- ext 字段是 JSON 字符串，需要二次解析
- 时间字段格式为 "YYYY-MM-DD HH:MM:SS"

### 8.3 JSONP 处理
- 如果使用 callback 参数，返回 JSONP 格式
- 不使用 callback 参数时返回纯 JSON

---

## 9. 总结

新浪财经直播 API 提供了丰富的财经快讯数据：

1. **主要功能**: 7x24 小时财经快讯实时获取
2. **频道覆盖**: 财经、期货、外汇、美股、港股等
3. **数据结构**: 包含标题、内容、标签、关联股票等
4. **使用场景**: 财经新闻聚合、市场情绪分析、量化交易信号

建议将此 API 与其他财经数据源结合使用，构建更完整的财经信息系统。

"""
新浪财经直播数据采集模块
Sina Finance Zhibo Data Collector Module

支持 7x24 财经快讯和多个财经频道
"""
import requests
import json
import re
import time
import traceback
import sys
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from loguru import logger


@dataclass
class ZhiboItem:
    """新浪财经直播条目数据类"""
    id: int
    zhibo_id: int
    content: str
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    doc_url: Optional[str] = None
    doc_id: Optional[str] = None
    stocks: List[Dict] = field(default_factory=list)
    is_focus: bool = False
    like_nums: int = 0
    source: str = "sina_zhibo"
    channel_name: str = ""
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        data['create_time'] = self.create_time.isoformat() if self.create_time else None
        data['update_time'] = self.update_time.isoformat() if self.update_time else None
        return data


@dataclass
class ZhiboChannel:
    """新浪财经直播频道配置"""
    zhibo_id: int
    name: str
    description: str = ""
    enabled: bool = True
    priority: int = 5


class SinaZhiboCollector:
    """新浪财经直播采集器"""
    
    BASE_URL = "https://zhibo.sina.com.cn/api/zhibo/feed"
    
    # 频道配置
    CHANNELS = [
        ZhiboChannel(152, "7x24财经", "24小时财经快讯", True, 10),
        ZhiboChannel(153, "期货直播", "期货市场资讯", True, 7),
        ZhiboChannel(155, "外汇直播", "外汇市场资讯", True, 7),
        ZhiboChannel(156, "美股直播", "美股市场资讯", True, 8),
        ZhiboChannel(157, "港股直播", "港股市场资讯", True, 8),
        ZhiboChannel(158, "债券直播", "债券市场资讯", True, 6),
        ZhiboChannel(159, "基金直播", "基金相关资讯", True, 6),
        ZhiboChannel(160, "保险直播", "保险相关资讯", False, 4),
        ZhiboChannel(161, "银行直播", "银行相关资讯", False, 4),
    ]
    
    # 标签配置
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
            timeout: 请求超时时间(秒)
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://finance.sina.com.cn",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"
        })
        self._channel_map = {c.zhibo_id: c for c in self.CHANNELS}
    
    def _parse_jsonp(self, jsonp_str: str) -> Optional[Dict]:
        """解析 JSONP 响应"""
        try:
            # 移除 try-catch 包装
            jsonp_str = re.sub(r'^try\s*\{', '', jsonp_str)
            jsonp_str = re.sub(r'\}\s*catch\s*\([^)]*\)\s*\{\s*\}[\s;]*$', '', jsonp_str)
            
            # 提取 JSON 部分
            match = re.search(r'jQuery[^(]*\((\{.*\})\)', jsonp_str, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            
            # 尝试直接解析
            match = re.search(r'\{.*\}', jsonp_str, re.DOTALL)
            if match:
                return json.loads(match.group())
                
        except Exception as e:
            logger.error(f"解析 JSONP 失败: {e}")
        return None
    
    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """解析时间字符串"""
        if not dt_str:
            return None
        try:
            return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            try:
                return datetime.strptime(dt_str, "%Y-%m-%d")
            except Exception:
                return None
    
    def _fix_mojibake(self, text: str) -> str:
        """
        修复乱码(Mojibake): UTF-8内容被错误地用Latin-1解析
        
        例如: "ãæå±±" -> "昆山"
        
        Args:
            text: 可能包含乱码的文本
            
        Returns:
            修复后的文本
        """
        if not text:
            return ""
        
        # 检测是否是乱码 (包含典型的乱码字符)
        mojibake_indicators = ['ã', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 
                               'î', 'ï', 'ð', 'ñ', 'ò', 'ó', 'ô', 'õ', 'ö', '÷',
                               'ø', 'ù', 'ú', 'û', 'ü', 'ý', 'þ', 'ÿ', 'Ã', 'Â']
        
        # 如果包含乱码特征字符，尝试修复
        if any(char in text for char in mojibake_indicators):
            try:
                # UTF-8被错误解析为Latin-1，需要反向转换
                fixed = text.encode('latin-1').decode('utf-8')
                # 验证修复后的文本是否合理（包含中文）
                if any('\u4e00' <= c <= '\u9fff' for c in fixed):
                    return fixed
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass
            
            try:
                # 尝试 Windows-1252 编码
                fixed = text.encode('cp1252').decode('utf-8')
                if any('\u4e00' <= c <= '\u9fff' for c in fixed):
                    return fixed
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass
        
        return text
    
    def _decode_unicode(self, text: str) -> str:
        """解码 Unicode 转义字符并修复乱码"""
        if not text:
            return ""
        
        # 首先尝试修复乱码
        text = self._fix_mojibake(text)
        
        # 处理 \uXXXX 转义序列
        if '\\u' in text:
            try:
                # 使用 unicode_escape 解码
                text = text.encode('utf-8').decode('unicode_escape')
            except Exception:
                try:
                    text = text.encode('latin-1').decode('unicode_escape')
                except Exception:
                    pass
        
        return text
    
    def _clean_content(self, text: str) -> str:
        """清理内容文本"""
        if not text:
            return ""
        
        # 解码 Unicode
        text = self._decode_unicode(text)
        
        # 移除 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _parse_item(self, item: Dict, channel: ZhiboChannel) -> ZhiboItem:
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
            for t in item['tag']:
                if isinstance(t, dict) and t.get('name'):
                    tags.append(t['name'])
                elif isinstance(t, str):
                    tags.append(t)
        
        # 清理内容
        content = self._clean_content(item.get('rich_text', ''))
        
        return ZhiboItem(
            id=item.get('id', 0),
            zhibo_id=item.get('zhibo_id', channel.zhibo_id),
            content=content,
            create_time=self._parse_datetime(item.get('create_time', '')),
            update_time=self._parse_datetime(item.get('update_time', '')),
            tags=tags,
            doc_url=item.get('docurl') or ext.get('docurl'),
            doc_id=ext.get('docid'),
            stocks=ext.get('stocks', []),
            is_focus=bool(item.get('is_focus', 0)),
            like_nums=item.get('like_nums', 0),
            channel_name=channel.name
        )
    
    def fetch_feed(
        self,
        zhibo_id: int = 152,
        tag_id: int = 0,
        page: int = 1,
        page_size: int = 20
    ) -> List[ZhiboItem]:
        """
        获取直播流
        
        Args:
            zhibo_id: 频道 ID，默认 152 (7x24财经)
            tag_id: 标签 ID，默认 0 (全部)
            page: 页码
            page_size: 每页数量
            
        Returns:
            ZhiboItem 列表
        """
        channel = self._channel_map.get(zhibo_id)
        if not channel:
            channel = ZhiboChannel(zhibo_id, f"频道{zhibo_id}")
        
        logger.info(f"获取新浪财经直播: {channel.name}, 页码: {page}")
        
        params = {
            "page": page,
            "page_size": page_size,
            "zhibo_id": zhibo_id,
            "tag_id": tag_id,
            "dire": "f",
            "dpc": 1
        }
        
        try:
            resp = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=self.timeout
            )
            resp.raise_for_status()
            
            # 强制使用 UTF-8 编码（新浪API返回UTF-8）
            resp.encoding = 'utf-8'
            response_text = resp.text
            
            # 如果仍然检测到乱码，尝试修复
            response_text = self._fix_mojibake(response_text)
            
            # 尝试解析响应
            data = None
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                data = self._parse_jsonp(response_text)
            
            if not data:
                logger.error("无法解析响应数据")
                return []
            
            # 提取条目
            items = []
            
            # 尝试不同的数据路径
            feed_data = (
                data.get('result', {}).get('数据', {}).get('feed', {}) or
                data.get('result', {}).get('data', {}).get('feed', {}) or
                data.get('data', {}).get('feed', {}) or
                {}
            )
            
            item_list = (
                feed_data.get('列表', []) or
                feed_data.get('list', []) or
                []
            )
            
            for item in item_list:
                try:
                    parsed = self._parse_item(item, channel)
                    if parsed.content:  # 只保留有内容的条目
                        items.append(parsed)
                except Exception as e:
                    logger.error(f"解析条目失败: {e}")
                    continue
            
            logger.info(f"从 {channel.name} 获取 {len(items)} 条内容")
            return items
            
        except requests.RequestException as e:
            logger.error(f"请求失败: {e}")
            return []
        except Exception as e:
            logger.error(f"获取直播流失败: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def fetch_multiple_pages(
        self,
        zhibo_id: int = 152,
        tag_id: int = 0,
        pages: int = 5,
        page_size: int = 20,
        delay: float = 0.5
    ) -> List[ZhiboItem]:
        """
        获取多页内容
        
        Args:
            zhibo_id: 频道 ID
            tag_id: 标签 ID
            pages: 页数
            page_size: 每页数量
            delay: 请求间隔(秒)
            
        Returns:
            ZhiboItem 列表
        """
        all_items = []
        
        for page in range(1, pages + 1):
            items = self.fetch_feed(zhibo_id, tag_id, page, page_size)
            all_items.extend(items)
            
            if not items:
                break
            
            if page < pages:
                time.sleep(delay)
        
        return all_items
    
    def fetch_all_channels(
        self,
        pages_per_channel: int = 2,
        only_enabled: bool = True
    ) -> Dict[str, List[ZhiboItem]]:
        """
        获取所有频道内容
        
        Args:
            pages_per_channel: 每个频道获取的页数
            only_enabled: 是否只获取启用的频道
            
        Returns:
            {频道名: ZhiboItem列表}
        """
        result = {}
        
        channels = self.CHANNELS
        if only_enabled:
            channels = [c for c in channels if c.enabled]
        
        for channel in channels:
            logger.info(f"获取频道: {channel.name}")
            items = self.fetch_multiple_pages(
                zhibo_id=channel.zhibo_id,
                pages=pages_per_channel
            )
            result[channel.name] = items
            time.sleep(0.5)
        
        return result
    
    def fetch_by_tag(
        self,
        tag_id: int,
        pages: int = 5
    ) -> List[ZhiboItem]:
        """
        按标签获取内容
        
        Args:
            tag_id: 标签 ID
            pages: 页数
            
        Returns:
            ZhiboItem 列表
        """
        return self.fetch_multiple_pages(
            zhibo_id=152,  # 使用主频道
            tag_id=tag_id,
            pages=pages
        )
    
    def get_channels(self) -> List[ZhiboChannel]:
        """获取所有频道配置"""
        return self.CHANNELS.copy()
    
    def get_tags(self) -> Dict[int, str]:
        """获取所有标签配置"""
        return self.TAGS.copy()


if __name__ == "__main__":
    # 配置 loguru
    logger.remove()
    logger.add(
        sys.stderr,
        format="{time:HH:mm:ss} | {level: <8} | {message}",
        level="INFO"
    )
    
    # 测试采集器
    collector = SinaZhiboCollector()
    
    print("\n=== 频道列表 ===")
    for channel in collector.get_channels():
        status = "✓" if channel.enabled else "✗"
        print(f"  [{status}] {channel.zhibo_id}: {channel.name} - {channel.description}")
    
    print("\n=== 标签列表 ===")
    for tag_id, tag_name in collector.get_tags().items():
        print(f"  {tag_id}: {tag_name}")
    
    print("\n=== 测试获取 7x24 财经快讯 ===")
    items = collector.fetch_feed(zhibo_id=152, page_size=5)
    
    for item in items:
        print(f"\n[{item.create_time}] {item.channel_name}")
        print(f"  内容: {item.content[:100]}...")
        print(f"  标签: {', '.join(item.tags)}")
        if item.stocks:
            print(f"  关联股票: {item.stocks}")
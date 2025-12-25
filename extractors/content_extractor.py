"""
内容提取模块
Content Extraction Module

使用 trafilatura 从网页中提取主要内容
"""
import requests
import traceback
import sys
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

try:
    import trafilatura
    from trafilatura import fetch_url, extract, bare_extraction
    from trafilatura.settings import use_config
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    logger.warning("trafilatura 未安装，部分功能不可用")


@dataclass
class ExtractedContent:
    """提取的内容数据类"""
    url: str
    title: Optional[str] = None
    text: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    sitename: Optional[str] = None
    description: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    comments: Optional[str] = None
    language: Optional[str] = None
    raw_html: Optional[str] = None
    extraction_time: Optional[datetime] = None
    success: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        data['extraction_time'] = self.extraction_time.isoformat() if self.extraction_time else None
        return data


class ContentExtractor:
    """内容提取器"""
    
    def __init__(
        self,
        timeout: int = 30,
        include_comments: bool = True,
        include_tables: bool = True,
        include_links: bool = False,
        deduplicate: bool = True,
        min_length: int = 50,
        max_workers: int = 5
    ):
        """
        初始化内容提取器
        
        Args:
            timeout: 请求超时时间(秒)
            include_comments: 是否包含评论
            include_tables: 是否包含表格
            include_links: 是否包含链接
            deduplicate: 是否去重
            min_length: 最小内容长度
            max_workers: 最大并发工作线程数
        """
        self.timeout = timeout
        self.include_comments = include_comments
        self.include_tables = include_tables
        self.include_links = include_links
        self.deduplicate = deduplicate
        self.min_length = min_length
        self.max_workers = max_workers
        
        # 配置 trafilatura
        if TRAFILATURA_AVAILABLE:
            self.config = use_config()
            self.config.set("DEFAULT", "MIN_EXTRACTED_SIZE", str(min_length))
        
        # 配置 requests session
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5,zh-CN;q=0.3",
        })
    
    def fetch_html(self, url: str) -> Optional[str]:
        """
        获取网页 HTML 内容
        
        Args:
            url: 网页 URL
            
        Returns:
            HTML 字符串或 None
        """
        try:
            # 首先尝试使用 trafilatura
            if TRAFILATURA_AVAILABLE:
                html = fetch_url(url)
                if html:
                    return html
            
            # 降级使用 requests
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.text
            
        except Exception as e:
            logger.error(f"获取 HTML 失败 {url}: {e}")
            return None
    
    def extract_from_html(self, html: str, url: str = "") -> ExtractedContent:
        """
        从 HTML 中提取内容
        
        Args:
            html: HTML 字符串
            url: 原始 URL (用于元数据)
            
        Returns:
            ExtractedContent 对象
        """
        result = ExtractedContent(
            url=url,
            extraction_time=datetime.now()
        )
        
        if not TRAFILATURA_AVAILABLE:
            result.error = "trafilatura 未安装"
            return result
        
        try:
            # 使用 bare_extraction 获取完整数据
            extracted = bare_extraction(
                html,
                url=url,
                include_comments=self.include_comments,
                include_tables=self.include_tables,
                include_links=self.include_links,
                deduplicate=self.deduplicate,
                with_metadata=True,
                config=self.config
            )
            
            if extracted:
                result.title = extracted.title
                result.text = extracted.text
                result.author = extracted.author
                result.date = extracted.date
                result.sitename = extracted.sitename
                result.description = extracted.description
                result.categories = extracted.categories
                result.tags = extracted.tags
                result.comments = extracted.comments
                result.language = extracted.language
                result.success = bool(result.text and len(result.text) >= self.min_length)
                
                if not result.success:
                    result.error = "提取的内容过短或为空"
            else:
                result.error = "trafilatura 未能提取内容"
                
        except Exception as e:
            result.error = str(e)
            logger.error(f"内容提取失败: {e}")
            logger.error(traceback.format_exc())
        
        return result
    
    def extract_from_url(self, url: str) -> ExtractedContent:
        """
        从 URL 提取内容
        
        Args:
            url: 网页 URL
            
        Returns:
            ExtractedContent 对象
        """
        result = ExtractedContent(
            url=url,
            extraction_time=datetime.now()
        )
        
        # 获取 HTML
        html = self.fetch_html(url)
        if not html:
            result.error = "无法获取网页内容"
            return result
        
        result.raw_html = html
        
        # 提取内容
        return self.extract_from_html(html, url)
    
    def extract_batch(self, urls: List[str]) -> List[ExtractedContent]:
        """
        批量提取内容
        
        Args:
            urls: URL 列表
            
        Returns:
            ExtractedContent 列表
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_url = {
                executor.submit(self.extract_from_url, url): url
                for url in urls
            }
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"提取失败 {url}: {e}")
                    results.append(ExtractedContent(
                        url=url,
                        error=str(e),
                        extraction_time=datetime.now()
                    ))
        
        success_count = sum(1 for r in results if r.success)
        logger.info(f"批量提取完成: {success_count}/{len(urls)} 成功")
        
        return results
    
    def extract_text_only(self, url: str) -> Optional[str]:
        """
        仅提取纯文本内容
        
        Args:
            url: 网页 URL
            
        Returns:
            文本内容或 None
        """
        if not TRAFILATURA_AVAILABLE:
            return None
        
        try:
            html = self.fetch_html(url)
            if html:
                return extract(
                    html,
                    include_comments=False,
                    include_tables=False,
                    deduplicate=True
                )
        except Exception as e:
            logger.error(f"文本提取失败 {url}: {e}")
        
        return None
    
    def extract_with_metadata(self, url: str) -> Dict[str, Any]:
        """
        提取内容并返回元数据
        
        Args:
            url: 网页 URL
            
        Returns:
            包含内容和元数据的字典
        """
        result = self.extract_from_url(url)
        return result.to_dict()


class SimpleHTMLExtractor:
    """简单 HTML 内容提取器 (不依赖 trafilatura)"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
    
    def extract(self, url: str) -> Dict[str, Any]:
        """从 URL 提取内容"""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return {"error": "BeautifulSoup 未安装"}
        
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # 移除脚本和样式
            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                element.decompose()
            
            # 提取标题
            title = None
            if soup.title:
                title = soup.title.string
            elif soup.find('h1'):
                title = soup.find('h1').get_text()
            
            # 提取正文
            article = soup.find('article')
            if article:
                text = article.get_text(separator='\n', strip=True)
            else:
                # 查找主要内容区域
                main = soup.find(['main', 'div'], class_=['content', 'article', 'post'])
                if main:
                    text = main.get_text(separator='\n', strip=True)
                else:
                    text = soup.get_text(separator='\n', strip=True)
            
            # 清理文本
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            return {
                "url": url,
                "title": title,
                "text": text[:10000],  # 限制长度
                "success": len(text) > 100
            }
            
        except Exception as e:
            return {"url": url, "error": str(e), "success": False}


def get_extractor(use_trafilatura: bool = True, **kwargs) -> Any:
    """
    获取内容提取器工厂方法
    
    Args:
        use_trafilatura: 是否使用 trafilatura
        **kwargs: 其他参数
        
    Returns:
        提取器实例
    """
    if use_trafilatura and TRAFILATURA_AVAILABLE:
        return ContentExtractor(**kwargs)
    else:
        return SimpleHTMLExtractor(timeout=kwargs.get('timeout', 30))


if __name__ == "__main__":
    # 配置 loguru
    logger.remove()
    logger.add(
        sys.stderr,
        format="{time:HH:mm:ss} | {level: <8} | {message}",
        level="INFO"
    )
    
    # 测试提取器
    extractor = ContentExtractor()
    
    # 测试 URL
    test_urls = [
        "https://github.blog/2019-03-29-leader-spotlight-erin-spiceland/",
    ]
    
    print("\n=== 测试内容提取 ===")
    
    for url in test_urls:
        print(f"\n提取: {url}")
        result = extractor.extract_from_url(url)
        
        if result.success:
            print(f"  标题: {result.title}")
            print(f"  作者: {result.author}")
            print(f"  日期: {result.date}")
            print(f"  站点: {result.sitename}")
            print(f"  内容长度: {len(result.text or '')} 字符")
            if result.text:
                print(f"  内容预览: {result.text[:200]}...")
        else:
            print(f"  失败: {result.error}")

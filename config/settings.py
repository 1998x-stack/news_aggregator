"""
新闻聚合分析系统 - 配置文件
News Aggregator & Analyzer System - Configuration

包含所有系统配置、分类定义、重要性评估标准等
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from pathlib import Path


# ============================================================
# 1. 路径配置
# ============================================================

class PathConfig:
    """路径配置"""
    # 项目根目录
    ROOT_DIR = Path(__file__).parent.parent
    
    # 数据目录
    DATA_DIR = ROOT_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    
    # 输出目录
    OUTPUT_DIR = ROOT_DIR / "outputs"
    REPORTS_DIR = OUTPUT_DIR / "reports"
    JSON_DIR = OUTPUT_DIR / "json"
    
    # 日志目录
    LOG_DIR = ROOT_DIR / "logs"
    
    @classmethod
    def ensure_dirs(cls):
        """确保所有目录存在"""
        for attr in dir(cls):
            if attr.endswith('_DIR'):
                path = getattr(cls, attr)
                if isinstance(path, Path):
                    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 内容分类定义
# ============================================================

class ContentCategory(str, Enum):
    """内容分类枚举"""
    # 技术类
    AI_ML = "ai_ml"                     # 人工智能/机器学习
    PROGRAMMING = "programming"         # 编程开发
    CLOUD_INFRA = "cloud_infra"        # 云计算/基础设施
    SECURITY = "security"              # 网络安全
    DATA_SCIENCE = "data_science"      # 数据科学
    BLOCKCHAIN = "blockchain"          # 区块链/加密货币
    HARDWARE = "hardware"              # 硬件/芯片
    
    # 商业类
    STARTUP = "startup"                # 创业/融资
    BIG_TECH = "big_tech"              # 大厂动态
    MARKET = "market"                  # 市场/经济
    POLICY = "policy"                  # 政策/法规
    
    # 其他
    RESEARCH = "research"              # 学术研究
    PRODUCT = "product"                # 产品发布
    OTHER = "other"                    # 其他


class ImportanceLevel(int, Enum):
    """重要性级别"""
    CRITICAL = 5      # 极其重要 - 重大突破、重要政策
    HIGH = 4          # 高度重要 - 行业重大新闻
    MEDIUM = 3        # 中等重要 - 值得关注
    LOW = 2           # 较低重要 - 一般资讯
    MINIMAL = 1       # 最低重要 - 边缘信息


# ============================================================
# 3. 分类规则配置
# ============================================================

CATEGORY_KEYWORDS: Dict[ContentCategory, List[str]] = {
    ContentCategory.AI_ML: [
        "artificial intelligence", "machine learning", "deep learning",
        "neural network", "gpt", "llm", "transformer", "chatgpt",
        "openai", "anthropic", "claude", "gemini", "人工智能", "机器学习",
        "大模型", "深度学习", "神经网络", "nlp", "computer vision"
    ],
    ContentCategory.PROGRAMMING: [
        "programming", "coding", "developer", "software", "python",
        "javascript", "rust", "golang", "api", "framework", "编程",
        "开发者", "代码", "开源", "github", "docker", "kubernetes"
    ],
    ContentCategory.CLOUD_INFRA: [
        "cloud", "aws", "azure", "gcp", "infrastructure", "devops",
        "serverless", "微服务", "云计算", "容器", "数据中心"
    ],
    ContentCategory.SECURITY: [
        "security", "cybersecurity", "hack", "breach", "vulnerability",
        "malware", "ransomware", "网络安全", "漏洞", "数据泄露"
    ],
    ContentCategory.DATA_SCIENCE: [
        "data science", "analytics", "big data", "数据分析", "大数据",
        "数据挖掘", "数据可视化", "bi", "etl"
    ],
    ContentCategory.BLOCKCHAIN: [
        "blockchain", "crypto", "bitcoin", "ethereum", "web3", "nft",
        "defi", "区块链", "加密货币", "比特币", "以太坊"
    ],
    ContentCategory.HARDWARE: [
        "chip", "semiconductor", "nvidia", "amd", "intel", "processor",
        "gpu", "芯片", "半导体", "处理器", "硬件"
    ],
    ContentCategory.STARTUP: [
        "startup", "funding", "series a", "series b", "vc", "venture",
        "创业", "融资", "独角兽", "ipo", "估值"
    ],
    ContentCategory.BIG_TECH: [
        "google", "apple", "microsoft", "meta", "amazon", "tesla",
        "谷歌", "苹果", "微软", "阿里巴巴", "腾讯", "字节跳动"
    ],
    ContentCategory.MARKET: [
        "market", "stock", "nasdaq", "economy", "recession",
        "市场", "股票", "经济", "金融", "投资"
    ],
    ContentCategory.POLICY: [
        "regulation", "policy", "antitrust", "gdpr", "法规",
        "政策", "监管", "反垄断", "合规"
    ],
    ContentCategory.RESEARCH: [
        "research", "paper", "study", "academic", "arxiv",
        "研究", "论文", "学术", "实验"
    ],
    ContentCategory.PRODUCT: [
        "launch", "release", "announcement", "product", "feature",
        "发布", "新品", "更新", "功能"
    ],
}


# ============================================================
# 4. 重要性评估规则
# ============================================================

@dataclass
class ImportanceRule:
    """重要性评估规则"""
    name: str
    keywords: List[str]
    boost: int  # 重要性加成 (-2 到 +2)
    description: str


IMPORTANCE_RULES: List[ImportanceRule] = [
    # 高重要性规则
    ImportanceRule(
        name="突破性进展",
        keywords=["breakthrough", "revolutionary", "first", "突破", "首次", "里程碑"],
        boost=2,
        description="技术或商业突破性进展"
    ),
    ImportanceRule(
        name="重大融资",
        keywords=["billion", "series d", "series e", "ipo", "十亿", "上市"],
        boost=2,
        description="大额融资或上市"
    ),
    ImportanceRule(
        name="安全事件",
        keywords=["breach", "hack", "vulnerability", "泄露", "攻击", "漏洞"],
        boost=2,
        description="重大安全事件"
    ),
    ImportanceRule(
        name="政策法规",
        keywords=["regulation", "ban", "law", "act", "法规", "禁止", "法案"],
        boost=1,
        description="政策法规变化"
    ),
    
    # 中等重要性规则
    ImportanceRule(
        name="产品发布",
        keywords=["launch", "release", "announce", "发布", "推出"],
        boost=1,
        description="产品或功能发布"
    ),
    ImportanceRule(
        name="行业报告",
        keywords=["report", "survey", "study", "报告", "调查", "研究"],
        boost=0,
        description="行业研究报告"
    ),
    
    # 低重要性规则
    ImportanceRule(
        name="招聘信息",
        keywords=["hiring", "job", "career", "招聘", "岗位"],
        boost=-1,
        description="招聘相关"
    ),
    ImportanceRule(
        name="活动预告",
        keywords=["event", "conference", "webinar", "活动", "会议"],
        boost=-1,
        description="活动或会议预告"
    ),
]


# ============================================================
# 5. Ollama 模型配置
# ============================================================

@dataclass
class OllamaModelConfig:
    """Ollama 模型配置"""
    name: str                    # 模型名称
    model_id: str               # 模型 ID
    purpose: str                # 用途
    temperature: float = 0.3    # 温度
    max_tokens: int = 2048      # 最大 token 数
    top_p: float = 0.9          # Top P
    timeout: int = 60           # 超时时间


OLLAMA_MODELS = {
    "classifier": OllamaModelConfig(
        name="分类模型",
        model_id="qwen2.5:0.5b",
        purpose="内容分类和重要性评估",
        temperature=0.1,
        max_tokens=512,
        timeout=30
    ),
    "extractor": OllamaModelConfig(
        name="提取模型", 
        model_id="qwen3:4b",
        purpose="5W2H 信息提取和摘要生成",
        temperature=0.3,
        max_tokens=2048,
        timeout=60
    ),
}


@dataclass
class OllamaConfig:
    """Ollama 服务配置"""
    host: str = "http://localhost:11434"
    api_endpoint: str = "/api/generate"
    chat_endpoint: str = "/api/chat"
    models_endpoint: str = "/api/tags"
    pull_endpoint: str = "/api/pull"


# ============================================================
# 6. 数据源配置
# ============================================================

@dataclass  
class DataSourceConfig:
    """数据源配置"""
    name: str
    enabled: bool = True
    priority: int = 5
    fetch_interval: int = 300  # 秒
    max_items: int = 100


DATA_SOURCES = {
    "hackernews": DataSourceConfig(
        name="HackerNews",
        enabled=True,
        priority=10,
        fetch_interval=300,
        max_items=50
    ),
    "sina_zhibo": DataSourceConfig(
        name="新浪财经",
        enabled=True,
        priority=8,
        fetch_interval=180,
        max_items=100
    ),
    "rss_tech": DataSourceConfig(
        name="科技RSS",
        enabled=True,
        priority=7,
        fetch_interval=600,
        max_items=200
    ),
}


# ============================================================
# 7. 报告配置
# ============================================================

@dataclass
class ReportConfig:
    """报告配置"""
    # 报告类型
    generate_daily_report: bool = True
    generate_category_report: bool = True
    generate_timeline_report: bool = True
    
    # 报告格式
    output_formats: List[str] = field(default_factory=lambda: ["markdown", "json", "html"])
    
    # 报告内容
    max_items_per_category: int = 20
    include_raw_content: bool = False
    include_analysis: bool = True
    
    # 文件命名
    date_format: str = "%Y%m%d"
    time_format: str = "%H%M%S"


# ============================================================
# 8. 日志配置
# ============================================================

LOG_CONFIG = {
    "rotation": "10 MB",
    "retention": "7 days",
    "compression": "zip",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    "level": "INFO",
    "backtrace": True,
    "diagnose": True,
}


# ============================================================
# 9. 系统配置汇总
# ============================================================

@dataclass
class SystemConfig:
    """系统配置汇总"""
    paths: PathConfig = field(default_factory=PathConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    
    # 全局设置
    debug_mode: bool = False
    max_concurrent_requests: int = 5
    request_timeout: int = 30
    retry_times: int = 3
    retry_delay: int = 5


# 默认配置实例
config = SystemConfig()


# ============================================================
# 10. 兼容性别名和辅助定义
# ============================================================

# CLASSIFICATION_KEYWORDS - 别名，兼容其他模块引用
CLASSIFICATION_KEYWORDS: Dict[str, List[str]] = {
    cat.value: keywords for cat, keywords in CATEGORY_KEYWORDS.items()
}

# IMPORTANCE_LEVELS - 重要性级别定义
IMPORTANCE_LEVELS: Dict[int, Dict[str, str]] = {
    5: {"label": "关键", "description": "重大突破、颠覆性事件、重要政策"},
    4: {"label": "重要", "description": "行业重大新闻、大额融资"},
    3: {"label": "中等", "description": "值得关注的信息"},
    2: {"label": "一般", "description": "常规资讯"},
    1: {"label": "边缘", "description": "边缘信息、低相关度"}
}

# IMPORTANCE_KEYWORDS - 重要性关键词及其权重
IMPORTANCE_KEYWORDS: Dict[str, int] = {
    # 高重要性关键词 (+2)
    "breakthrough": 2, "revolutionary": 2, "first": 2,
    "突破": 2, "首次": 2, "里程碑": 2,
    "billion": 2, "ipo": 2, "上市": 2, "十亿": 2,
    "breach": 2, "hack": 2, "泄露": 2, "攻击": 2,
    # 中等重要性关键词 (+1)
    "launch": 1, "release": 1, "announce": 1,
    "发布": 1, "推出": 1, "宣布": 1,
    "regulation": 1, "ban": 1, "law": 1,
    "法规": 1, "禁止": 1, "法案": 1,
    # 低重要性关键词 (-1)
    "hiring": -1, "job": -1, "career": -1,
    "招聘": -1, "岗位": -1,
    "event": -1, "conference": -1, "webinar": -1,
    "活动": -1, "会议": -1,
}

# 分离的提升/惩罚关键词列表（用于分类器）
IMPORTANCE_BOOST_KEYWORDS: List[str] = [
    # +2 关键词
    "breakthrough", "revolutionary", "first", "突破", "首次", "里程碑",
    "billion", "ipo", "上市", "十亿",
    "breach", "hack", "泄露", "攻击",
    # +1 关键词
    "launch", "release", "announce", "发布", "推出", "宣布",
    "regulation", "ban", "law", "法规", "禁止", "法案",
]

IMPORTANCE_PENALTY_KEYWORDS: List[str] = [
    "hiring", "job", "career", "招聘", "岗位",
    "event", "conference", "webinar", "活动", "会议",
]


class LLMSettings:
    """LLM模型设置"""
    # 分类模型 - 使用小模型，快速分类
    CLASSIFIER_MODEL: str = "qwen2.5:0.5b"
    CLASSIFIER_TEMPERATURE: float = 0.1
    CLASSIFIER_MAX_TOKENS: int = 512
    
    # 抽取模型 - 使用较大模型，深度理解
    EXTRACTOR_MODEL: str = "qwen3:4b"
    EXTRACTOR_TEMPERATURE: float = 0.3
    EXTRACTOR_MAX_TOKENS: int = 2048
    
    # Ollama服务配置
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: int = 120
    OLLAMA_MAX_RETRIES: int = 3


# ============================================================
# 11. 配置验证和初始化
# ============================================================

def validate_config() -> bool:
    """验证配置有效性"""
    errors = []
    
    # 检查 Ollama 模型配置
    for name, model_config in OLLAMA_MODELS.items():
        if not model_config.model_id:
            errors.append(f"模型 {name} 缺少 model_id")
    
    # 检查分类关键词
    for category, keywords in CATEGORY_KEYWORDS.items():
        if not keywords:
            errors.append(f"分类 {category.value} 没有关键词")
    
    if errors:
        for error in errors:
            print(f"配置错误: {error}")
        return False
    
    return True


def init_system():
    """初始化系统"""
    # 确保目录存在
    PathConfig.ensure_dirs()
    
    # 验证配置
    if not validate_config():
        raise ValueError("配置验证失败")
    
    print("系统初始化完成")


if __name__ == "__main__":
    # 验证配置
    if validate_config():
        print("配置验证通过")
    
    # 打印分类信息
    print("\n=== 内容分类 ===")
    for category in ContentCategory:
        keywords = CATEGORY_KEYWORDS.get(category, [])[:3]
        print(f"  {category.value}: {', '.join(keywords)}...")
    
    # 打印重要性级别
    print("\n=== 重要性级别 ===")
    for level in ImportanceLevel:
        print(f"  {level.name}: {level.value}")
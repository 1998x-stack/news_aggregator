#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件工具模块

提供文件命名、保存、路径管理等功能:
- 规范化文件命名系统
- 自动目录创建
- JSON/Markdown/CSV文件保存
- 文件去重与版本管理
- 报告文件组织

依赖:
    - loguru: 日志记录

作者: News Aggregator System
创建日期: 2025-12-25
"""

import os
import json
import hashlib
import traceback
import sys
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger


class FileType(Enum):
    """文件类型枚举"""
    JSON = "json"
    MARKDOWN = "md"
    CSV = "csv"
    TXT = "txt"
    HTML = "html"


class ReportType(Enum):
    """报告类型枚举"""
    DAILY_SUMMARY = "daily_summary"           # 每日摘要
    CATEGORY_REPORT = "category_report"       # 分类报告
    TREND_ANALYSIS = "trend_analysis"         # 趋势分析
    TIMELINE = "timeline"                     # 时间线报告
    HOT_TOPICS = "hot_topics"                 # 热点话题
    INSIGHT = "insight"                       # 洞察报告
    RAW_DATA = "raw_data"                     # 原始数据
    PROCESSED_DATA = "processed_data"         # 处理后数据


class DataSource(Enum):
    """数据源枚举"""
    HACKERNEWS = "hn"
    RSS = "rss"
    SINA_ZHIBO = "sina"
    MIXED = "mixed"
    ALL = "all"


@dataclass
class FileNameConfig:
    """文件命名配置"""
    base_dir: str = "./outputs"
    date_format: str = "%Y%m%d"
    time_format: str = "%H%M%S"
    datetime_format: str = "%Y%m%d_%H%M%S"
    separator: str = "_"
    max_name_length: int = 200
    
    # 子目录结构
    subdirs: Dict[str, str] = field(default_factory=lambda: {
        "daily": "daily",
        "category": "category",
        "trend": "trend",
        "timeline": "timeline",
        "raw": "raw_data",
        "processed": "processed",
        "archive": "archive"
    })


@dataclass
class FileInfo:
    """文件信息数据类"""
    filepath: str
    filename: str
    directory: str
    file_type: FileType
    created_at: datetime
    size_bytes: int = 0
    checksum: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "filepath": self.filepath,
            "filename": self.filename,
            "directory": self.directory,
            "file_type": self.file_type.value,
            "created_at": self.created_at.isoformat(),
            "size_bytes": self.size_bytes,
            "checksum": self.checksum
        }


class FileManager:
    """
    文件管理器
    
    负责文件的命名、保存、组织和管理
    
    使用示例:
        >>> manager = FileManager()
        >>> filepath = manager.save_json(data, ReportType.DAILY_SUMMARY)
        >>> print(filepath)
    """
    
    def __init__(self, config: Optional[FileNameConfig] = None):
        """
        初始化文件管理器
        
        Args:
            config: 文件命名配置，为None时使用默认配置
        """
        self.config = config or FileNameConfig()
        self._ensure_directories()
        logger.info(f"FileManager初始化完成，基础目录: {self.config.base_dir}")
    
    def _ensure_directories(self):
        """确保所有必需的目录存在"""
        base_path = Path(self.config.base_dir)
        base_path.mkdir(parents=True, exist_ok=True)
        
        for subdir in self.config.subdirs.values():
            (base_path / subdir).mkdir(parents=True, exist_ok=True)
        
        logger.debug("目录结构已创建")
    
    def _sanitize_filename(self, name: str) -> str:
        """
        清理文件名，移除非法字符
        
        Args:
            name: 原始文件名
            
        Returns:
            str: 清理后的文件名
        """
        # 替换非法字符
        illegal_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in illegal_chars:
            name = name.replace(char, '_')
        
        # 移除连续的分隔符
        while '__' in name:
            name = name.replace('__', '_')
        
        # 截断过长的文件名
        if len(name) > self.config.max_name_length:
            name = name[:self.config.max_name_length]
        
        return name.strip('_')
    
    def generate_filename(
        self,
        report_type: ReportType,
        source: DataSource = DataSource.ALL,
        category: Optional[str] = None,
        date: Optional[datetime] = None,
        suffix: Optional[str] = None,
        file_type: FileType = FileType.JSON
    ) -> str:
        """
        生成规范化的文件名
        
        文件命名规则:
            {date}_{report_type}_{source}[_{category}][_{suffix}].{ext}
        
        示例:
            20251225_daily_summary_all.json
            20251225_category_report_hn_ai_ml.md
            20251225_143022_trend_analysis_mixed_weekly.json
        
        Args:
            report_type: 报告类型
            source: 数据源
            category: 内容分类（可选）
            date: 日期时间（默认为当前时间）
            suffix: 额外后缀（可选）
            file_type: 文件类型
            
        Returns:
            str: 生成的文件名
        """
        date = date or datetime.now()
        sep = self.config.separator
        
        # 构建文件名各部分
        parts = [
            date.strftime(self.config.date_format),
            report_type.value,
            source.value
        ]
        
        if category:
            parts.append(self._sanitize_filename(category))
        
        if suffix:
            parts.append(self._sanitize_filename(suffix))
        
        filename = sep.join(parts) + f".{file_type.value}"
        return self._sanitize_filename(filename)
    
    def generate_filepath(
        self,
        report_type: ReportType,
        source: DataSource = DataSource.ALL,
        category: Optional[str] = None,
        date: Optional[datetime] = None,
        suffix: Optional[str] = None,
        file_type: FileType = FileType.JSON
    ) -> str:
        """
        生成完整的文件路径
        
        Args:
            report_type: 报告类型
            source: 数据源
            category: 内容分类（可选）
            date: 日期时间
            suffix: 额外后缀
            file_type: 文件类型
            
        Returns:
            str: 完整的文件路径
        """
        filename = self.generate_filename(
            report_type=report_type,
            source=source,
            category=category,
            date=date,
            suffix=suffix,
            file_type=file_type
        )
        
        # 确定子目录
        if report_type in [ReportType.RAW_DATA]:
            subdir = self.config.subdirs["raw"]
        elif report_type in [ReportType.PROCESSED_DATA]:
            subdir = self.config.subdirs["processed"]
        elif report_type in [ReportType.DAILY_SUMMARY]:
            subdir = self.config.subdirs["daily"]
        elif report_type in [ReportType.CATEGORY_REPORT]:
            subdir = self.config.subdirs["category"]
        elif report_type in [ReportType.TREND_ANALYSIS, ReportType.TIMELINE]:
            subdir = self.config.subdirs["trend"]
        else:
            subdir = ""
        
        return os.path.join(self.config.base_dir, subdir, filename)
    
    def _calculate_checksum(self, content: Union[str, bytes]) -> str:
        """
        计算内容的MD5校验和
        
        Args:
            content: 文件内容
            
        Returns:
            str: MD5校验和
        """
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.md5(content).hexdigest()
    
    def save_json(
        self,
        data: Union[Dict, List],
        report_type: ReportType,
        source: DataSource = DataSource.ALL,
        category: Optional[str] = None,
        date: Optional[datetime] = None,
        suffix: Optional[str] = None,
        indent: int = 2,
        ensure_ascii: bool = False
    ) -> FileInfo:
        """
        保存JSON文件
        
        Args:
            data: 要保存的数据
            report_type: 报告类型
            source: 数据源
            category: 内容分类
            date: 日期时间
            suffix: 额外后缀
            indent: JSON缩进
            ensure_ascii: 是否确保ASCII编码
            
        Returns:
            FileInfo: 文件信息
        """
        filepath = self.generate_filepath(
            report_type=report_type,
            source=source,
            category=category,
            date=date,
            suffix=suffix,
            file_type=FileType.JSON
        )
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            content = json.dumps(
                data, 
                indent=indent, 
                ensure_ascii=ensure_ascii,
                default=lambda o: o.isoformat() if isinstance(o, (datetime, date)) else str(o)
            )
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            file_info = FileInfo(
                filepath=filepath,
                filename=os.path.basename(filepath),
                directory=os.path.dirname(filepath),
                file_type=FileType.JSON,
                created_at=datetime.now(),
                size_bytes=os.path.getsize(filepath),
                checksum=self._calculate_checksum(content)
            )
            
            logger.info(f"JSON文件已保存: {filepath}")
            return file_info
            
        except Exception as e:
            logger.error(f"保存JSON文件失败: {e}")
            traceback.print_exc(file=sys.stderr)
            raise
    
    def save_markdown(
        self,
        content: str,
        report_type: ReportType,
        source: DataSource = DataSource.ALL,
        category: Optional[str] = None,
        date: Optional[datetime] = None,
        suffix: Optional[str] = None
    ) -> FileInfo:
        """
        保存Markdown文件
        
        Args:
            content: Markdown内容
            report_type: 报告类型
            source: 数据源
            category: 内容分类
            date: 日期时间
            suffix: 额外后缀
            
        Returns:
            FileInfo: 文件信息
        """
        filepath = self.generate_filepath(
            report_type=report_type,
            source=source,
            category=category,
            date=date,
            suffix=suffix,
            file_type=FileType.MARKDOWN
        )
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            file_info = FileInfo(
                filepath=filepath,
                filename=os.path.basename(filepath),
                directory=os.path.dirname(filepath),
                file_type=FileType.MARKDOWN,
                created_at=datetime.now(),
                size_bytes=os.path.getsize(filepath),
                checksum=self._calculate_checksum(content)
            )
            
            logger.info(f"Markdown文件已保存: {filepath}")
            return file_info
            
        except Exception as e:
            logger.error(f"保存Markdown文件失败: {e}")
            traceback.print_exc(file=sys.stderr)
            raise
    
    def save_csv(
        self,
        data: List[Dict[str, Any]],
        report_type: ReportType,
        source: DataSource = DataSource.ALL,
        category: Optional[str] = None,
        date: Optional[datetime] = None,
        suffix: Optional[str] = None,
        headers: Optional[List[str]] = None
    ) -> FileInfo:
        """
        保存CSV文件
        
        Args:
            data: 要保存的数据（字典列表）
            report_type: 报告类型
            source: 数据源
            category: 内容分类
            date: 日期时间
            suffix: 额外后缀
            headers: 列标题（可选，默认从数据推断）
            
        Returns:
            FileInfo: 文件信息
        """
        import csv
        
        filepath = self.generate_filepath(
            report_type=report_type,
            source=source,
            category=category,
            date=date,
            suffix=suffix,
            file_type=FileType.CSV
        )
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if not data:
                headers = headers or []
            else:
                headers = headers or list(data[0].keys())
            
            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(data)
            
            file_info = FileInfo(
                filepath=filepath,
                filename=os.path.basename(filepath),
                directory=os.path.dirname(filepath),
                file_type=FileType.CSV,
                created_at=datetime.now(),
                size_bytes=os.path.getsize(filepath),
                checksum=""
            )
            
            logger.info(f"CSV文件已保存: {filepath}")
            return file_info
            
        except Exception as e:
            logger.error(f"保存CSV文件失败: {e}")
            traceback.print_exc(file=sys.stderr)
            raise
    
    def save_text(
        self,
        content: str,
        filename: str,
        subdir: Optional[str] = None
    ) -> FileInfo:
        """
        保存纯文本文件
        
        Args:
            content: 文本内容
            filename: 文件名
            subdir: 子目录
            
        Returns:
            FileInfo: 文件信息
        """
        if subdir:
            filepath = os.path.join(self.config.base_dir, subdir, filename)
        else:
            filepath = os.path.join(self.config.base_dir, filename)
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            file_info = FileInfo(
                filepath=filepath,
                filename=os.path.basename(filepath),
                directory=os.path.dirname(filepath),
                file_type=FileType.TXT,
                created_at=datetime.now(),
                size_bytes=os.path.getsize(filepath),
                checksum=self._calculate_checksum(content)
            )
            
            logger.info(f"文本文件已保存: {filepath}")
            return file_info
            
        except Exception as e:
            logger.error(f"保存文本文件失败: {e}")
            traceback.print_exc(file=sys.stderr)
            raise
    
    def load_json(self, filepath: str) -> Dict[str, Any]:
        """
        加载JSON文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            Dict: 加载的数据
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载JSON文件失败: {e}")
            traceback.print_exc(file=sys.stderr)
            return {}
    
    def load_markdown(self, filepath: str) -> str:
        """
        加载Markdown文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            str: 文件内容
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"加载Markdown文件失败: {e}")
            traceback.print_exc(file=sys.stderr)
            return ""
    
    def list_files(
        self,
        subdir: Optional[str] = None,
        file_type: Optional[FileType] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[FileInfo]:
        """
        列出文件
        
        Args:
            subdir: 子目录
            file_type: 文件类型过滤
            date_from: 起始日期
            date_to: 结束日期
            
        Returns:
            List[FileInfo]: 文件信息列表
        """
        if subdir:
            search_dir = os.path.join(self.config.base_dir, subdir)
        else:
            search_dir = self.config.base_dir
        
        files = []
        
        try:
            for root, _, filenames in os.walk(search_dir):
                for filename in filenames:
                    filepath = os.path.join(root, filename)
                    
                    # 文件类型过滤
                    ext = filename.split('.')[-1] if '.' in filename else ''
                    try:
                        ft = FileType(ext)
                    except ValueError:
                        continue
                    
                    if file_type and ft != file_type:
                        continue
                    
                    # 获取文件信息
                    stat = os.stat(filepath)
                    created = datetime.fromtimestamp(stat.st_ctime)
                    
                    # 日期过滤
                    if date_from and created < date_from:
                        continue
                    if date_to and created > date_to:
                        continue
                    
                    files.append(FileInfo(
                        filepath=filepath,
                        filename=filename,
                        directory=root,
                        file_type=ft,
                        created_at=created,
                        size_bytes=stat.st_size
                    ))
            
            return sorted(files, key=lambda x: x.created_at, reverse=True)
            
        except Exception as e:
            logger.error(f"列出文件失败: {e}")
            traceback.print_exc(file=sys.stderr)
            return []
    
    def archive_old_files(
        self,
        days_old: int = 30,
        move: bool = True
    ) -> int:
        """
        归档旧文件
        
        Args:
            days_old: 多少天前的文件需要归档
            move: 是否移动文件（False时仅统计）
            
        Returns:
            int: 归档的文件数量
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        archive_dir = os.path.join(self.config.base_dir, self.config.subdirs["archive"])
        
        files = self.list_files(date_to=cutoff_date)
        archived_count = 0
        
        for file_info in files:
            if self.config.subdirs["archive"] in file_info.filepath:
                continue
            
            if move:
                try:
                    # 保持目录结构
                    rel_path = os.path.relpath(file_info.filepath, self.config.base_dir)
                    new_path = os.path.join(archive_dir, rel_path)
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    os.rename(file_info.filepath, new_path)
                    archived_count += 1
                except Exception as e:
                    logger.warning(f"归档文件失败 {file_info.filepath}: {e}")
            else:
                archived_count += 1
        
        logger.info(f"归档完成，共 {archived_count} 个文件")
        return archived_count
    
    def get_latest_file(
        self,
        report_type: ReportType,
        source: Optional[DataSource] = None
    ) -> Optional[FileInfo]:
        """
        获取最新的文件
        
        Args:
            report_type: 报告类型
            source: 数据源
            
        Returns:
            FileInfo: 最新文件信息，不存在时返回None
        """
        # 确定子目录
        if report_type in [ReportType.RAW_DATA]:
            subdir = self.config.subdirs["raw"]
        elif report_type in [ReportType.DAILY_SUMMARY]:
            subdir = self.config.subdirs["daily"]
        elif report_type in [ReportType.CATEGORY_REPORT]:
            subdir = self.config.subdirs["category"]
        else:
            subdir = None
        
        files = self.list_files(subdir=subdir)
        
        # 过滤匹配的文件
        matching = []
        for f in files:
            if report_type.value in f.filename:
                if source is None or source.value in f.filename:
                    matching.append(f)
        
        return matching[0] if matching else None


def create_file_manager(base_dir: str = "./outputs") -> FileManager:
    """
    工厂函数：创建文件管理器
    
    Args:
        base_dir: 基础输出目录
        
    Returns:
        FileManager: 文件管理器实例
    """
    config = FileNameConfig(base_dir=base_dir)
    return FileManager(config)


# ============================================================
# 独立工具函数 (不需要FileManager实例)
# ============================================================

def ensure_dir(path: str) -> None:
    """
    确保目录存在，如不存在则创建
    
    Args:
        path: 目录路径
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(
    data: Union[Dict, List],
    filepath: str,
    indent: int = 2,
    ensure_ascii: bool = False
) -> str:
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        filepath: 文件路径
        indent: 缩进空格数
        ensure_ascii: 是否转义非ASCII字符
        
    Returns:
        str: 保存的文件路径
    """
    try:
        # 确保目录存在
        ensure_dir(os.path.dirname(filepath) or ".")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(
                data, 
                f, 
                indent=indent, 
                ensure_ascii=ensure_ascii,
                default=lambda o: o.isoformat() if isinstance(o, (datetime, date)) else str(o)
            )
        
        logger.debug(f"JSON已保存: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"保存JSON失败: {e}")
        raise


def load_json(filepath: str) -> Union[Dict, List]:
    """
    从JSON文件加载数据
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载的数据（字典或列表）
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"文件不存在: {filepath}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {e}")
        return {}
    except Exception as e:
        logger.error(f"加载JSON失败: {e}")
        return {}


def get_file_hash(filepath: str, algorithm: str = "md5") -> str:
    """
    计算文件的哈希值
    
    Args:
        filepath: 文件路径
        algorithm: 哈希算法 (md5, sha1, sha256)
        
    Returns:
        str: 哈希值（十六进制字符串）
    """
    try:
        hash_func = getattr(hashlib, algorithm)()
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
        
    except Exception as e:
        logger.error(f"计算文件哈希失败: {e}")
        return ""


def safe_filename(name: str, max_length: int = 200) -> str:
    """
    生成安全的文件名，移除非法字符
    
    Args:
        name: 原始文件名
        max_length: 最大长度
        
    Returns:
        str: 安全的文件名
    """
    # 替换非法字符
    illegal_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '\n', '\r', '\t']
    for char in illegal_chars:
        name = name.replace(char, '_')
    
    # 移除连续的下划线
    while '__' in name:
        name = name.replace('__', '_')
    
    # 移除首尾的下划线和空格
    name = name.strip('_ ')
    
    # 截断过长的文件名
    if len(name) > max_length:
        name = name[:max_length]
    
    return name


if __name__ == "__main__":
    # 测试代码
    logger.add(sys.stderr, level="DEBUG")
    
    manager = create_file_manager("./test_outputs")
    
    # 测试文件名生成
    print("=== 文件名生成测试 ===")
    filename = manager.generate_filename(
        report_type=ReportType.DAILY_SUMMARY,
        source=DataSource.HACKERNEWS
    )
    print(f"每日摘要文件名: {filename}")
    
    filename = manager.generate_filename(
        report_type=ReportType.CATEGORY_REPORT,
        source=DataSource.RSS,
        category="ai_ml"
    )
    print(f"分类报告文件名: {filename}")
    
    # 测试JSON保存
    print("\n=== JSON保存测试 ===")
    test_data = {
        "title": "测试数据",
        "items": [1, 2, 3],
        "metadata": {"created": "2025-12-25"}
    }
    file_info = manager.save_json(
        data=test_data,
        report_type=ReportType.RAW_DATA,
        source=DataSource.HACKERNEWS
    )
    print(f"保存路径: {file_info.filepath}")
    print(f"文件大小: {file_info.size_bytes} bytes")
    
    # 测试Markdown保存
    print("\n=== Markdown保存测试 ===")
    md_content = """# 测试报告

## 概述
这是一个测试报告。

## 内容
- 项目1
- 项目2
"""
    file_info = manager.save_markdown(
        content=md_content,
        report_type=ReportType.DAILY_SUMMARY,
        source=DataSource.ALL
    )
    print(f"保存路径: {file_info.filepath}")
    
    # 测试文件列表
    print("\n=== 文件列表 ===")
    files = manager.list_files()
    for f in files:
        print(f"  {f.filename} - {f.size_bytes} bytes")
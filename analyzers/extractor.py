#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5W2H信息抽取模块

使用Ollama LLM从新闻内容中抽取结构化信息:
- What: 发生了什么事件
- Who: 涉及哪些人物/组织
- When: 时间信息
- Where: 地点信息
- Why: 原因分析
- How: 如何发生/解决方案
- How Much: 规模/数量/金额

作者: Claude
日期: 2024-12
"""

import sys
import json
import traceback
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, 'news_aggregator')

from utils.ollama_client import OllamaClient, OllamaConfig
from prompts.llm_prompts import get_prompt, PromptType
from config.settings import LLMSettings


@dataclass
class ExtractedInfo:
    """抽取的5W2H信息结构"""
    
    # 原始信息
    source_id: str = ""
    source_url: str = ""
    source_title: str = ""
    
    # 5W2H核心信息
    what: str = ""  # 事件描述
    who: List[str] = field(default_factory=list)  # 相关人物/组织
    when: str = ""  # 时间
    where: str = ""  # 地点
    why: str = ""  # 原因
    how: str = ""  # 方式/方法
    how_much: str = ""  # 规模/数量
    
    # 衍生信息
    key_points: List[str] = field(default_factory=list)  # 关键要点
    summary: str = ""  # 摘要(50字以内)
    entities: Dict[str, List[str]] = field(default_factory=dict)  # 命名实体
    
    # 元数据
    extraction_time: str = ""
    model_used: str = ""
    confidence: float = 0.0
    raw_response: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "source_id": self.source_id,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "what": self.what,
            "who": self.who,
            "when": self.when,
            "where": self.where,
            "why": self.why,
            "how": self.how,
            "how_much": self.how_much,
            "key_points": self.key_points,
            "summary": self.summary,
            "entities": self.entities,
            "extraction_time": self.extraction_time,
            "model_used": self.model_used,
            "confidence": self.confidence
        }
    
    def is_valid(self) -> bool:
        """检查是否有效抽取"""
        return bool(self.what or self.summary or self.key_points)
    
    def get_completeness_score(self) -> float:
        """计算信息完整度评分(0-1)"""
        fields = [
            self.what, self.who, self.when, self.where,
            self.why, self.how, self.how_much, self.summary
        ]
        filled = sum(1 for f in fields if f and (isinstance(f, str) and f.strip() or isinstance(f, list) and len(f) > 0))
        return filled / len(fields)


class ContentExtractorLLM:
    """
    基于LLM的内容信息抽取器
    
    使用Ollama运行本地LLM模型进行5W2H信息抽取
    
    Attributes:
        client: Ollama客户端
        model: 使用的模型名称
        max_retries: 最大重试次数
    """
    
    def __init__(
        self,
        model: str = None,
        base_url: str = "http://localhost:11434",
        max_retries: int = 2,
        timeout: float = 60.0
    ):
        """
        初始化抽取器
        
        Args:
            model: 模型名称，默认使用配置中的extractor模型
            base_url: Ollama服务地址
            max_retries: 最大重试次数
            timeout: 请求超时时间
        """
        self.model = model or LLMSettings.EXTRACTOR_MODEL
        self.max_retries = max_retries
        
        config = OllamaConfig(
            base_url=base_url,
            timeout=timeout,
            default_model=self.model
        )
        self.client = OllamaClient(config)
        
        logger.info(f"ContentExtractorLLM初始化完成, 模型: {self.model}")
    
    def extract(
        self,
        title: str,
        content: str,
        source: str = "",
        publish_time: str = "",
        url: str = "",
        item_id: str = ""
    ) -> ExtractedInfo:
        """
        从单条内容中抽取5W2H信息
        
        Args:
            title: 标题
            content: 正文内容
            source: 来源
            publish_time: 发布时间
            url: 原文链接
            item_id: 条目ID
            
        Returns:
            ExtractedInfo: 抽取的结构化信息
        """
        result = ExtractedInfo(
            source_id=item_id,
            source_url=url,
            source_title=title,
            extraction_time=datetime.now().isoformat(),
            model_used=self.model
        )
        
        if not content and not title:
            logger.warning("内容和标题均为空，跳过抽取")
            return result
        
        # 构建提示词
        prompt_template = get_prompt(PromptType.EXTRACTION)
        user_prompt = prompt_template.format(
            title=title or "无标题",
            source=source or "未知来源",
            content=content[:3000] if content else title,  # 限制内容长度
            publish_time=publish_time or "未知时间"
        )
        
        # 调用LLM
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.generate(
                    prompt=user_prompt,
                    model=self.model,
                    temperature=LLMSettings.EXTRACTOR_TEMPERATURE,
                    max_tokens=LLMSettings.EXTRACTOR_MAX_TOKENS
                )
                
                if not response.success:
                    logger.warning(f"LLM调用失败 (尝试 {attempt + 1}): {response.error}")
                    continue
                
                result.raw_response = response.text
                
                # 解析JSON响应
                parsed = self._parse_response(response.text)
                if parsed:
                    result.what = parsed.get("what", "")
                    result.who = parsed.get("who", [])
                    if isinstance(result.who, str):
                        result.who = [result.who] if result.who else []
                    result.when = parsed.get("when", "")
                    result.where = parsed.get("where", "")
                    result.why = parsed.get("why", "")
                    result.how = parsed.get("how", "")
                    result.how_much = parsed.get("how_much", "")
                    result.key_points = parsed.get("key_points", [])
                    if isinstance(result.key_points, str):
                        result.key_points = [result.key_points] if result.key_points else []
                    result.summary = parsed.get("summary", "")[:100]  # 限制摘要长度
                    result.confidence = 0.8 if result.is_valid() else 0.3
                    
                    logger.debug(f"成功抽取: {title[:50]}...")
                    return result
                    
            except Exception as e:
                logger.error(f"抽取异常 (尝试 {attempt + 1}): {e}")
                if attempt == self.max_retries:
                    logger.error(traceback.format_exc())
        
        # 抽取失败，使用标题作为基本信息
        result.what = title
        result.summary = title[:50] if title else ""
        result.confidence = 0.1
        
        return result
    
    def _parse_response(self, text: str) -> Optional[Dict[str, Any]]:
        """
        解析LLM响应为JSON
        
        Args:
            text: LLM响应文本
            
        Returns:
            解析后的字典，失败返回None
        """
        if not text:
            return None
        
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取JSON块
        import re
        
        # 匹配 ```json ... ``` 或 ``` ... ```
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}'
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    # 清理可能的问题字符
                    clean_match = match.strip()
                    if not clean_match.startswith('{'):
                        continue
                    return json.loads(clean_match)
                except json.JSONDecodeError:
                    continue
        
        logger.warning(f"无法解析JSON响应: {text[:200]}...")
        return None
    
    def extract_batch(
        self,
        items: List[Dict[str, Any]],
        max_workers: int = 3,
        show_progress: bool = True
    ) -> List[ExtractedInfo]:
        """
        批量抽取信息
        
        Args:
            items: 待抽取的条目列表，每个条目应包含title, content等字段
            max_workers: 并行工作线程数
            show_progress: 是否显示进度
            
        Returns:
            抽取结果列表
        """
        results = []
        total = len(items)
        
        if total == 0:
            return results
        
        logger.info(f"开始批量抽取, 共{total}条, 并行数: {max_workers}")
        
        def extract_item(item: Dict[str, Any]) -> ExtractedInfo:
            return self.extract(
                title=item.get("title", ""),
                content=item.get("content", ""),
                source=item.get("source", ""),
                publish_time=item.get("publish_time", ""),
                url=item.get("url", ""),
                item_id=item.get("id", "")
            )
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {
                executor.submit(extract_item, item): i 
                for i, item in enumerate(items)
            }
            
            completed = 0
            for future in as_completed(future_to_item):
                idx = future_to_item[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                except Exception as e:
                    logger.error(f"批量抽取异常 (索引 {idx}): {e}")
                    results.append((idx, ExtractedInfo()))
                
                completed += 1
                if show_progress and completed % 10 == 0:
                    logger.info(f"抽取进度: {completed}/{total}")
        
        # 按原始顺序排序
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    def extract_with_entities(
        self,
        title: str,
        content: str,
        **kwargs
    ) -> ExtractedInfo:
        """
        抽取信息并识别命名实体
        
        Args:
            title: 标题
            content: 正文内容
            **kwargs: 其他参数传递给extract方法
            
        Returns:
            包含实体识别结果的ExtractedInfo
        """
        result = self.extract(title, content, **kwargs)
        
        # 简单的实体识别（基于who字段和规则）
        entities = {
            "persons": [],
            "organizations": [],
            "locations": [],
            "products": [],
            "amounts": []
        }
        
        # 从who字段提取
        for who in result.who:
            who_lower = who.lower()
            if any(kw in who_lower for kw in ["公司", "集团", "corp", "inc", "ltd", "有限"]):
                entities["organizations"].append(who)
            else:
                entities["persons"].append(who)
        
        # 从where字段提取
        if result.where:
            entities["locations"].append(result.where)
        
        # 从how_much字段提取金额
        if result.how_much:
            entities["amounts"].append(result.how_much)
        
        result.entities = {k: v for k, v in entities.items() if v}
        
        return result


def create_extractor(
    model: str = None,
    base_url: str = "http://localhost:11434"
) -> ContentExtractorLLM:
    """
    工厂函数: 创建抽取器实例
    
    Args:
        model: 模型名称
        base_url: Ollama服务地址
        
    Returns:
        ContentExtractorLLM实例
    """
    return ContentExtractorLLM(model=model, base_url=base_url)


# 测试代码
if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    
    # 测试用例
    test_content = """
    OpenAI今日宣布推出GPT-5模型，这是其最新一代大型语言模型。
    据悉，该模型在多项基准测试中超越了前代产品，参数量达到1万亿。
    OpenAI CEO Sam Altman在旧金山举行的发布会上表示，GPT-5将于下月向开发者开放API访问，
    定价为每百万token 30美元。业内分析人士认为，这将对AI行业产生深远影响。
    """
    
    print("=" * 60)
    print("5W2H信息抽取测试")
    print("=" * 60)
    
    extractor = create_extractor()
    
    result = extractor.extract(
        title="OpenAI发布GPT-5模型",
        content=test_content,
        source="Tech News",
        publish_time="2024-12-20"
    )
    
    print(f"\n抽取结果:")
    print(f"  What: {result.what}")
    print(f"  Who: {result.who}")
    print(f"  When: {result.when}")
    print(f"  Where: {result.where}")
    print(f"  Why: {result.why}")
    print(f"  How: {result.how}")
    print(f"  How Much: {result.how_much}")
    print(f"  Key Points: {result.key_points}")
    print(f"  Summary: {result.summary}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Completeness: {result.get_completeness_score():.2%}")

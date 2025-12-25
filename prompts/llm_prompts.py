"""
新闻聚合分析系统 - LLM Prompts 模块
News Aggregator & Analyzer System - LLM Prompts

包含所有 LLM 分析任务的详细 Prompts
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


# ============================================================
# 1. 分类 Prompt (使用小模型 qwen2.5:0.5b)
# ============================================================

CLASSIFICATION_SYSTEM_PROMPT = """你是一个专业的新闻内容分类助手。你的任务是对科技/财经新闻进行分类和重要性评估。

## 可用分类:
- ai_ml: 人工智能/机器学习相关
- programming: 编程/开发相关
- cloud_infra: 云计算/基础设施
- security: 网络安全
- data_science: 数据科学
- blockchain: 区块链/加密货币
- hardware: 硬件/芯片
- startup: 创业/融资
- big_tech: 科技巨头动态
- market: 市场/经济
- policy: 政策/法规
- research: 学术研究
- product: 产品发布
- other: 其他

## 重要性级别 (1-5):
- 5: 极其重要 - 重大突破、颠覆性事件
- 4: 高度重要 - 行业重大新闻
- 3: 中等重要 - 值得关注的信息
- 2: 较低重要 - 一般资讯
- 1: 最低重要 - 边缘信息

## 输出格式要求:
必须以 JSON 格式输出，包含以下字段:
- category: 分类标签
- importance: 重要性级别(1-5)
- confidence: 置信度(0-1)
- reason: 简短理由(不超过20字)"""


CLASSIFICATION_USER_PROMPT = """请对以下新闻内容进行分类和重要性评估:

标题: {title}
来源: {source}
内容摘要: {summary}

请以 JSON 格式输出分类结果:"""


CLASSIFICATION_OUTPUT_EXAMPLE = """{
  "category": "ai_ml",
  "importance": 4,
  "confidence": 0.85,
  "reason": "重大AI模型发布"
}"""


# ============================================================
# 2. 批量分类 Prompt
# ============================================================

BATCH_CLASSIFICATION_SYSTEM_PROMPT = """你是一个高效的新闻批量分类助手。请对多条新闻进行快速分类。

## 分类标签:
ai_ml|programming|cloud_infra|security|data_science|blockchain|hardware|startup|big_tech|market|policy|research|product|other

## 输出格式:
对于每条新闻，输出一行: ID|分类|重要性(1-5)
不要添加任何其他内容。"""


BATCH_CLASSIFICATION_USER_PROMPT = """请对以下新闻进行分类:

{news_list}

每条新闻输出一行: ID|分类|重要性"""


# ============================================================
# 3. 5W2H 信息提取 Prompt (使用大模型 qwen3:4b)
# ============================================================

EXTRACTION_5W2H_SYSTEM_PROMPT = """你是一个专业的新闻信息提取助手。你的任务是从新闻内容中提取 5W2H 结构化信息。

## 5W2H 框架:
- What (什么): 发生了什么事件/发布了什么产品/宣布了什么消息
- Who (谁): 涉及的公司/人物/组织
- When (何时): 时间信息(具体日期或相对时间)
- Where (何地): 地点信息(如适用)
- Why (为何): 原因/背景/动机
- How (如何): 方式/方法/过程
- How much (多少): 数量/金额/规模(如适用)

## 输出要求:
1. 必须以 JSON 格式输出
2. 如果某个字段无法从内容中提取，填写 null
3. 提取的信息要准确，不要推测
4. 使用与原文相同的语言输出

## JSON 结构:
{
  "what": "事件描述",
  "who": ["相关方1", "相关方2"],
  "when": "时间信息",
  "where": "地点信息",
  "why": "原因/背景",
  "how": "方式/过程",
  "how_much": "数量/金额",
  "key_points": ["要点1", "要点2", "要点3"],
  "summary": "一句话摘要(不超过50字)"
}"""


EXTRACTION_5W2H_USER_PROMPT = """请从以下新闻内容中提取 5W2H 结构化信息:

标题: {title}
来源: {source}
发布时间: {publish_time}

正文内容:
{content}

请以 JSON 格式输出提取结果:"""


# ============================================================
# 4. 评论分析 Prompt
# ============================================================

COMMENT_ANALYSIS_SYSTEM_PROMPT = """你是一个专业的评论分析助手。你的任务是分析新闻评论，提取有价值的观点和洞见。

## 分析维度:
1. 情感倾向: positive/neutral/negative
2. 观点类型: support/oppose/question/insight/humor
3. 专业程度: expert/informed/general
4. 价值程度: high/medium/low

## 输出要求:
1. 识别高价值评论(提供专业见解、独特视角、补充信息)
2. 提取主要讨论主题
3. 总结社区整体态度
4. 标记值得深入了解的观点

## JSON 结构:
{
  "overall_sentiment": "positive/neutral/negative",
  "main_topics": ["主题1", "主题2"],
  "valuable_comments": [
    {
      "author": "用户名",
      "content_summary": "观点摘要",
      "value_type": "insight/expert/question",
      "relevance": 0.9
    }
  ],
  "community_consensus": "社区共识描述",
  "controversies": ["争议点1", "争议点2"]
}"""


COMMENT_ANALYSIS_USER_PROMPT = """请分析以下新闻的评论:

新闻标题: {title}
新闻摘要: {summary}

评论列表:
{comments}

请以 JSON 格式输出分析结果:"""


# ============================================================
# 5. 内容摘要生成 Prompt
# ============================================================

SUMMARY_GENERATION_SYSTEM_PROMPT = """你是一个专业的新闻摘要生成助手。你的任务是生成简洁、准确的新闻摘要。

## 摘要要求:
1. 长度控制在 100-150 字
2. 包含核心信息(5W1H)
3. 使用客观中立的语言
4. 保持原文的语言风格
5. 突出最重要的信息点

## 输出格式:
直接输出摘要文本，不需要额外的格式。"""


SUMMARY_GENERATION_USER_PROMPT = """请为以下新闻生成摘要:

标题: {title}
正文:
{content}

请生成 100-150 字的摘要:"""


# ============================================================
# 6. 趋势分析 Prompt
# ============================================================

TREND_ANALYSIS_SYSTEM_PROMPT = """你是一个专业的科技趋势分析师。你的任务是从一组新闻中识别趋势和模式。

## 分析维度:
1. 热点话题: 高频出现的主题
2. 新兴趋势: 新出现或快速增长的话题
3. 行业动态: 各行业的主要变化
4. 重要事件: 值得特别关注的事件
5. 预测展望: 基于当前趋势的短期预测

## 输出要求:
1. 基于证据进行分析，引用具体新闻
2. 区分事实和推测
3. 提供可操作的洞察
4. 使用数据支持结论

## JSON 结构:
{
  "hot_topics": [
    {"topic": "话题名", "count": 10, "trend": "rising/stable/declining"}
  ],
  "emerging_trends": [
    {"trend": "趋势描述", "evidence": ["证据1", "证据2"]}
  ],
  "industry_dynamics": {
    "ai_ml": "行业动态描述",
    "startup": "行业动态描述"
  },
  "key_events": [
    {"event": "事件描述", "importance": 5, "source": "来源"}
  ],
  "outlook": "短期展望描述",
  "analysis_period": "分析时间范围"
}"""


TREND_ANALYSIS_USER_PROMPT = """请分析以下新闻集合，识别趋势和模式:

时间范围: {time_range}
新闻数量: {count}

新闻列表:
{news_list}

请以 JSON 格式输出趋势分析结果:"""


# ============================================================
# 7. 报告生成 Prompt
# ============================================================

REPORT_GENERATION_SYSTEM_PROMPT = """你是一个专业的科技新闻报告撰写助手。你的任务是基于分析结果生成结构化的洞察报告。

## 报告结构:
1. 执行摘要: 核心发现和结论
2. 热点追踪: 当前热门话题
3. 重要新闻: 高价值新闻详解
4. 行业洞察: 各领域动态分析
5. 趋势预测: 短期发展预测
6. 建议关注: 值得持续关注的方向

## 写作要求:
1. 语言专业但易读
2. 结论有数据支撑
3. 提供可操作建议
4. 控制适当篇幅

## 格式要求:
使用 Markdown 格式，包含标题、列表、引用等元素。"""


REPORT_GENERATION_USER_PROMPT = """请基于以下分析数据生成洞察报告:

分析时间: {analysis_time}
数据来源: {sources}
新闻总数: {total_count}

分类统计:
{category_stats}

重要新闻:
{important_news}

趋势分析:
{trend_analysis}

请生成完整的洞察报告 (Markdown 格式):"""


# ============================================================
# 8. Prompt 模板管理
# ============================================================

class PromptType(str, Enum):
    """Prompt 类型枚举"""
    CLASSIFICATION = "classification"
    BATCH_CLASSIFICATION = "batch_classification"
    EXTRACTION_5W2H = "extraction_5w2h"
    COMMENT_ANALYSIS = "comment_analysis"
    SUMMARY_GENERATION = "summary_generation"
    TREND_ANALYSIS = "trend_analysis"
    REPORT_GENERATION = "report_generation"


@dataclass
class PromptTemplate:
    """Prompt 模板"""
    prompt_type: PromptType
    system_prompt: str
    user_prompt: str
    example_output: Optional[str] = None
    model_preference: str = "default"  # classifier/extractor


# Prompt 模板注册表
PROMPT_TEMPLATES: Dict[PromptType, PromptTemplate] = {
    PromptType.CLASSIFICATION: PromptTemplate(
        prompt_type=PromptType.CLASSIFICATION,
        system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
        user_prompt=CLASSIFICATION_USER_PROMPT,
        example_output=CLASSIFICATION_OUTPUT_EXAMPLE,
        model_preference="classifier"
    ),
    PromptType.BATCH_CLASSIFICATION: PromptTemplate(
        prompt_type=PromptType.BATCH_CLASSIFICATION,
        system_prompt=BATCH_CLASSIFICATION_SYSTEM_PROMPT,
        user_prompt=BATCH_CLASSIFICATION_USER_PROMPT,
        model_preference="classifier"
    ),
    PromptType.EXTRACTION_5W2H: PromptTemplate(
        prompt_type=PromptType.EXTRACTION_5W2H,
        system_prompt=EXTRACTION_5W2H_SYSTEM_PROMPT,
        user_prompt=EXTRACTION_5W2H_USER_PROMPT,
        model_preference="extractor"
    ),
    PromptType.COMMENT_ANALYSIS: PromptTemplate(
        prompt_type=PromptType.COMMENT_ANALYSIS,
        system_prompt=COMMENT_ANALYSIS_SYSTEM_PROMPT,
        user_prompt=COMMENT_ANALYSIS_USER_PROMPT,
        model_preference="extractor"
    ),
    PromptType.SUMMARY_GENERATION: PromptTemplate(
        prompt_type=PromptType.SUMMARY_GENERATION,
        system_prompt=SUMMARY_GENERATION_SYSTEM_PROMPT,
        user_prompt=SUMMARY_GENERATION_USER_PROMPT,
        model_preference="extractor"
    ),
    PromptType.TREND_ANALYSIS: PromptTemplate(
        prompt_type=PromptType.TREND_ANALYSIS,
        system_prompt=TREND_ANALYSIS_SYSTEM_PROMPT,
        user_prompt=TREND_ANALYSIS_USER_PROMPT,
        model_preference="extractor"
    ),
    PromptType.REPORT_GENERATION: PromptTemplate(
        prompt_type=PromptType.REPORT_GENERATION,
        system_prompt=REPORT_GENERATION_SYSTEM_PROMPT,
        user_prompt=REPORT_GENERATION_USER_PROMPT,
        model_preference="extractor"
    ),
}


def get_prompt_template(prompt_type: PromptType) -> PromptTemplate:
    """获取 Prompt 模板"""
    return PROMPT_TEMPLATES.get(prompt_type)


def get_prompt(prompt_type: PromptType) -> str:
    """
    获取指定类型的用户提示词模板
    
    Args:
        prompt_type: Prompt类型
        
    Returns:
        str: 用户提示词模板字符串
    """
    template = PROMPT_TEMPLATES.get(prompt_type)
    if template:
        return template.user_prompt
    raise ValueError(f"未知的Prompt类型: {prompt_type}")


def format_prompt(
    prompt_type: PromptType,
    **kwargs
) -> tuple[str, str]:
    """
    格式化 Prompt
    
    Args:
        prompt_type: Prompt 类型
        **kwargs: 格式化参数
        
    Returns:
        (system_prompt, formatted_user_prompt)
    """
    template = get_prompt_template(prompt_type)
    if not template:
        raise ValueError(f"未知的 Prompt 类型: {prompt_type}")
    
    formatted_user = template.user_prompt.format(**kwargs)
    return template.system_prompt, formatted_user


def format_classification_prompt(
    title: str,
    source: str,
    summary: str
) -> tuple[str, str]:
    """
    生成分类提示词
    
    Args:
        title: 标题
        source: 来源
        summary: 摘要
        
    Returns:
        (system_prompt, user_prompt)
    """
    return format_prompt(
        PromptType.CLASSIFICATION,
        title=title,
        source=source,
        summary=summary[:500]
    )


def format_extraction_prompt(
    title: str,
    source: str,
    content: str,
    publish_time: str = ""
) -> tuple[str, str]:
    """
    生成5W2H抽取提示词
    
    Args:
        title: 标题
        source: 来源
        content: 正文内容
        publish_time: 发布时间
        
    Returns:
        (system_prompt, user_prompt)
    """
    return format_prompt(
        PromptType.EXTRACTION_5W2H,
        title=title,
        source=source,
        content=content[:2000],
        publish_time=publish_time or "未知"
    )


# ============================================================
# 9. 特定场景 Prompt 生成器
# ============================================================

class PromptGenerator:
    """Prompt 生成器"""
    
    @staticmethod
    def classification_prompt(
        title: str,
        source: str,
        summary: str
    ) -> tuple[str, str]:
        """生成分类 Prompt"""
        return format_prompt(
            PromptType.CLASSIFICATION,
            title=title,
            source=source,
            summary=summary[:500]  # 限制长度
        )
    
    @staticmethod
    def extraction_prompt(
        title: str,
        source: str,
        content: str,
        publish_time: str = ""
    ) -> tuple[str, str]:
        """生成 5W2H 提取 Prompt"""
        return format_prompt(
            PromptType.EXTRACTION_5W2H,
            title=title,
            source=source,
            content=content[:2000],  # 限制长度
            publish_time=publish_time
        )
    
    @staticmethod
    def batch_classification_prompt(
        news_items: List[Dict]
    ) -> tuple[str, str]:
        """生成批量分类 Prompt"""
        news_list = "\n".join([
            f"{i+1}. [{item.get('source', 'Unknown')}] {item.get('title', '')[:100]}"
            for i, item in enumerate(news_items[:20])  # 最多20条
        ])
        return format_prompt(
            PromptType.BATCH_CLASSIFICATION,
            news_list=news_list
        )
    
    @staticmethod
    def comment_analysis_prompt(
        title: str,
        summary: str,
        comments: List[Dict]
    ) -> tuple[str, str]:
        """生成评论分析 Prompt"""
        comments_text = "\n".join([
            f"- [{c.get('author', 'anonymous')}]: {c.get('text', '')[:200]}"
            for c in comments[:30]  # 最多30条评论
        ])
        return format_prompt(
            PromptType.COMMENT_ANALYSIS,
            title=title,
            summary=summary[:300],
            comments=comments_text
        )


if __name__ == "__main__":
    # 测试 Prompt 生成
    print("=== 分类 Prompt 示例 ===")
    sys_prompt, user_prompt = PromptGenerator.classification_prompt(
        title="OpenAI 发布 GPT-5",
        source="TechCrunch",
        summary="OpenAI 今日宣布发布最新的大语言模型 GPT-5..."
    )
    print(f"System: {sys_prompt[:200]}...")
    print(f"User: {user_prompt}")
    
    print("\n=== 5W2H 提取 Prompt 示例 ===")
    sys_prompt, user_prompt = PromptGenerator.extraction_prompt(
        title="苹果发布新款 iPhone",
        source="The Verge",
        content="苹果公司今天在其秋季发布会上推出了新款 iPhone...",
        publish_time="2025-12-25"
    )
    print(f"System: {sys_prompt[:200]}...")
    print(f"User: {user_prompt}")
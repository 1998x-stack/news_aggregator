# 新闻聚合分析系统 (News Aggregator System)

一个全面的新闻聚合与智能分析系统，整合多数据源采集、LLM内容分析、趋势识别和报告生成。

## 功能特性

### 📊 多源数据采集
- **HackerNews**: 技术社区热点 (Official API + Algolia Search)
- **RSS订阅**: 20+科技媒体源 (TechCrunch, The Verge, Wired, arXiv等)
- **新浪财经直播**: 7x24小时金融新闻 (9个频道)

### 🤖 智能分析
- **内容分类**: 14个类别自动分类 (AI/ML, 云计算, 安全等)
- **重要性评估**: 5级重要性评分
- **5W2H信息抽取**: 结构化信息提取
- **趋势分析**: 热点话题、新兴趋势识别

### 📝 报告生成
- **Markdown报告**: 完整分析报告
- **JSON数据**: 结构化数据导出
- **HTML报告**: 可视化网页报告

## 系统架构

```

├── main.py                 # 主入口/流水线编排
├── requirements.txt        # 依赖列表
├── config/
│   ├── __init__.py
│   └── settings.py         # 系统配置
├── collectors/             # 数据采集模块
│   ├── __init__.py
│   ├── hackernews_collector.py
│   ├── rss_collector.py
│   └── sina_zhibo_collector.py
├── extractors/             # 内容抽取模块
│   ├── __init__.py
│   └── content_extractor.py
├── analyzers/              # 分析模块
│   ├── __init__.py
│   ├── classifier.py       # 内容分类
│   ├── extractor.py        # 5W2H抽取
│   ├── trend_analyzer.py   # 趋势分析
│   └── report_generator.py # 报告生成
├── prompts/                # LLM提示词
│   ├── __init__.py
│   └── llm_prompts.py
├── utils/                  # 工具模块
│   ├── __init__.py
│   ├── dashscope_client.py   # DashScope客户端
│   └── file_utils.py       # 文件工具
├── reports/                # API研究文档
├── outputs/                # 生成的报告
├── cache/                  # 缓存数据
└── logs/                   # 日志文件
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 配置DashScope API Key (用于LLM分析)
export DASHSCOPE_API_KEY="your-dashscope-api-key"
# 可选：指定模型 (默认: qwen-max)
export DASHSCOPE_MODEL="qwen-max"
```

### 2. 运行系统

```bash
# 完整流水线
python main.py

# 仅采集数据
python main.py --collect-only

# 使用缓存数据分析
python main.py --analyze-only

# 指定数据源
python main.py --sources hn,rss

# 禁用LLM（使用规则分类）
python main.py --no-llm

# 调试模式
python main.py --debug
```

### 3. 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--collect-only` | 仅采集数据 | - |
| `--analyze-only` | 仅分析（使用缓存） | - |
| `--sources` | 数据源 (hn,rss,sina) | all |
| `--no-llm` | 禁用LLM分析 | - |
| `--formats` | 报告格式 | markdown,json |
| `--output-dir` | 输出目录 | outputs/ |
| `--debug` | 调试模式 | - |

## 配置说明

### 内容类别 (ContentCategory)

| 类别 | 说明 |
|------|------|
| ai_ml | 人工智能/机器学习 |
| programming | 编程开发 |
| cloud_infra | 云计算/基础设施 |
| security | 网络安全 |
| data_science | 数据科学 |
| blockchain | 区块链/加密货币 |
| hardware | 硬件/芯片 |
| startup | 创业/融资 |
| big_tech | 大厂动态 |
| market | 市场/股票 |
| policy | 政策/监管 |
| research | 学术研究 |
| product | 产品发布 |
| other | 其他 |

### 重要性等级

| 等级 | 标签 | 说明 |
|------|------|------|
| 5 | 关键 | 突破性进展/重大政策 |
| 4 | 重要 | 行业重大新闻 |
| 3 | 中等 | 值得关注 |
| 2 | 一般 | 常规信息 |
| 1 | 边缘 | 边缘内容 |

### LLM模型配置

```python
# config/settings.py
class LLMSettings:
    CLASSIFIER_MODEL = "qwen2.5:0.5b"  # 分类模型（快速）
    EXTRACTOR_MODEL = "qwen3:4b"     # 抽取模型（深度）
    
    CLASSIFIER_TEMPERATURE = 0.1       # 低温度，稳定输出
    EXTRACTOR_TEMPERATURE = 0.3        # 略高，允许创造性
```

## API参考

### 数据源API

#### HackerNews
- Official API: `https://hacker-news.firebaseio.com/v0/`
- Algolia API: `https://hn.algolia.com/api/v1/`
- 无需认证，无速率限制

#### RSS源
- 支持RSS 2.0和Atom格式
- 使用feedparser解析
- 20+预配置源

#### 新浪财经直播
- Endpoint: `https://zhibo.sina.com.cn/api/zhibo/feed`
- 需要Referer头
- JSONP响应格式

## 输出示例

### Markdown报告
```markdown
# 新闻聚合分析报告

**日期**: 2024-12-20
**分析文章数**: 130

## 热点话题

| 排名 | 话题 | 热度 | 提及次数 |
|------|------|------|----------|
| 1 | GPT-5 | 🔥🔥🔥🔥🔥 95 | 15 |
| 2 | 量子计算 | 🔥🔥🔥🔥 80 | 10 |
...
```

### JSON数据
```json
{
  "metadata": {
    "date": "2024-12-20",
    "total_articles": 130
  },
  "trend_analysis": {
    "hot_topics": [...],
    "emerging_trends": [...],
    "industry_dynamics": [...]
  },
  "articles": [...]
}
```

## 开发指南

### 添加新数据源

1. 在`collectors/`创建新采集器
2. 实现`fetch_*`方法
3. 在`main.py`中注册

### 自定义分析

1. 在`prompts/llm_prompts.py`添加提示词
2. 在`analyzers/`创建分析器
3. 集成到流水线

### 扩展报告格式

1. 在`ReportGenerator`添加生成方法
2. 注册到`generate_daily_report`

## 注意事项

1. **DashScope API Key**: LLM分析需要配置DashScope API Key
2. **网络访问**: 数据采集需要访问外部API
3. **存储空间**: 缓存和日志会占用磁盘空间
4. **API限制**: 部分源可能有速率限制

## 许可证

MIT License

## 作者

Claude - AI Assistant by Anthropic

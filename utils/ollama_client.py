#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama API 客户端模块

提供与Ollama本地LLM服务的交互功能，支持:
- 模型管理（列表、拉取、删除）
- 文本生成（同步/流式）
- JSON结构化输出
- 批量处理
- 错误重试机制

依赖:
    - requests: HTTP客户端
    - loguru: 日志记录

作者: News Aggregator System
创建日期: 2025-12-25
"""

import json
import time
import traceback
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Generator, Callable
from enum import Enum

import requests
from loguru import logger


class OllamaModel(Enum):
    """Ollama模型枚举"""
    QWEN_0_5B = "qwen2.5:0.5b"  # 快速分类模型
    QWEN_3B = "qwen3:4b"      # 深度提取模型
    QWEN_7B = "qwen2.5:7b"      # 高质量模型（可选）
    LLAMA_3_2_1B = "llama3.2:1b"  # 备选轻量模型
    LLAMA_3_2_3B = "llama3.2:3b"  # 备选中等模型


@dataclass
class OllamaConfig:
    """Ollama配置类"""
    base_url: str = "http://localhost:11434"
    timeout: int = 120  # 请求超时时间（秒）
    max_retries: int = 3  # 最大重试次数
    retry_delay: float = 1.0  # 重试间隔（秒）
    default_model: str = OllamaModel.QWEN_3B.value
    
    # 生成参数默认值
    default_temperature: float = 0.3
    default_max_tokens: int = 2048
    default_top_p: float = 0.9
    default_top_k: int = 40
    

@dataclass
class GenerationResult:
    """生成结果数据类"""
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "text": self.text,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_duration_ms": self.total_duration_ms,
            "success": self.success,
            "error": self.error
        }


class OllamaClient:
    """
    Ollama API客户端
    
    提供与本地Ollama服务交互的完整功能集
    
    使用示例:
        >>> client = OllamaClient()
        >>> result = client.generate("你好，请介绍一下自己")
        >>> print(result.text)
    """
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        """
        初始化Ollama客户端
        
        Args:
            config: Ollama配置对象，为None时使用默认配置
        """
        self.config = config or OllamaConfig()
        self._session = requests.Session()
        logger.info(f"OllamaClient初始化完成，服务地址: {self.config.base_url}")
    
    def _make_request(
        self,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> requests.Response:
        """
        发送HTTP请求到Ollama API
        
        Args:
            endpoint: API端点路径
            method: HTTP方法
            data: 请求数据
            stream: 是否使用流式传输
            
        Returns:
            requests.Response: 响应对象
            
        Raises:
            requests.RequestException: 请求异常
        """
        url = f"{self.config.base_url}{endpoint}"
        
        for attempt in range(self.config.max_retries):
            try:
                if method == "POST":
                    response = self._session.post(
                        url,
                        json=data,
                        timeout=self.config.timeout,
                        stream=stream
                    )
                elif method == "GET":
                    response = self._session.get(
                        url,
                        timeout=self.config.timeout
                    )
                elif method == "DELETE":
                    response = self._session.delete(
                        url,
                        json=data,
                        timeout=self.config.timeout
                    )
                else:
                    raise ValueError(f"不支持的HTTP方法: {method}")
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise
    
    def check_health(self) -> bool:
        """
        检查Ollama服务健康状态
        
        Returns:
            bool: 服务是否可用
        """
        try:
            response = self._make_request("/api/tags", method="GET")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        获取已安装的模型列表
        
        Returns:
            List[Dict]: 模型信息列表
        """
        try:
            response = self._make_request("/api/tags", method="GET")
            data = response.json()
            models = data.get("models", [])
            logger.debug(f"获取到 {len(models)} 个已安装模型")
            return models
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            traceback.print_exc(file=sys.stderr)
            return []
    
    def model_exists(self, model_name: str) -> bool:
        """
        检查模型是否已安装
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 模型是否存在
        """
        models = self.list_models()
        return any(m.get("name", "").startswith(model_name.split(":")[0]) 
                   for m in models)
    
    def pull_model(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> bool:
        """
        拉取/下载模型
        
        Args:
            model_name: 模型名称
            progress_callback: 进度回调函数
            
        Returns:
            bool: 是否成功
        """
        logger.info(f"开始拉取模型: {model_name}")
        
        try:
            response = self._make_request(
                "/api/pull",
                data={"name": model_name},
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    try:
                        status = json.loads(line)
                        if progress_callback:
                            progress_callback(status)
                        
                        if status.get("status") == "success":
                            logger.info(f"模型 {model_name} 拉取成功")
                            return True
                        
                        # 显示下载进度
                        if "completed" in status and "total" in status:
                            progress = status["completed"] / status["total"] * 100
                            logger.debug(f"下载进度: {progress:.1f}%")
                            
                    except json.JSONDecodeError:
                        continue
            
            return True
            
        except Exception as e:
            logger.error(f"拉取模型失败: {e}")
            traceback.print_exc(file=sys.stderr)
            return False
    
    def delete_model(self, model_name: str) -> bool:
        """
        删除模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 是否成功
        """
        try:
            self._make_request(
                "/api/delete",
                method="DELETE",
                data={"name": model_name}
            )
            logger.info(f"模型 {model_name} 已删除")
            return True
        except Exception as e:
            logger.error(f"删除模型失败: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None,
        format_json: bool = False
    ) -> GenerationResult:
        """
        生成文本（同步方式）
        
        Args:
            prompt: 用户提示词
            model: 模型名称，为None时使用默认模型
            system: 系统提示词
            temperature: 温度参数（0-1）
            max_tokens: 最大生成token数
            top_p: nucleus采样参数
            top_k: top-k采样参数
            stop: 停止词列表
            format_json: 是否要求JSON格式输出
            
        Returns:
            GenerationResult: 生成结果
        """
        model = model or self.config.default_model
        
        # 构建请求数据
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.default_temperature,
                "num_predict": max_tokens or self.config.default_max_tokens,
                "top_p": top_p or self.config.default_top_p,
                "top_k": top_k or self.config.default_top_k,
            }
        }
        
        if system:
            data["system"] = system
        
        if stop:
            data["options"]["stop"] = stop
        
        if format_json:
            data["format"] = "json"
        
        logger.debug(f"生成请求 - 模型: {model}, prompt长度: {len(prompt)}")
        
        try:
            start_time = time.time()
            response = self._make_request("/api/generate", data=data)
            duration = (time.time() - start_time) * 1000
            
            result = response.json()
            
            return GenerationResult(
                text=result.get("response", ""),
                model=model,
                prompt_tokens=result.get("prompt_eval_count", 0),
                completion_tokens=result.get("eval_count", 0),
                total_duration_ms=duration,
                success=True,
                raw_response=result
            )
            
        except Exception as e:
            logger.error(f"生成失败: {e}")
            traceback.print_exc(file=sys.stderr)
            return GenerationResult(
                text="",
                model=model,
                success=False,
                error=str(e)
            )
    
    def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Generator[str, None, None]:
        """
        流式生成文本
        
        Args:
            prompt: 用户提示词
            model: 模型名称
            system: 系统提示词
            temperature: 温度参数
            max_tokens: 最大生成token数
            
        Yields:
            str: 生成的文本片段
        """
        model = model or self.config.default_model
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature or self.config.default_temperature,
                "num_predict": max_tokens or self.config.default_max_tokens,
            }
        }
        
        if system:
            data["system"] = system
        
        try:
            response = self._make_request("/api/generate", data=data, stream=True)
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            yield chunk["response"]
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"流式生成失败: {e}")
            traceback.print_exc(file=sys.stderr)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        format_json: bool = False
    ) -> GenerationResult:
        """
        多轮对话生成
        
        Args:
            messages: 消息列表，格式为 [{"role": "user/assistant/system", "content": "..."}]
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成token数
            format_json: 是否要求JSON格式输出
            
        Returns:
            GenerationResult: 生成结果
        """
        model = model or self.config.default_model
        
        data = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.default_temperature,
                "num_predict": max_tokens or self.config.default_max_tokens,
            }
        }
        
        if format_json:
            data["format"] = "json"
        
        try:
            start_time = time.time()
            response = self._make_request("/api/chat", data=data)
            duration = (time.time() - start_time) * 1000
            
            result = response.json()
            message = result.get("message", {})
            
            return GenerationResult(
                text=message.get("content", ""),
                model=model,
                prompt_tokens=result.get("prompt_eval_count", 0),
                completion_tokens=result.get("eval_count", 0),
                total_duration_ms=duration,
                success=True,
                raw_response=result
            )
            
        except Exception as e:
            logger.error(f"对话生成失败: {e}")
            traceback.print_exc(file=sys.stderr)
            return GenerationResult(
                text="",
                model=model,
                success=False,
                error=str(e)
            )
    
    def generate_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        生成JSON结构化输出
        
        Args:
            prompt: 用户提示词
            model: 模型名称
            system: 系统提示词
            temperature: 温度参数
            max_tokens: 最大生成token数
            
        Returns:
            Dict: 解析后的JSON对象，失败时返回空字典
        """
        result = self.generate(
            prompt=prompt,
            model=model,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            format_json=True
        )
        
        if not result.success:
            return {}
        
        try:
            # 尝试直接解析
            return json.loads(result.text)
        except json.JSONDecodeError:
            # 尝试提取JSON块
            text = result.text
            start_markers = ["{", "["]
            end_markers = ["}", "]"]
            
            for start, end in zip(start_markers, end_markers):
                start_idx = text.find(start)
                end_idx = text.rfind(end)
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    try:
                        json_str = text[start_idx:end_idx + 1]
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
            
            logger.warning(f"无法解析JSON输出: {result.text[:200]}")
            return {}
    
    def batch_generate(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        delay: float = 0.1
    ) -> List[GenerationResult]:
        """
        批量生成文本
        
        Args:
            prompts: 提示词列表
            model: 模型名称
            system: 系统提示词
            temperature: 温度参数
            max_tokens: 最大生成token数
            delay: 请求间隔（秒）
            
        Returns:
            List[GenerationResult]: 生成结果列表
        """
        results = []
        total = len(prompts)
        
        logger.info(f"开始批量生成，共 {total} 个请求")
        
        for i, prompt in enumerate(prompts):
            result = self.generate(
                prompt=prompt,
                model=model,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens
            )
            results.append(result)
            
            if i < total - 1 and delay > 0:
                time.sleep(delay)
            
            if (i + 1) % 10 == 0:
                logger.info(f"批量生成进度: {i + 1}/{total}")
        
        success_count = sum(1 for r in results if r.success)
        logger.info(f"批量生成完成，成功: {success_count}/{total}")
        
        return results
    
    def close(self):
        """关闭客户端连接"""
        self._session.close()
        logger.info("OllamaClient连接已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_ollama_client(
    base_url: str = "http://localhost:11434",
    timeout: int = 120,
    max_retries: int = 3
) -> OllamaClient:
    """
    工厂函数：创建Ollama客户端
    
    Args:
        base_url: Ollama服务地址
        timeout: 请求超时时间
        max_retries: 最大重试次数
        
    Returns:
        OllamaClient: 客户端实例
    """
    config = OllamaConfig(
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries
    )
    return OllamaClient(config)


# 单例客户端（可选使用）
_default_client: Optional[OllamaClient] = None

def get_default_client() -> OllamaClient:
    """
    获取默认的单例客户端
    
    Returns:
        OllamaClient: 默认客户端实例
    """
    global _default_client
    if _default_client is None:
        _default_client = create_ollama_client()
    return _default_client


def check_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """
    检查Ollama服务是否可用
    
    Args:
        base_url: Ollama服务地址
        
    Returns:
        bool: 服务是否可用
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.debug(f"Ollama服务不可用: {e}")
        return False


# 兼容性别名
OllamaResponse = GenerationResult


if __name__ == "__main__":
    # 测试代码
    logger.add(sys.stderr, level="DEBUG")
    
    client = create_ollama_client()
    
    # 检查服务状态
    print("=== 健康检查 ===")
    if client.check_health():
        print("Ollama服务正常运行")
    else:
        print("Ollama服务不可用，请确保已启动: ollama serve")
        sys.exit(1)
    
    # 列出模型
    print("\n=== 已安装模型 ===")
    models = client.list_models()
    for m in models:
        print(f"  - {m.get('name')}: {m.get('size', 0) / 1e9:.2f}GB")
    
    # 测试生成
    print("\n=== 文本生成测试 ===")
    result = client.generate(
        prompt="请用一句话介绍人工智能",
        temperature=0.5,
        max_tokens=100
    )
    print(f"生成结果: {result.text}")
    print(f"耗时: {result.total_duration_ms:.0f}ms")
    
    # 测试JSON生成
    print("\n=== JSON生成测试 ===")
    json_result = client.generate_json(
        prompt="请生成一个包含name, age, city字段的JSON对象",
        system="你是一个JSON生成器，只输出有效的JSON格式"
    )
    print(f"JSON结果: {json.dumps(json_result, ensure_ascii=False, indent=2)}")
    
    client.close()
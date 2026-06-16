"""
LLM Provider（支持模型覆盖）
文档：可通过 config.yaml 配置子 Agent 使用不同的模型

支持多种 LLM 后端：
- Anthropic (Claude)
- OpenAI (GPT 系列)
- OpenRouter (多模型代理)
- 自定义 OpenAI 兼容端点
"""
import anthropic
from typing import List, Dict, Optional, Any
from config.settings import settings
from src.utils.logger import logger


class LLMProvider:
    """
    LLM 提供者

    支持模型覆盖：
    - 父 Agent 使用默认配置
    - 子 Agent 可通过 config.yaml 覆盖模型/provider/base_url/api_key
    """
    
    def __init__(
        self,
        model_override: Optional[str] = None,
        provider_override: Optional[str] = None,
        base_url_override: Optional[str] = None,
        api_key_override: Optional[str] = None,
    ):
        # 文档：子 Agent 可覆盖模型和 provider
        self.model = model_override or "claude-sonnet-4-5"
        self.provider = provider_override or "anthropic"
        
        # 初始化客户端
        self._client = self._init_client(
            base_url=base_url_override or settings.delegation.base_url,
            api_key=api_key_override or settings.delegation.api_key
        )
        
        logger.debug(f"LLMProvider 初始化: model={self.model}, provider={self.provider}")
    
    def _init_client(self, base_url=None, api_key=None):
        """初始化对应的 LLM 客户端"""
        if self.provider == "anthropic":
            kwargs = {"api_key": settings.anthropic_api_key}
            if base_url:
                kwargs["base_url"] = base_url
            if api_key:
                kwargs["api_key"] = api_key
            return anthropic.Anthropic(**kwargs)
        
        elif self.provider in ("openai", "openrouter"):
            from openai import OpenAI
            kwargs = {"api_key": api_key or settings.openai_api_key}
            if self.provider == "openrouter":
                kwargs["base_url"] = "https://openrouter.ai/api/v1"
            if base_url:
                kwargs["base_url"] = base_url
            return OpenAI(**kwargs)
        
        # 自定义端点（文档：base_url + api_key）
        elif base_url:
            from openai import OpenAI
            return OpenAI(
                base_url=base_url,
                api_key=api_key or "local-key"
            )
        
        raise ValueError(f"不支持的 provider: {self.provider}")
    
    def chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        system: str = "",
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        发送聊天请求到 LLM

        Args:
            messages: 对话消息列表
            tools: 工具定义列表（可选）
            system: 系统提示词
            max_tokens: 最大输出 token 数

        Returns:
            统一格式的响应字典，包含 content, stop_reason, usage
        """
        if self.provider == "anthropic":
            return self._anthropic_chat(messages, tools, system, max_tokens)
        else:
            return self._openai_chat(messages, tools, system, max_tokens)
    
    def _anthropic_chat(self, messages, tools, system, max_tokens):
        """Anthropic API 调用"""
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        
        response = self._client.messages.create(**kwargs)
        return {
            "content": response.content,
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        }
    
    def _openai_chat(self, messages, tools, system, max_tokens):
        """OpenAI / OpenRouter / 自定义端点 API 调用"""
        if system:
            messages = [{"role": "system", "content": system}] + messages
        
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = [{
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"]
                }
            } for t in tools]
        
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        return {
            "content": choice.message.content or "",
            "stop_reason": choice.finish_reason,
            "tool_calls": getattr(choice.message, "tool_calls", None),
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        }
    
    def stream_chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        system: str = "",
        max_tokens: int = 4096,
    ):
        """
        流式聊天请求（用于实时输出）
        """
        if self.provider == "anthropic":
            yield from self._anthropic_stream(messages, tools, system, max_tokens)
        else:
            yield from self._openai_stream(messages, tools, system, max_tokens)
    
    def _anthropic_stream(self, messages, tools, system, max_tokens):
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        
        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield {"type": "text", "content": text}
    
    def _openai_stream(self, messages, tools, system, max_tokens):
        if system:
            messages = [{"role": "system", "content": system}] + messages
        
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = [{
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"]
                }
            } for t in tools]
        
        stream = self._client.chat.completions.create(**kwargs)
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield {"type": "text", "content": chunk.choices[0].delta.content}
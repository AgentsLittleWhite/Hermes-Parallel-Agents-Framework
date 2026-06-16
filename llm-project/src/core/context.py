"""
Agent 上下文隔离
每个 Agent 拥有完全独立的上下文，不对父 Agent 历史有任何了解。
"""
from typing import List, Dict, Optional


class AgentContext:
    """
    Agent 上下文隔离容器

    文档：
    - 子 Agent 对父 Agent 对话历史一无所知
    - 仅通过 goal 和 context 字段获取必要信息
    """
    
    def __init__(self, agent_id: str, depth: int = 0):
        self.agent_id = agent_id
        self.depth = depth
        self.messages: List[Dict] = []
        self.metadata: Dict = {}
    
    def add_message(self, role: str, content):
        self.messages.append({"role": role, "content": content})
    
    def clear(self):
        self.messages.clear()
        self.metadata.clear()
    
    @property
    def message_count(self) -> int:
        return len(self.messages)
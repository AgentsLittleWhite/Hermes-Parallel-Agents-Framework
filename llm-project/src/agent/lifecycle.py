"""
Agent 生命周期管理
"""
from enum import Enum
from typing import Optional
from src.utils.logger import logger


class AgentState(str, Enum):
    CREATED    = "created"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"
    INTERRUPTED = "interrupted"


class AgentLifecycle:
    """管理 Agent 生命周期状态"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.state = AgentState.CREATED
    
    def start(self):
        self.state = AgentState.RUNNING
        logger.info(f"Agent [{self.agent_id}] 启动")
    
    def complete(self):
        self.state = AgentState.COMPLETED
        logger.info(f"Agent [{self.agent_id}] 完成")
    
    def fail(self, error: str):
        self.state = AgentState.FAILED
        logger.error(f"Agent [{self.agent_id}] 失败: {error}")
    
    def interrupt(self):
        self.state = AgentState.INTERRUPTED
        logger.warning(f"Agent [{self.agent_id}] 被中断")
    
    def is_active(self) -> bool:
        return self.state == AgentState.RUNNING
"""
中断传播机制
文档：中断父 Agent 将中断所有活跃子 Agent
"""
import threading
from typing import Dict, Set
from src.utils.logger import logger


class InterruptManager:
    """
    线程安全的中断管理器

    文档规则：
    - 中断父 Agent → 中断所有活跃子 Agent
    - 每个子 Agent 拥有独立的中断事件
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._child_events: Dict[str, threading.Event] = {}
        self._parent_interrupted = threading.Event()
    
    def register_child(self, child_id: str) -> threading.Event:
        """注册子 Agent，返回其专属中断事件"""
        with self._lock:
            event = threading.Event()
            self._child_events[child_id] = event
            # 若父已被中断，立即触发
            if self._parent_interrupted.is_set():
                event.set()
            logger.debug(f"注册子 Agent 中断事件: {child_id}")
            return event
    
    def unregister_child(self, child_id: str):
        with self._lock:
            self._child_events.pop(child_id, None)
    
    def interrupt_all(self):
        """
        中断父 Agent → 传播到所有活跃子 Agent
        文档：中断传播
        """
        with self._lock:
            self._parent_interrupted.set()
            count = len(self._child_events)
            for event in self._child_events.values():
                event.set()
            logger.warning(f"🛑 中断传播：已中断 {count} 个子 Agent")
    
    def is_parent_interrupted(self) -> bool:
        return self._parent_interrupted.is_set()
    
    def reset(self):
        with self._lock:
            self._parent_interrupted.clear()
            self._child_events.clear()
    
    @property
    def active_children_count(self) -> int:
        with self._lock:
            return len(self._child_events)


# 全局中断管理器
interrupt_manager = InterruptManager()
"""
独立终端会话管理
每个子 Agent 拥有独立的终端会话 ID
"""
import uuid
from typing import Dict, Optional


class TerminalSession:
    """
    独立终端会话

    文档：每个子 Agent 拥有独立的终端会话
    """
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"session-{uuid.uuid4().hex[:8]}"
        self.command_history: list = []
    
    def record_command(self, command: str, output: str, exit_code: int):
        self.command_history.append({
            "command": command,
            "output": output[:500],
            "exit_code": exit_code
        })
    
    @property
    def history_count(self) -> int:
        return len(self.command_history)
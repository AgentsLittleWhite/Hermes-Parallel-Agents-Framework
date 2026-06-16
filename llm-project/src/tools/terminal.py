import subprocess
from src.tools.base import BaseTool, ToolResult


class RunCommandTool(BaseTool):
    name = "run_command"
    description = "在独立终端会话中执行 shell 命令"
    toolset = "terminal"
    
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id   # 每个子 Agent 独立会话
    
    def execute(self, command: str, timeout: int = 30) -> ToolResult:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd="./workspace"
            )
            output = result.stdout or result.stderr or "(无输出)"
            return ToolResult(
                success=result.returncode == 0,
                output=output[:8000],     # 限制输出长度
                error=result.stderr if result.returncode != 0 else None
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, output=None, error=f"命令超时 ({timeout}s)")
    
    def _get_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "要执行的 shell 命令"},
                "timeout": {"type": "integer", "description": "超时秒数", "default": 30}
            },
            "required": ["command"]
        }
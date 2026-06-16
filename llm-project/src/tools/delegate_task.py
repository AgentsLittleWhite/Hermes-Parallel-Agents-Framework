"""
delegate_task 工具

这是整个框架的核心工具，父 Agent 通过此工具触发子 Agent 委派。
文档：Agent 会根据任务复杂程度自动处理委托，无需明确要求。
"""
import json
from typing import Optional
from src.tools.base import BaseTool, ToolResult
from src.delegation.models import DelegationRequest, BatchTaskItem
from src.delegation.executor import delegation_executor
from src.delegation.progress import TaskProgressTracker, console
from src.utils.logger import logger


class DelegateTaskTool(BaseTool):
    """
    delegate_task 工具

    支持两种调用形式（严格对应文档）：

    1. 单任务：
       delegate_task(
           goal="...",
           context="...",
           toolsets=["terminal", "file"],
           max_iterations=50
       )

    2. 批量并行（最多 3 个）：
       delegate_task(tasks=[
           {"goal": "...", "context": "...", "toolsets": ["web"]},
           ...
       ])
    """
    name = "delegate_task"
    description = """启动子 Agent 实例执行任务。
子 Agent 拥有完全隔离的上下文、受限工具集和独立终端会话。
只有最终摘要会进入当前上下文，保持 token 效率。

支持单任务（goal=...）或批量并行任务（tasks=[...]），最多 3 个并发。"""
    toolset = "core"    # core 工具：父 Agent 可用，子 Agent 禁用
    
    def execute(
        self,
        goal: Optional[str] = None,
        context: str = "",
        toolsets: list = None,
        max_iterations: int = 50,
        tasks: list = None,
    ) -> ToolResult:
        
        try:
            # 构建请求
            request = DelegationRequest(
                goal=goal,
                context=context,
                toolsets=toolsets or [],
                max_iterations=max_iterations,
                tasks=[BatchTaskItem(**t) for t in tasks] if tasks else None
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"参数验证失败: {e}"
            )
        
        task_list = request.to_single_tasks()
        task_count = len(task_list)
        
        logger.info(f"🚀 delegate_task 启动，任务数: {task_count}")
        
        # 进度追踪（树形视图）
        tracker = TaskProgressTracker(task_count)
        for i, task in enumerate(task_list):
            tracker.set_goal(i, task.goal)
        
        def progress_cb(index, status, action, iteration):
            tracker.update(index, status, action, iteration)
        
        # 更新初始状态
        for i in range(task_count):
            tracker.update(i, "pending")
        for i in range(task_count):
            tracker.update(i, "running")
        
        # 执行
        with tracker:
            summaries = delegation_executor.execute(
                request=request,
                progress_callback=progress_cb
            )
        
        # 更新最终状态
        for s in summaries:
            tracker.update(s.task_index, s.status.value, "完成", s.iterations_used)
        
        # 组装返回给父 Agent 的摘要（文档：只有最终摘要进入父 Agent 上下文）
        summary_texts = [s.to_context_string() for s in summaries]
        combined = "\n\n---\n\n".join(summary_texts)
        
        success_count = sum(1 for s in summaries if s.status.value == "completed")
        
        return ToolResult(
            success=True,
            output=combined,
            metadata={
                "total_tasks": task_count,
                "completed": success_count,
                "failed": task_count - success_count,
                "summaries": [s.model_dump() for s in summaries]
            }
        )
    
    def _get_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "子 Agent 的目标（单任务模式）。必须包含子 Agent 完成任务所需的全部信息。"
                },
                "context": {
                    "type": "string",
                    "description": "任务的完整上下文。子 Agent 对父对话一无所知，必须在此提供所有必要信息。"
                },
                "toolsets": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["terminal", "file", "web"]},
                    "description": "允许子 Agent 使用的工具集。terminal=命令行, file=文件读写, web=网络搜索"
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "子 Agent 最大工具调用轮次（默认 50）",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 100
                },
                "tasks": {
                    "type": "array",
                    "description": "批量并行任务（最多 3 个）。与 goal 互斥。",
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "goal":           {"type": "string"},
                            "context":        {"type": "string"},
                            "toolsets":       {"type": "array", "items": {"type": "string"}},
                            "max_iterations": {"type": "integer", "default": 50}
                        },
                        "required": ["goal"]
                    }
                }
            }
        }
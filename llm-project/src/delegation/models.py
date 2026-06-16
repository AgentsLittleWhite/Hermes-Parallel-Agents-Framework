"""
严格按文档定义的核心数据模型
"""
from enum import Enum
from typing import Any, Optional, List, Dict
from pydantic import BaseModel, Field, model_validator
import uuid
from datetime import datetime


# ── 文档定义的工具集 ────────────────────────────────────────────
VALID_TOOLSETS = {"terminal", "file", "web"}

# 文档明确禁止子 Agent 使用的工具
FORBIDDEN_CHILD_TOOLS = frozenset({
    "delegate_task",    # 禁止递归委派（防止无限生成）
    "clarify",          # 子 Agent 无法与用户交互
    "memory",           # 无法写入共享持久记忆
    "code_execution",   # 子 Agent 应逐步推理
    "send_message",     # 无跨平台副作用
})

# 文档：最大并发子 Agent 数
MAX_CONCURRENT_CHILDREN = 3

# 文档：最大委派深度
MAX_DELEGATION_DEPTH = 2


class TaskStatus(str, Enum):
    PENDING     = "pending"
    RUNNING     = "running"
    COMPLETED   = "completed"
    FAILED      = "failed"
    INTERRUPTED = "interrupted"


class SingleTask(BaseModel):
    """
    单任务委派模型
    对应文档：delegate_task(goal=..., context=..., toolsets=...)
    """
    goal: str = Field(
        ...,
        description="子 Agent 的目标（必须自包含，子 Agent 对父上下文一无所知）"
    )
    context: str = Field(
        default="",
        description="子 Agent 所需的全部上下文信息"
    )
    toolsets: List[str] = Field(
        default_factory=list,
        description="允许的工具集：terminal / file / web"
    )
    max_iterations: int = Field(
        default=50,
        description="最大工具调用轮次（文档默认值：50）"
    )
    
    @model_validator(mode="after")
    def validate_toolsets(self):
        """验证工具集合法性"""
        invalid = set(self.toolsets) - VALID_TOOLSETS
        if invalid:
            raise ValueError(
                f"非法工具集: {invalid}，"
                f"合法值: {VALID_TOOLSETS}"
            )
        return self


class BatchTaskItem(BaseModel):
    """
    并行批量任务中的单项
    对应文档：tasks=[{goal, context, toolsets}, ...]
    """
    goal: str
    context: str = ""
    toolsets: List[str] = Field(default_factory=list)
    max_iterations: int = 50
    
    @model_validator(mode="after")
    def validate_toolsets(self):
        invalid = set(self.toolsets) - VALID_TOOLSETS
        if invalid:
            raise ValueError(f"非法工具集: {invalid}")
        return self


class DelegationRequest(BaseModel):
    """
    delegate_task 完整请求（单任务 OR 批量任务）
    """
    # 单任务字段
    goal: Optional[str]            = None
    context: str                   = ""
    toolsets: List[str]            = Field(default_factory=list)
    max_iterations: int            = 50
    
    # 批量任务字段
    tasks: Optional[List[BatchTaskItem]] = None
    
    @model_validator(mode="after")
    def validate_request(self):
        if self.goal is None and self.tasks is None:
            raise ValueError("必须提供 goal（单任务）或 tasks（批量任务）之一")
        if self.goal and self.tasks:
            raise ValueError("goal 和 tasks 不能同时提供")
        
        # 文档：超过 3 个任务截断为 3
        if self.tasks and len(self.tasks) > MAX_CONCURRENT_CHILDREN:
            self.tasks = self.tasks[:MAX_CONCURRENT_CHILDREN]
        
        return self
    
    def to_single_tasks(self) -> List[SingleTask]:
        """统一转换为单任务列表"""
        if self.goal:
            return [SingleTask(
                goal=self.goal,
                context=self.context,
                toolsets=self.toolsets or [],
                max_iterations=self.max_iterations
            )]
        return [
            SingleTask(
                goal=item.goal,
                context=item.context,
                toolsets=item.toolsets,
                max_iterations=item.max_iterations
            )
            for item in (self.tasks or [])
        ]


class SubAgentSummary(BaseModel):
    """
    子 Agent 执行结果摘要
    文档：只有最终摘要才会进入父 Agent 的上下文
    """
    task_index: int                           # 按输入顺序排序
    goal: str
    status: TaskStatus
    
    # 结构化摘要（文档要求）
    actions_taken: List[str]     = []        # 执行的操作
    findings: List[str]          = []        # 发现的内容
    modified_files: List[str]    = []        # 修改的文件
    issues_encountered: List[str]= []        # 遇到的问题
    final_answer: str            = ""        # 最终答案
    
    error: Optional[str]         = None
    iterations_used: int         = 0
    started_at: str              = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
    completed_at: Optional[str]  = None
    
    def to_context_string(self) -> str:
        """
        转换为进入父 Agent 上下文的摘要字符串
        保持 token 使用效率
        """
        lines = [
            f"## 子任务结果 [{self.task_index}]: {self.goal}",
            f"状态: {self.status.value} | 迭代次数: {self.iterations_used}",
        ]
        if self.actions_taken:
            lines.append(f"\n**执行操作:**")
            lines.extend(f"  - {a}" for a in self.actions_taken)
        if self.findings:
            lines.append(f"\n**发现内容:**")
            lines.extend(f"  - {f}" for f in self.findings)
        if self.modified_files:
            lines.append(f"\n**修改文件:** {', '.join(self.modified_files)}")
        if self.issues_encountered:
            lines.append(f"\n**遇到问题:**")
            lines.extend(f"  - {i}" for i in self.issues_encountered)
        if self.final_answer:
            lines.append(f"\n**结果:**\n{self.final_answer}")
        if self.error:
            lines.append(f"\n**错误:** {self.error}")
        return "\n".join(lines)

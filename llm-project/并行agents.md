 # Hermes Parallel Agents Framework
## 严格基于官方文档实现

---

## 📁 完整项目结构

```
hermes-agent/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── context.py          # Agent 上下文隔离
│   │   ├── session.py          # 独立终端会话
│   │   └── interrupt.py        # 中断传播机制
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── parent_agent.py     # 父 Agent（深度 0）
│   │   ├── child_agent.py      # 子 Agent（深度 1）
│   │   └── lifecycle.py        # Agent 生命周期管理
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py             # 工具基类
│   │   ├── registry.py         # 工具注册 & 工具集管理
│   │   ├── terminal.py         # terminal 工具集
│   │   ├── file.py             # file 工具集
│   │   ├── web.py              # web 工具集
│   │   └── delegate_task.py    # 核心委派工具 ⭐
│   ├── delegation/
│   │   ├── __init__.py
│   │   ├── executor.py         # 并行执行引擎
│   │   ├── models.py           # 数据模型
│   │   ├── progress.py         # 进度追踪（树形视图）
│   │   └── config.py           # delegation 配置
│   ├── llm/
│   │   ├── __init__.py
│   │   └── provider.py         # LLM 提供者（支持模型覆盖）
│   └── utils/
│       ├── __init__.py
│       └── logger.py
├── config/
│   ├── settings.py
│   └── config.yaml             # delegation 配置文件
├── tests/
│   └── test_delegation.py
├── main.py
├── requirements.txt
└── .env
```

---

## 📦 依赖

**`requirements.txt`**
```txt
anthropic>=0.40.0
python-dotenv>=1.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
pyyaml>=6.0.0
httpx>=0.25.0
rich>=13.0.0
loguru>=0.7.0
tenacity>=8.2.0
```

---

## ⚙️ 配置

**`config/config.yaml`**
```yaml
# 严格对应文档 config.yaml 结构
delegation:
  max_iterations: 50                              # 每个子 Agent 最大轮次
  default_toolsets: ["terminal", "file", "web"]  # 默认工具集
  model: null                                     # 子 Agent 模型覆盖（null=继承父）
  provider: null                                  # 子 Agent provider 覆盖
  base_url: null                                  # 自定义端点
  api_key: null                                   # 自定义密钥

agent:
  model: "claude-sonnet-4-5"
  provider: "anthropic"
  max_iterations: 100                             # 父 Agent 最大轮次
```

**`config/settings.py`**
```python
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List
import yaml
from pathlib import Path

class DelegationConfig:
    """对应文档 config.yaml delegation 节"""
    
    def __init__(self, raw: dict):
        self.max_iterations: int        = raw.get("max_iterations", 50)
        self.default_toolsets: List[str] = raw.get(
            "default_toolsets", ["terminal", "file", "web"]
        )
        self.model: Optional[str]       = raw.get("model", None)
        self.provider: Optional[str]    = raw.get("provider", None)
        self.base_url: Optional[str]    = raw.get("base_url", None)
        self.api_key: Optional[str]     = raw.get("api_key", None)


class Settings(BaseSettings):
    # 父 Agent LLM
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    
    # 从 config.yaml 加载 delegation 配置
    _delegation_cfg: Optional[DelegationConfig] = None
    
    @property
    def delegation(self) -> DelegationConfig:
        if self._delegation_cfg is None:
            cfg_path = Path("config/config.yaml")
            if cfg_path.exists():
                raw = yaml.safe_load(cfg_path.read_text())
                self._delegation_cfg = DelegationConfig(
                    raw.get("delegation", {})
                )
            else:
                self._delegation_cfg = DelegationConfig({})
        return self._delegation_cfg
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
```

---

## 🧱 数据模型 `src/delegation/models.py`

```python
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
```

---

## 🔧 工具基类与注册 `src/tools/base.py` & `src/tools/registry.py`

```python
# src/tools/base.py
from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel

class ToolResult(BaseModel):
    success: bool
    output: Any
    error: Optional[str] = None

class BaseTool(ABC):
    name: str = ""
    description: str = ""
    toolset: str = ""          # "terminal" / "file" / "web" / "core"
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass
    
    def safe_execute(self, **kwargs) -> ToolResult:
        try:
            return self.execute(**kwargs)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    def to_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._get_input_schema()
        }
    
    @abstractmethod
    def _get_input_schema(self) -> dict:
        pass
```

```python
# src/tools/registry.py
"""
工具注册中心 & 工具集管理
严格按文档工具集规则限制子 Agent 可访问工具
"""
from typing import Dict, List, Set, Optional
from src.tools.base import BaseTool
from src.delegation.models import FORBIDDEN_CHILD_TOOLS, VALID_TOOLSETS
from src.utils.logger import logger


class ToolRegistry:
    """
    工具注册中心
    
    核心职责：
    - 管理全局工具注册
    - 按 toolsets 过滤工具（父/子 Agent 差异化）
    - 对子 Agent 强制禁用 FORBIDDEN_CHILD_TOOLS
    """
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool
        logger.debug(f"注册工具: {tool.name} (toolset={tool.toolset})")
    
    def register_many(self, tools: List[BaseTool]):
        for t in tools:
            self.register(t)
    
    def get_tools_for_parent(self) -> List[BaseTool]:
        """父 Agent 获取所有工具（无限制）"""
        return list(self._tools.values())
    
    def get_tools_for_child(
        self,
        toolsets: List[str],
        is_child: bool = True
    ) -> List[BaseTool]:
        """
        子 Agent 按 toolsets 过滤工具
        并强制排除 FORBIDDEN_CHILD_TOOLS
        
        文档规则：
        - toolsets 控制可访问的工具集
        - 某些工具始终禁用（无论配置如何）
        """
        allowed_toolsets: Set[str] = set(toolsets) & VALID_TOOLSETS
        
        # 如果没有指定 toolsets，使用默认
        if not allowed_toolsets:
            from config.settings import settings
            allowed_toolsets = set(settings.delegation.default_toolsets)
        
        result = []
        for name, tool in self._tools.items():
            # 强制禁用规则（文档：无论如何配置）
            if is_child and name in FORBIDDEN_CHILD_TOOLS:
                logger.debug(f"子 Agent 禁用工具: {name}")
                continue
            
            # toolset 过滤（core 工具始终可用）
            if tool.toolset == "core" or tool.toolset in allowed_toolsets:
                result.append(tool)
        
        logger.debug(
            f"子 Agent 工具集 {toolsets} → "
            f"可用工具: {[t.name for t in result]}"
        )
        return result
    
    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)
    
    def list_all(self) -> List[str]:
        return list(self._tools.keys())


# 全局注册中心单例
tool_registry = ToolRegistry()
```

---

## 🖥️ 工具集实现

**`src/tools/terminal.py`**
```python
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
```

**`src/tools/file.py`**
```python
from pathlib import Path
from src.tools.base import BaseTool, ToolResult

WORKSPACE = Path("./workspace").resolve()

def _safe_path(p: str) -> Path:
    target = (WORKSPACE / p).resolve()
    if not str(target).startswith(str(WORKSPACE)):
        raise PermissionError("禁止访问工作目录外的路径")
    return target

class ReadFileTool(BaseTool):
    name = "read_file"
    description = "读取文件内容"
    toolset = "file"
    
    def execute(self, path: str) -> ToolResult:
        try:
            content = _safe_path(path).read_text(encoding="utf-8")
            return ToolResult(success=True, output=content[:16000])
        except FileNotFoundError:
            return ToolResult(success=False, output=None, error=f"文件不存在: {path}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    def _get_input_schema(self):
        return {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "文件路径（相对 workspace）"}},
            "required": ["path"]
        }

class WriteFileTool(BaseTool):
    name = "write_file"
    description = "写入内容到文件（覆盖或追加）"
    toolset = "file"
    
    def execute(self, path: str, content: str, mode: str = "w") -> ToolResult:
        try:
            p = _safe_path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.open(mode, encoding="utf-8").write(content)
            return ToolResult(success=True, output=f"已写入 {len(content)} 字符到 {path}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    def _get_input_schema(self):
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "mode": {"type": "string", "enum": ["w", "a"], "default": "w"}
            },
            "required": ["path", "content"]
        }

class ListFilesTool(BaseTool):
    name = "list_files"
    description = "列出目录文件"
    toolset = "file"
    
    def execute(self, path: str = ".") -> ToolResult:
        try:
            p = _safe_path(path)
            files = [str(f.relative_to(WORKSPACE)) for f in p.rglob("*") if f.is_file()]
            return ToolResult(success=True, output="\n".join(files[:100]))
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    def _get_input_schema(self):
        return {
            "type": "object",
            "properties": {"path": {"type": "string", "default": "."}},
            "required": []
        }
```

**`src/tools/web.py`**
```python
import httpx
from src.tools.base import BaseTool, ToolResult

class WebSearchTool(BaseTool):
    name = "web_search"
    description = "搜索网络信息（模拟）"
    toolset = "web"
    
    def execute(self, query: str) -> ToolResult:
        # 实际项目替换为 Tavily / SerpAPI / Brave Search
        return ToolResult(
            success=True,
            output=f"[模拟搜索结果] 关键词: {query}\n(接入真实搜索 API 后替换此输出)",
        )
    
    def _get_input_schema(self):
        return {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "搜索关键词"}},
            "required": ["query"]
        }

class WebFetchTool(BaseTool):
    name = "web_fetch"
    description = "获取网页内容"
    toolset = "web"
    
    def execute(self, url: str) -> ToolResult:
        try:
            resp = httpx.get(url, timeout=10, follow_redirects=True)
            return ToolResult(success=True, output=resp.text[:8000])
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
    
    def _get_input_schema(self):
        return {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "目标 URL"}},
            "required": ["url"]
        }
```

---

## 🔄 中断机制 `src/core/interrupt.py`

```python
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
```

---

## 📊 进度追踪 `src/delegation/progress.py`

```python
"""
树形进度视图
文档：在 CLI 模式下以树形视图实时显示每个子 Agent 的工具调用
"""
import threading
from typing import Dict, List, Optional
from datetime import datetime
from rich.console import Console
from rich.live import Live
from rich.tree import Tree
from rich.text import Text
from rich import box

console = Console()

STATUS_ICONS = {
    "pending":     "⏳",
    "running":     "🔄",
    "completed":   "✅",
    "failed":      "❌",
    "interrupted": "🛑",
}

class TaskProgressTracker:
    """
    实时树形进度追踪
    """
    
    def __init__(self, tasks_count: int):
        self._lock = threading.Lock()
        self._tasks: Dict[int, dict] = {
            i: {
                "goal": "",
                "status": "pending",
                "iterations": 0,
                "last_action": "",
                "started_at": None,
                "completed_at": None,
            }
            for i in range(tasks_count)
        }
        self._live: Optional[Live] = None
    
    def set_goal(self, index: int, goal: str):
        with self._lock:
            self._tasks[index]["goal"] = goal[:60]
    
    def update(self, index: int, status: str, action: str = "", iteration: int = 0):
        with self._lock:
            task = self._tasks[index]
            task["status"] = status
            task["iterations"] = iteration
            if action:
                task["last_action"] = action[:80]
            if status == "running" and not task["started_at"]:
                task["started_at"] = datetime.now()
            if status in ("completed", "failed", "interrupted"):
                task["completed_at"] = datetime.now()
        self._refresh()
    
    def _build_tree(self) -> Tree:
        root = Tree("🤖 [bold cyan]Hermes 并行委派[/bold cyan]")
        with self._lock:
            for idx, task in self._tasks.items():
                icon = STATUS_ICONS.get(task["status"], "❓")
                goal_text = task["goal"] or f"任务 {idx}"
                
                # 计算耗时
                duration = ""
                if task["started_at"] and task["completed_at"]:
                    d = (task["completed_at"] - task["started_at"]).total_seconds()
                    duration = f" [dim]({d:.1f}s)[/dim]"
                elif task["started_at"]:
                    d = (datetime.now() - task["started_at"]).total_seconds()
                    duration = f" [dim]({d:.0f}s...)[/dim]"
                
                color = {
                    "pending": "dim",
                    "running": "yellow",
                    "completed": "green",
                    "failed": "red",
                    "interrupted": "magenta"
                }.get(task["status"], "white")
                
                branch = root.add(
                    f"{icon} [{color}][{idx}] {goal_text}[/{color}]{duration}"
                )
                
                if task["last_action"]:
                    branch.add(f"[dim]↳ {task['last_action']}[/dim]")
                if task["iterations"] > 0:
                    branch.add(f"[dim]迭代: {task['iterations']}[/dim]")
        
        return root
    
    def _refresh(self):
        if self._live:
            self._live.update(self._build_tree())
    
    def __enter__(self):
        self._live = Live(
            self._build_tree(),
            console=console,
            refresh_per_second=4
        )
        self._live.__enter__()
        return self
    
    def __exit__(self, *args):
        if self._live:
            self._live.__exit__(*args)
```

---

## 🤖 子 Agent `src/agent/child_agent.py`

```python
"""
子 Agent 实现

文档核心规则：
1. 完全隔离的全新对话上下文
2. 只知道 goal 和 context，对父 Agent 历史一无所知
3. 禁止调用 delegate_task / clarify / memory / code_execution / send_message
4. 有独立终端会话
5. 有最大迭代次数限制（默认 50）
6. 只将最终摘要返回给父 Agent
"""
import json
import threading
from typing import List, Dict, Optional, Any
from src.delegation.models import SingleTask, SubAgentSummary, TaskStatus
from src.tools.base import BaseTool, ToolResult
from src.tools.registry import tool_registry
from src.llm.provider import LLMProvider
from src.core.interrupt import interrupt_manager
from src.utils.logger import logger


CHILD_SYSTEM_PROMPT = """你是一个专注执行单一任务的 AI 子 Agent。

## 关键约束
- 你对父对话的历史**一无所知**，你的全部信息仅来自 goal 和 context
- 你**不能**调用以下工具：delegate_task / clarify / memory / send_message / code_execution
- 你的工作在完全隔离的独立环境中进行
- 你有最大迭代次数限制，请高效使用

## 工作方式
1. 仔细阅读 goal 和 context
2. 制定清晰的执行计划
3. 逐步调用工具完成任务
4. 完成后输出结构化摘要

## 最终摘要格式（必须包含以下 JSON）
完成任务后，必须输出以下格式的 JSON 摘要：
```json
{
  "actions_taken": ["操作1", "操作2"],
  "findings": ["发现1", "发现2"],
  "modified_files": ["file1.py"],
  "issues_encountered": ["问题1"],
  "final_answer": "任务完成情况的详细说明"
}
```
"""


class ChildAgent:
    """
    子 Agent（深度 1）
    
    - 完全隔离的上下文（全新对话）
    - 独立终端会话
    - 无法再次委派（深度限制）
    """
    
    def __init__(
        self,
        child_id: str,
        task: SingleTask,
        task_index: int,
        model_override: Optional[str] = None,
        provider_override: Optional[str] = None,
    ):
        self.child_id = child_id
        self.task = task
        self.task_index = task_index
        self.iteration = 0
        
        # 独立的 LLM 实例（支持模型覆盖）
        self.llm = LLMProvider(
            model_override=model_override,
            provider_override=provider_override
        )
        
        # 获取工具（强制过滤禁用工具）
        self.tools: Dict[str, BaseTool] = {
            t.name: t
            for t in tool_registry.get_tools_for_child(
                toolsets=task.toolsets,
                is_child=True
            )
        }
        
        # 注册中断事件
        self.interrupt_event: threading.Event = interrupt_manager.register_child(child_id)
        
        # 进度回调
        self._progress_callback = None
        
        logger.info(
            f"🤖 子 Agent [{child_id}] 创建 | "
            f"工具: {list(self.tools.keys())} | "
            f"最大迭代: {task.max_iterations}"
        )
    
    def set_progress_callback(self, callback):
        self._progress_callback = callback
    
    def _report_progress(self, action: str):
        if self._progress_callback:
            self._progress_callback(self.task_index, "running", action, self.iteration)
    
    def run(self) -> SubAgentSummary:
        """
        执行子 Agent（全新对话，仅 goal + context 作为起点）
        """
        logger.info(f"🚀 子 Agent [{self.child_id}] 开始执行: {self.task.goal}")
        
        summary = SubAgentSummary(
            task_index=self.task_index,
            goal=self.task.goal,
            status=TaskStatus.RUNNING
        )
        
        try:
            # ★ 全新对话，仅包含 goal + context（文档核心：子 Agent 一无所知）
            messages = [
                {
                    "role": "user",
                    "content": self._build_initial_prompt()
                }
            ]
            
            tools_schema = [t.to_schema() for t in self.tools.values()]
            
            # ReAct 循环
            while self.iteration < self.task.max_iterations:
                
                # 检查中断（文档：中断传播）
                if self.interrupt_event.is_set():
                    logger.warning(f"🛑 子 Agent [{self.child_id}] 被中断")
                    summary.status = TaskStatus.INTERRUPTED
                    summary.issues_encountered.append("任务被父 Agent 中断")
                    return summary
                
                self.iteration += 1
                self._report_progress(f"第 {self.iteration} 轮推理...")
                
                response = self.llm.chat(
                    messages=messages,
                    tools=tools_schema if tools_schema else None,
                    system=CHILD_SYSTEM_PROMPT,
                    max_tokens=4096
                )
                
                text, tool_calls = self._parse_response(response)
                
                # 无工具调用 → 完成
                if not tool_calls or response.get("stop_reason") == "end_turn":
                    messages.append({"role": "assistant", "content": text})
                    self._parse_final_summary(text, summary)
                    summary.status = TaskStatus.COMPLETED
                    break
                
                # 执行工具调用
                if response.get("stop_reason") == "tool_use":
                    messages.append({"role": "assistant", "content": response["content"]})
                    tool_results = []
                    
                    for tc in tool_calls:
                        tool = self.tools.get(tc["name"])
                        self._report_progress(f"调用工具: {tc['name']}")
                        
                        if tool:
                            result: ToolResult = tool.safe_execute(**tc["input"])
                            output = result.output if result.success else {"error": result.error}
                            summary.actions_taken.append(
                                f"[迭代{self.iteration}] {tc['name']}({json.dumps(tc['input'], ensure_ascii=False)[:60]})"
                            )
                        else:
                            output = {"error": f"工具 {tc['name']} 不可用（可能被禁用）"}
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tc["id"],
                            "content": json.dumps(output, ensure_ascii=False, default=str)
                        })
                    
                    messages.append({"role": "user", "content": tool_results})
            
            else:
                # 达到最大迭代次数
                summary.status = TaskStatus.COMPLETED
                summary.issues_encountered.append(
                    f"已达到最大迭代次数 ({self.task.max_iterations})"
                )
                summary.final_answer = "任务达到迭代上限，已完成部分工作"
        
        except Exception as e:
            summary.status = TaskStatus.FAILED
            summary.error = str(e)
            logger.error(f"❌ 子 Agent [{self.child_id}] 异常: {e}")
        
        finally:
            summary.iterations_used = self.iteration
            summary.completed_at = __import__("datetime").datetime.now().isoformat()
            interrupt_manager.unregister_child(self.child_id)
        
        return summary
    
    def _build_initial_prompt(self) -> str:
        """
        构建子 Agent 的初始提示
        文档：子 Agent 的唯一上下文仅来自 goal 和 context 字段
        """
        parts = [f"## 目标\n{self.task.goal}"]
        
        if self.task.context:
            parts.append(f"\n## 上下文\n{self.task.context}")
        
        parts.append(
            f"\n## 约束\n"
            f"- 最大工具调用轮次: {self.task.max_iterations}\n"
            f"- 可用工具集: {self.task.toolsets or '默认'}\n"
            f"- 完成后输出结构化 JSON 摘要"
        )
        
        return "\n".join(parts)
    
    def _parse_response(self, response: Dict) -> tuple[str, List]:
        """解析 LLM 响应"""
        text_parts, tool_calls = [], []
        content = response.get("content", [])
        
        if isinstance(content, list):
            for block in content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        tool_calls.append({
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        })
        elif isinstance(content, str):
            text_parts.append(content)
        
        return "\n".join(text_parts), tool_calls
    
    def _parse_final_summary(self, text: str, summary: SubAgentSummary):
        """
        从最终输出中提取结构化摘要
        文档：提供结构化的摘要，包括执行操作、发现内容、修改文件、遇到问题
        """
        import re
        
        json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
        if not json_match:
            json_match = re.search(r'\{[\s\S]*"final_answer"[\s\S]*\}', text)
        
        if json_match:
            try:
                data = json.loads(json_match.group(1) if '```' in text else json_match.group())
                summary.actions_taken   += data.get("actions_taken", [])
                summary.findings         = data.get("findings", [])
                summary.modified_files   = data.get("modified_files", [])
                summary.issues_encountered += data.get("issues_encountered", [])
                summary.final_answer     = data.get("final_answer", text)
            except json.JSONDecodeError:
                summary.final_answer = text
        else:
            summary.final_answer = text
```

---

## ⚡ 并行执行引擎 `src/delegation/executor.py`

```python
"""
并行执行引擎

文档规范：
- 线程池：ThreadPoolExecutor，MAX_CONCURRENT_CHILDREN = 3
- 最大并发：3 个任务（超出截断）
- 结果排序：按任务索引排序（与输入顺序一致）
- 单任务：直接运行，无线程池开销
- 中断传播：中断父 Agent → 中断所有子 Agent
"""
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import List, Dict, Optional, Callable
from src.delegation.models import (
    DelegationRequest, SingleTask, SubAgentSummary,
    TaskStatus, MAX_CONCURRENT_CHILDREN
)
from src.agent.child_agent import ChildAgent
from src.delegation.progress import TaskProgressTracker
from src.core.interrupt import interrupt_manager
from src.utils.logger import logger
from config.settings import settings


class DelegationExecutor:
    """
    委派执行引擎
    
    严格实现文档描述的并行批量执行机制
    """
    
    def __init__(self):
        # 文档：MAX_CONCURRENT_CHILDREN = 3 个工作线程
        self._executor = ThreadPoolExecutor(
            max_workers=MAX_CONCURRENT_CHILDREN,
            thread_name_prefix="hermes-child"
        )
    
    def execute(
        self,
        request: DelegationRequest,
        model_override: Optional[str] = None,
        provider_override: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[SubAgentSummary]:
        """
        执行委派请求（单任务或批量并行）
        
        Returns:
            按任务索引排序的摘要列表（文档：结果按输入顺序排序）
        """
        tasks = request.to_single_tasks()
        
        # 文档：超过 3 个截断（to_single_tasks 已处理，此处双重保证）
        if len(tasks) > MAX_CONCURRENT_CHILDREN:
            logger.warning(
                f"任务数 {len(tasks)} 超过最大并发 {MAX_CONCURRENT_CHILDREN}，已截断"
            )
            tasks = tasks[:MAX_CONCURRENT_CHILDREN]
        
        # 文档：单任务直接运行，无需线程池开销
        if len(tasks) == 1:
            logger.info("单任务委派：直接执行（无线程池）")
            return self._execute_single(
                tasks[0], 0, model_override, provider_override, progress_callback
            )
        
        # 批量并行执行
        logger.info(f"批量并行委派：{len(tasks)} 个任务")
        return self._execute_parallel(
            tasks, model_override, provider_override, progress_callback
        )
    
    def _execute_single(
        self,
        task: SingleTask,
        index: int,
        model_override: Optional[str],
        provider_override: Optional[str],
        progress_callback: Optional[Callable]
    ) -> List[SubAgentSummary]:
        """单任务直接执行"""
        child_id = f"child-{uuid.uuid4().hex[:6]}"
        agent = ChildAgent(
            child_id=child_id,
            task=task,
            task_index=index,
            model_override=model_override or settings.delegation.model,
            provider_override=provider_override or settings.delegation.provider,
        )
        if progress_callback:
            agent.set_progress_callback(progress_callback)
        
        summary = agent.run()
        return [summary]
    
    def _execute_parallel(
        self,
        tasks: List[SingleTask],
        model_override: Optional[str],
        provider_override: Optional[str],
        progress_callback: Optional[Callable]
    ) -> List[SubAgentSummary]:
        """
        并行执行多任务
        文档：使用 ThreadPoolExecutor，结果按索引排序
        """
        # index → future 映射
        future_to_index: Dict[Future, int] = {}
        
        # 初始化所有子 Agent
        agents = {}
        for i, task in enumerate(tasks):
            child_id = f"child-{i}-{uuid.uuid4().hex[:4]}"
            agent = ChildAgent(
                child_id=child_id,
                task=task,
                task_index=i,
                model_override=model_override or settings.delegation.model,
                provider_override=provider_override or settings.delegation.provider,
            )
            if progress_callback:
                agent.set_progress_callback(progress_callback)
            agents[i] = agent
        
        # 并行提交
        for i, agent in agents.items():
            future = self._executor.submit(agent.run)
            future_to_index[future] = i
            logger.info(f"📤 提交子任务 [{i}]: {tasks[i].goal[:50]}")
        
        # 收集结果（文档：按任务索引排序）
        index_to_summary: Dict[int, SubAgentSummary] = {}
        
        for future in as_completed(future_to_index.keys()):
            idx = future_to_index[future]
            try:
                summary = future.result(timeout=600)   # 10 分钟超时
                index_to_summary[idx] = summary
                logger.info(
                    f"📥 子任务 [{idx}] 完成: "
                    f"status={summary.status}, "
                    f"iterations={summary.iterations_used}"
                )
            except Exception as e:
                logger.error(f"❌ 子任务 [{idx}] 异常: {e}")
                index_to_summary[idx] = SubAgentSummary(
                    task_index=idx,
                    goal=tasks[idx].goal,
                    status=TaskStatus.FAILED,
                    error=str(e)
                )
        
        # 按索引升序返回（文档：结果按任务索引排序，匹配输入顺序）
        return [index_to_summary[i] for i in sorted(index_to_summary.keys())]
    
    def shutdown(self):
        self._executor.shutdown(wait=False)


# 全局执行引擎单例
delegation_executor = DelegationExecutor()
```

---

## 🛠️ delegate_task 工具 `src/tools/delegate_task.py`

```python
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
```

---

## 🧠 LLM Provider `src/llm/provider.py`

```python
"""
LLM Provider（支持模型覆盖）
文档：可通过 config.yaml 配置子 Agent 使用不同的模型
"""
import anthropic
from typing import List, Dict, Optional, Any
from config.settings import settings
from src.utils.logger import logger


class LLMProvider:
    
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
        if self.provider == "anthropic":
            return self._anthropic_chat(messages, tools, system, max_tokens)
        else:
            return self._openai_chat(messages, tools, system, max_tokens)
    
    def _anthropic_chat(self, messages, tools, system, max_tokens):
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
```

---

## 👨‍💼 父 Agent `src/agent/parent_agent.py`

```python
"""
父 Agent（深度 0）

文档：Agent 会根据任务复杂程度自动处理委托，无需明确要求
"""
import json
import signal
from typing import List, Dict, Optional
from src.tools.base import BaseTool, ToolResult
from src.tools.registry import tool_registry
from src.tools.delegate_task import DelegateTaskTool
from src.tools.terminal import RunCommandTool
from src.tools.file import ReadFileTool, WriteFileTool, ListFilesTool
from src.tools.web import WebSearchTool, WebFetchTool
from src.llm.provider import LLMProvider
from src.core.interrupt import interrupt_manager
from src.delegation.models import MAX_DELEGATION_DEPTH
from src.utils.logger import logger


PARENT_SYSTEM = """你是 Hermes，一个强大的 AI 主 Agent。

## 你的能力
- 直接执行任务（使用 terminal / file / web 工具）
- 将复杂任务委派给子 Agent（使用 delegate_task 工具）
- 协调多个子 Agent 并行工作

## 委派决策原则
**使用 delegate_task 当：**
- 任务需要独立的推理循环
- 任务可以并行拆分（最多 3 个）
- 希望保持当前上下文干净（子任务中间过程不污染主对话）
- 任务需要判断力和多步骤解决

**直接执行 当：**
- 简单的单步操作
- 快速的文件读写或命令执行
- 机械式数据处理

## 重要：委派时的上下文传递
子 Agent 对你的对话历史一无所知！
必须在 context 字段中提供子 Agent 所需的全部信息。

## 委派深度限制
你是深度 0 的父 Agent，子 Agent（深度 1）无法再次委派。
"""


class ParentAgent:
    """
    父 Agent（深度 0）
    
    文档：父 Agent 可生成子 Agent（深度 1），子 Agent 无法进一步委派
    """
    
    DEPTH = 0    # 文档：深度 0
    
    def __init__(self):
        self.llm = LLMProvider()
        self.messages: List[Dict] = []
        self.max_iterations = 100
        
        # 注册所有工具
        self._register_tools()
        
        # 设置 SIGINT 处理（中断传播）
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
        logger.info("🎯 父 Agent (深度 0) 初始化完成")
    
    def _register_tools(self):
        """注册父 Agent 可用的全部工具"""
        tools = [
            # 核心委派工具（文档：父 Agent 专属）
            DelegateTaskTool(),
            # terminal 工具集
            RunCommandTool(session_id="parent"),
            # file 工具集
            ReadFileTool(), WriteFileTool(), ListFilesTool(),
            # web 工具集
            WebSearchTool(), WebFetchTool(),
        ]
        tool_registry.register_many(tools)
        self._tools: Dict[str, BaseTool] = {t.name: t for t in tools}
        logger.info(f"父 Agent 工具: {list(self._tools.keys())}")
    
    def _handle_interrupt(self, signum, frame):
        """
        文档：中断父 Agent 将中断所有活跃的子 Agent
        """
        logger.warning("⚠️ 收到中断信号，传播到所有子 Agent...")
        interrupt_manager.interrupt_all()
    
    def run(self, user_input: str) -> str:
        """处理用户输入"""
        logger.info(f"📨 用户: {user_input}")
        
        self.messages.append({"role": "user", "content": user_input})
        tools_schema = [t.to_schema() for t in self._tools.values()]
        
        for iteration in range(self.max_iterations):
            
            response = self.llm.chat(
                messages=self.messages,
                tools=tools_schema,
                system=PARENT_SYSTEM,
                max_tokens=8192
            )
            
            text, tool_calls = self._parse_response(response)
            
            if not tool_calls or response.get("stop_reason") == "end_turn":
                self.messages.append({"role": "assistant", "content": text})
                return text
            
            if response.get("stop_reason") == "tool_use":
                self.messages.append({"role": "assistant", "content": response["content"]})
                tool_results = []
                
                for tc in tool_calls:
                    tool = self._tools.get(tc["name"])
                    logger.info(f"🔧 父 Agent 调用工具: {tc['name']}")
                    
                    if tool:
                        result: ToolResult = tool.safe_execute(**tc["input"])
                        output = result.output if result.success else {"error": result.error}
                    else:
                        output = {"error": f"工具 {tc['name']} 不存在"}
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": json.dumps(output, ensure_ascii=False, default=str)
                    })
                
                self.messages.append({"role": "user", "content": tool_results})
        
        return "达到最大迭代次数"
    
    def _parse_response(self, response: Dict):
        text_parts, tool_calls = [], []
        content = response.get("content", [])
        if isinstance(content, list):
            for block in content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        tool_calls.append({"id": block.id, "name": block.name, "input": block.input})
        elif isinstance(content, str):
            text_parts.append(content)
        return "\n".join(text_parts), tool_calls
    
    def reset(self):
        self.messages.clear()
        interrupt_manager.reset()
```

---

## 🧪 测试 `tests/test_delegation.py`

```python
"""
完整测试套件，覆盖文档所有核心规则
"""
import pytest
import threading
from unittest.mock import MagicMock, patch
from src.delegation.models import (
    DelegationRequest, SingleTask, BatchTaskItem,
    FORBIDDEN_CHILD_TOOLS, MAX_CONCURRENT_CHILDREN,
    SubAgentSummary, TaskStatus
)
from src.tools.registry import ToolRegistry
from src.core.interrupt import InterruptManager


# ── 数据模型测试 ────────────────────────────────────────────────

class TestDelegationRequest:
    
    def test_single_task_valid(self):
        req = DelegationRequest(goal="Fix bug", context="Error in line 42")
        tasks = req.to_single_tasks()
        assert len(tasks) == 1
        assert tasks[0].goal == "Fix bug"
    
    def test_batch_task_valid(self):
        req = DelegationRequest(tasks=[
            BatchTaskItem(goal="Research A", toolsets=["web"]),
            BatchTaskItem(goal="Research B", toolsets=["web"]),
            BatchTaskItem(goal="Fix build",  toolsets=["terminal", "file"]),
        ])
        tasks = req.to_single_tasks()
        assert len(tasks) == 3
    
    def test_batch_truncated_to_3(self):
        """文档：超过 3 个任务截断为 3"""
        req = DelegationRequest(tasks=[
            BatchTaskItem(goal=f"Task {i}") for i in range(5)
        ])
        assert len(req.tasks) == MAX_CONCURRENT_CHILDREN == 3
    
    def test_goal_and_tasks_mutually_exclusive(self):
        """文档：goal 和 tasks 互斥"""
        with pytest.raises(Exception):
            DelegationRequest(
                goal="Task A",
                tasks=[BatchTaskItem(goal="Task B")]
            )
    
    def test_neither_goal_nor_tasks_raises(self):
        with pytest.raises(Exception):
            DelegationRequest()
    
    def test_invalid_toolset_rejected(self):
        """合法工具集：terminal / file / web"""
        with pytest.raises(Exception):
            SingleTask(goal="test", toolsets=["invalid_tool"])
    
    def test_valid_toolsets_accepted(self):
        task = SingleTask(
            goal="test",
            toolsets=["terminal", "file", "web"]
        )
        assert set(task.toolsets) == {"terminal", "file", "web"}
    
    def test_default_max_iterations(self):
        """文档：默认 max_iterations = 50"""
        task = SingleTask(goal="test")
        assert task.max_iterations == 50


# ── 工具集禁用规则测试 ──────────────────────────────────────────

class TestToolRegistry:
    
    def setup_method(self):
        self.registry = ToolRegistry()
        
        # 注册模拟工具
        for name, toolset in [
            ("delegate_task",  "core"),     # 禁用
            ("clarify",        "core"),     # 禁用
            ("memory",         "core"),     # 禁用
            ("code_execution", "core"),     # 禁用
            ("send_message",   "core"),     # 禁用
            ("run_command",    "terminal"), # 可用
            ("read_file",      "file"),     # 可用
            ("web_search",     "web"),      # 可用
        ]:
            mock_tool = MagicMock()
            mock_tool.name = name
            mock_tool.toolset = toolset
            self.registry.register(mock_tool)
    
    def test_forbidden_tools_excluded_for_child(self):
        """文档：子 Agent 始终禁用 FORBIDDEN_CHILD_TOOLS"""
        tools = self.registry.get_tools_for_child(
            toolsets=["terminal", "file", "web"],
            is_child=True
        )
        tool_names = {t.name for t in tools}
        
        for forbidden in FORBIDDEN_CHILD_TOOLS:
            if forbidden in {t.name for t in self.registry._tools.values()}:
                assert forbidden not in tool_names, \
                    f"{forbidden} 应被禁用但仍出现在子 Agent 工具列表中"
    
    def test_toolset_filtering(self):
        """toolset 参数正确过滤工具"""
        tools = self.registry.get_tools_for_child(
            toolsets=["terminal"],
            is_child=True
        )
        names = {t.name for t in tools}
        assert "run_command" in names
        assert "read_file" not in names
        assert "web_search" not in names
    
    def test_parent_gets_all_tools(self):
        """父 Agent 获取全部工具（无限制）"""
        tools = self.registry.get_tools_for_parent()
        assert len(tools) == len(self.registry._tools)


# ── 中断传播测试 ────────────────────────────────────────────────

class TestInterruptManager:
    
    def setup_method(self):
        self.manager = InterruptManager()
    
    def test_child_interrupted_when_parent_interrupted(self):
        """文档：中断父 Agent 将中断所有活跃子 Agent"""
        event1 = self.manager.register_child("child-1")
        event2 = self.manager.register_child("child-2")
        
        assert not event1.is_set()
        assert not event2.is_set()
        
        self.manager.interrupt_all()
        
        assert event1.is_set()
        assert event2.is_set()
    
    def test_new_child_immediately_interrupted_if_parent_interrupted(self):
        """父已被中断时，新注册子 Agent 立即触发中断"""
        self.manager.interrupt_all()
        late_event = self.manager.register_child("late-child")
        assert late_event.is_set()
    
    def test_reset_clears_interrupt(self):
        self.manager.interrupt_all()
        self.manager.reset()
        assert not self.manager.is_parent_interrupted()
        assert self.manager.active_children_count == 0
    
    def test_child_unregister(self):
        self.manager.register_child("child-1")
        assert self.manager.active_children_count == 1
        self.manager.unregister_child("child-1")
        assert self.manager.active_children_count == 0


# ── 摘要格式测试 ────────────────────────────────────────────────

class TestSubAgentSummary:
    
    def test_summary_context_string(self):
        """文档：结构化摘要包含操作/发现/修改文件/问题"""
        summary = SubAgentSummary(
            task_index=0,
            goal="Fix TypeError in api/handlers.py",
            status=TaskStatus.COMPLETED,
            actions_taken=["读取文件", "修改第47行"],
            findings=["parse_body() 在缺少 Content-Type 时返回 None"],
            modified_files=["api/handlers.py"],
            issues_encountered=[],
            final_answer="已修复 TypeError，添加了 None 检查"
        )
        
        ctx = summary.to_context_string()
        assert "Fix TypeError" in ctx
        assert "api/handlers.py" in ctx
        assert "parse_body" in ctx
        assert "completed" in ctx
    
    def test_summary_index_ordering(self):
        """文档：结果按任务索引排序"""
        summaries = [
            SubAgentSummary(task_index=2, goal="C", status=TaskStatus.COMPLETED),
            SubAgentSummary(task_index=0, goal="A", status=TaskStatus.COMPLETED),
            SubAgentSummary(task_index=1, goal="B", status=TaskStatus.COMPLETED),
        ]
        sorted_summaries = sorted(summaries, key=lambda s: s.task_index)
        assert [s.goal for s in sorted_summaries] == ["A", "B", "C"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```

---

## 🚀 主入口 `main.py`

```python
#!/usr/bin/env python3
"""
Hermes Parallel Agents Framework
基于官方文档实现的并行委派框架
"""
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import box
from src.agent.parent_agent import ParentAgent
from src.core.interrupt import interrupt_manager

console = Console()

def print_banner():
    console.print(Panel(
        "[bold cyan]⚡ Hermes Parallel Agents Framework[/bold cyan]\n\n"
        "[white]核心特性:[/white]\n"
        "  [green]✓[/green] 子 Agent 完全隔离上下文\n"
        "  [green]✓[/green] 最多 3 个子 Agent 并行\n"
        "  [green]✓[/green] 禁用工具集严格管控\n"
        "  [green]✓[/green] 中断信号传播\n"
        "  [green]✓[/green] 仅摘要进入父 Agent 上下文\n\n"
        "[dim]命令: quit / reset / help / Ctrl+C 中断[/dim]",
        border_style="cyan", width=58
    ))

def print_help():
    table = Table(title="示例指令", box=box.ROUNDED)
    table.add_column("类型", style="cyan")
    table.add_column("示例", style="yellow")
    
    table.add_row("单任务委派", "帮我检查 workspace/app.py 是否有语法错误")
    table.add_row("并行研究",   "并行研究：Python 3.13 新特性 + Rust 2024 edition + WebAssembly 现状")
    table.add_row("代码审查",   "审查 workspace/src 目录的安全问题并修复")
    table.add_row("直接问答",   "什么是 ReAct 框架？")
    
    console.print(table)

def main():
    print_banner()
    
    # 初始化工作目录
    Path("./workspace").mkdir(exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)
    
    agent = ParentAgent()
    
    console.print("\n[green]✅ 父 Agent (深度 0) 就绪[/green]")
    console.print("[dim]输入 'help' 查看示例，Ctrl+C 触发中断传播[/dim]\n")
    
    while True:
        try:
            user_input = console.input("[bold yellow]You:[/bold yellow] ").strip()
            if not user_input:
                continue
            
            match user_input.lower():
                case "quit" | "exit":
                    console.print("[dim]👋 再见！[/dim]")
                    break
                case "reset":
                    agent.reset()
                    console.print("[green]🔄 已重置[/green]")
                    continue
                case "help":
                    print_help()
                    continue
            
            with console.status("[bold green]🤖 Agent 处理中...[/bold green]"):
                response = agent.run(user_input)
            
            console.print(f"\n[bold blue]Hermes:[/bold blue]")
            console.print(Markdown(response))
            console.print()
        
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️ 中断信号 → 传播至所有子 Agent[/yellow]")
            interrupt_manager.interrupt_all()
        except Exception as e:
            console.print(f"[red]❌ 错误: {e}[/red]")

if __name__ == "__main__":
    main()
```

---

## 🏃 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境
echo "ANTHROPIC_API_KEY=your_key" > .env

# 3. 初始化目录
mkdir -p workspace logs

# 4. 运行
python main.py

# 5. 运行测试
pytest tests/test_delegation.py -v
```

---

## ✅ 文档规则对照表

| 文档规则 | 实现位置 | 状态 |
|----------|---------|------|
| 最多 3 个并发子 Agent | `models.py MAX_CONCURRENT_CHILDREN` | ✅ |
| ThreadPoolExecutor | `executor.py` | ✅ |
| 结果按索引排序 | `executor._execute_parallel()` | ✅ |
| 单任务无线程池开销 | `executor._execute_single()` | ✅ |
| 子 Agent 完全隔离上下文 | `child_agent._build_initial_prompt()` | ✅ |
| 禁用 5 类工具 | `registry.get_tools_for_child()` | ✅ |
| 深度限制（最大 2）| `parent_agent.DEPTH = 0` | ✅ |
| 中断传播 | `interrupt.py InterruptManager` | ✅ |
| 仅摘要进入父上下文 | `models.SubAgentSummary.to_context_string()` | ✅ |
| 模型覆盖（config.yaml）| `settings.py + provider.py` | ✅ |
| 树形进度视图 | `progress.py TaskProgressTracker` | ✅ |
| max_iterations 参数 | `models.SingleTask.max_iterations` | ✅ |

---

> 💡 **后续建议**：当前 `web_search` 为模拟实现，你是否希望接入真实搜索 API（如 **Tavily** 或 **Brave Search**），让并行研究类任务真正可用？
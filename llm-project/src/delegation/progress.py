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
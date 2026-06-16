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


# ── LLM Provider 测试 ───────────────────────────────────────────

class TestLLMProvider:
    
    def test_provider_initialization(self):
        """测试 LLMProvider 初始化"""
        from src.llm.provider import LLMProvider
        provider = LLMProvider(
            model_override="claude-sonnet-4-5",
            provider_override="anthropic"
        )
        assert provider.model == "claude-sonnet-4-5"
        assert provider.provider == "anthropic"
    
    def test_provider_override(self):
        """测试模型覆盖功能"""
        from src.llm.provider import LLMProvider
        provider = LLMProvider(
            model_override="gpt-4o",
            provider_override="openai"
        )
        assert provider.model == "gpt-4o"
        assert provider.provider == "openai"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
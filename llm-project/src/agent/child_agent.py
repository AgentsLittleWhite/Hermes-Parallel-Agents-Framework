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
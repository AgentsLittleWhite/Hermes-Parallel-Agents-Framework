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
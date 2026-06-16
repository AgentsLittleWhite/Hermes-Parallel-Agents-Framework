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
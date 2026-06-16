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
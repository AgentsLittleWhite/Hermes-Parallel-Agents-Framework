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
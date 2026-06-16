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
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
                raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
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
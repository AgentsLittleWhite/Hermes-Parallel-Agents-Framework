"""
delegation 配置模块
"""
from typing import Optional, List
from config.settings import settings


class DelegationConfig:
    """delegation 运行时配置"""
    
    @staticmethod
    def get_max_iterations() -> int:
        return settings.delegation.max_iterations
    
    @staticmethod
    def get_default_toolsets() -> List[str]:
        return settings.delegation.default_toolsets
    
    @staticmethod
    def get_model_override() -> Optional[str]:
        return settings.delegation.model
    
    @staticmethod
    def get_provider_override() -> Optional[str]:
        return settings.delegation.provider
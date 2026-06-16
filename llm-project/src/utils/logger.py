from loguru import logger
import sys
from pathlib import Path

# 配置日志输出
Path("./logs").mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")
logger.add("./logs/hermes_{time}.log", rotation="10 MB", level="DEBUG")

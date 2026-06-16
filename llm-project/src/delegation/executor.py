"""
并行执行引擎

文档规范：
- 线程池：ThreadPoolExecutor，MAX_CONCURRENT_CHILDREN = 3
- 最大并发：3 个任务（超出截断）
- 结果排序：按任务索引排序（与输入顺序一致）
- 单任务：直接运行，无线程池开销
- 中断传播：中断父 Agent → 中断所有子 Agent
"""
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import List, Dict, Optional, Callable
from src.delegation.models import (
    DelegationRequest, SingleTask, SubAgentSummary,
    TaskStatus, MAX_CONCURRENT_CHILDREN
)
from src.agent.child_agent import ChildAgent
from src.delegation.progress import TaskProgressTracker
from src.core.interrupt import interrupt_manager
from src.utils.logger import logger
from config.settings import settings


class DelegationExecutor:
    """
    委派执行引擎

    严格实现文档描述的并行批量执行机制
    """
    
    def __init__(self):
        # 文档：MAX_CONCURRENT_CHILDREN = 3 个工作线程
        self._executor = ThreadPoolExecutor(
            max_workers=MAX_CONCURRENT_CHILDREN,
            thread_name_prefix="hermes-child"
        )
    
    def execute(
        self,
        request: DelegationRequest,
        model_override: Optional[str] = None,
        provider_override: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[SubAgentSummary]:
        """
        执行委派请求（单任务或批量并行）

        Returns:
            按任务索引排序的摘要列表（文档：结果按输入顺序排序）
        """
        tasks = request.to_single_tasks()
        
        # 文档：超过 3 个截断（to_single_tasks 已处理，此处双重保证）
        if len(tasks) > MAX_CONCURRENT_CHILDREN:
            logger.warning(
                f"任务数 {len(tasks)} 超过最大并发 {MAX_CONCURRENT_CHILDREN}，已截断"
            )
            tasks = tasks[:MAX_CONCURRENT_CHILDREN]
        
        # 文档：单任务直接运行，无需线程池开销
        if len(tasks) == 1:
            logger.info("单任务委派：直接执行（无线程池）")
            return self._execute_single(
                tasks[0], 0, model_override, provider_override, progress_callback
            )
        
        # 批量并行执行
        logger.info(f"批量并行委派：{len(tasks)} 个任务")
        return self._execute_parallel(
            tasks, model_override, provider_override, progress_callback
        )
    
    def _execute_single(
        self,
        task: SingleTask,
        index: int,
        model_override: Optional[str],
        provider_override: Optional[str],
        progress_callback: Optional[Callable]
    ) -> List[SubAgentSummary]:
        """单任务直接执行"""
        child_id = f"child-{uuid.uuid4().hex[:6]}"
        agent = ChildAgent(
            child_id=child_id,
            task=task,
            task_index=index,
            model_override=model_override or settings.delegation.model,
            provider_override=provider_override or settings.delegation.provider,
        )
        if progress_callback:
            agent.set_progress_callback(progress_callback)
        
        summary = agent.run()
        return [summary]
    
    def _execute_parallel(
        self,
        tasks: List[SingleTask],
        model_override: Optional[str],
        provider_override: Optional[str],
        progress_callback: Optional[Callable]
    ) -> List[SubAgentSummary]:
        """
        并行执行多任务
        文档：使用 ThreadPoolExecutor，结果按索引排序
        """
        # index → future 映射
        future_to_index: Dict[Future, int] = {}
        
        # 初始化所有子 Agent
        agents = {}
        for i, task in enumerate(tasks):
            child_id = f"child-{i}-{uuid.uuid4().hex[:4]}"
            agent = ChildAgent(
                child_id=child_id,
                task=task,
                task_index=i,
                model_override=model_override or settings.delegation.model,
                provider_override=provider_override or settings.delegation.provider,
            )
            if progress_callback:
                agent.set_progress_callback(progress_callback)
            agents[i] = agent
        
        # 并行提交
        for i, agent in agents.items():
            future = self._executor.submit(agent.run)
            future_to_index[future] = i
            logger.info(f"📤 提交子任务 [{i}]: {tasks[i].goal[:50]}")
        
        # 收集结果（文档：按任务索引排序）
        index_to_summary: Dict[int, SubAgentSummary] = {}
        
        for future in as_completed(future_to_index.keys()):
            idx = future_to_index[future]
            try:
                summary = future.result(timeout=600)   # 10 分钟超时
                index_to_summary[idx] = summary
                logger.info(
                    f"📥 子任务 [{idx}] 完成: "
                    f"status={summary.status}, "
                    f"iterations={summary.iterations_used}"
                )
            except Exception as e:
                logger.error(f"❌ 子任务 [{idx}] 异常: {e}")
                index_to_summary[idx] = SubAgentSummary(
                    task_index=idx,
                    goal=tasks[idx].goal,
                    status=TaskStatus.FAILED,
                    error=str(e)
                )
        
        # 按索引升序返回（文档：结果按任务索引排序，匹配输入顺序）
        return [index_to_summary[i] for i in sorted(index_to_summary.keys())]
    
    def shutdown(self):
        self._executor.shutdown(wait=False)


# 全局执行引擎单例
delegation_executor = DelegationExecutor()
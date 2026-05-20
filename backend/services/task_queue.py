from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor


class TaskQueue:
    def __init__(self, max_workers: int = 1):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="analysis-task")

    def submit(self, *args, **kwargs):
        from backend.services.analysis_service import run_single_task

        return self.executor.submit(run_single_task, *args, **kwargs)

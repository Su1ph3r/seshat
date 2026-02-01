"""
Task scheduler for Seshat monitoring.

Schedules and runs periodic checks.
"""

from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import time


@dataclass
class ScheduledTask:
    """Represents a scheduled task."""
    name: str
    callback: Callable
    interval_seconds: float
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    is_active: bool = True
    run_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None

    def should_run(self) -> bool:
        """Check if task should run now."""
        if not self.is_active:
            return False
        if self.next_run is None:
            return True
        return datetime.now() >= self.next_run

    def update_schedule(self):
        """Update schedule after running."""
        self.last_run = datetime.now()
        self.next_run = self.last_run + timedelta(seconds=self.interval_seconds)
        self.run_count += 1


class Scheduler:
    """
    Simple task scheduler for monitoring jobs.
    """

    def __init__(self):
        """Initialize scheduler."""
        self._tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def add_task(
        self,
        name: str,
        callback: Callable,
        interval_seconds: float,
        run_immediately: bool = False,
    ) -> ScheduledTask:
        """
        Add a scheduled task.

        Args:
            name: Task name
            callback: Function to call
            interval_seconds: Interval between runs
            run_immediately: Run immediately on add

        Returns:
            Created ScheduledTask
        """
        task = ScheduledTask(
            name=name,
            callback=callback,
            interval_seconds=interval_seconds,
        )

        if run_immediately:
            task.next_run = datetime.now()
        else:
            task.next_run = datetime.now() + timedelta(seconds=interval_seconds)

        with self._lock:
            self._tasks[name] = task

        return task

    def remove_task(self, name: str) -> bool:
        """
        Remove a scheduled task.

        Args:
            name: Task name

        Returns:
            True if removed
        """
        with self._lock:
            if name in self._tasks:
                del self._tasks[name]
                return True
        return False

    def pause_task(self, name: str):
        """Pause a task."""
        with self._lock:
            if name in self._tasks:
                self._tasks[name].is_active = False

    def resume_task(self, name: str):
        """Resume a paused task."""
        with self._lock:
            if name in self._tasks:
                self._tasks[name].is_active = True

    def list_tasks(self) -> List[Dict[str, Any]]:
        """
        List all scheduled tasks.

        Returns:
            List of task info dictionaries
        """
        with self._lock:
            return [
                {
                    "name": task.name,
                    "interval_seconds": task.interval_seconds,
                    "is_active": task.is_active,
                    "last_run": task.last_run.isoformat() if task.last_run else None,
                    "next_run": task.next_run.isoformat() if task.next_run else None,
                    "run_count": task.run_count,
                    "error_count": task.error_count,
                    "last_error": task.last_error,
                }
                for task in self._tasks.values()
            ]

    def run_once(self) -> List[Dict[str, Any]]:
        """
        Run all due tasks once.

        Returns:
            List of run results
        """
        results = []

        with self._lock:
            due_tasks = [t for t in self._tasks.values() if t.should_run()]

        for task in due_tasks:
            result = self._run_task(task)
            results.append(result)

        return results

    def _run_task(self, task: ScheduledTask) -> Dict[str, Any]:
        """Run a single task."""
        result = {
            "name": task.name,
            "started_at": datetime.now().isoformat(),
            "success": False,
            "error": None,
        }

        try:
            task.callback()
            result["success"] = True

        except Exception as e:
            result["error"] = str(e)
            task.error_count += 1
            task.last_error = str(e)

        finally:
            task.update_schedule()
            result["finished_at"] = datetime.now().isoformat()

        return result

    def start(self, poll_interval: float = 60.0):
        """
        Start the scheduler in background.

        Args:
            poll_interval: Seconds between checking for due tasks
        """
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(poll_interval,),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run_loop(self, poll_interval: float):
        """Main scheduler loop."""
        while self._running:
            self.run_once()
            time.sleep(poll_interval)

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

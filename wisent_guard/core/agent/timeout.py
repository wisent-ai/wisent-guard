"""
Timeout management for wisent-guard agent operations.

This module provides hard timeout enforcement to ensure operations 
don't exceed their allocated time budgets.
"""

import asyncio
import time
from typing import Optional, Any
from contextlib import asynccontextmanager


class TimeoutError(Exception):
    """Raised when an operation exceeds its time budget."""
    
    def __init__(self, message: str, elapsed_time: float, budget_time: float):
        super().__init__(message)
        self.elapsed_time = elapsed_time
        self.budget_time = budget_time


class TimeoutManager:
    """Manages hard timeouts for agent operations."""
    
    def __init__(self, budget_minutes: float):
        self.budget_seconds = budget_minutes * 60.0
        self.start_time = None
        self.deadline = None
        
    def start(self):
        """Start the timeout timer."""
        self.start_time = time.time()
        self.deadline = self.start_time + self.budget_seconds
        
    def check_timeout(self):
        """Check if we've exceeded the timeout. Raises TimeoutError if so."""
        if self.start_time is None:
            return  # Not started yet
            
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if current_time > self.deadline:
            raise TimeoutError(
                f"Operation exceeded time budget of {self.budget_seconds:.1f}s (elapsed: {elapsed:.1f}s)",
                elapsed_time=elapsed,
                budget_time=self.budget_seconds
            )
    
    def get_remaining_time(self) -> float:
        """Get remaining time in seconds. Returns 0 if expired."""
        if self.start_time is None:
            return self.budget_seconds
            
        current_time = time.time()
        remaining = self.deadline - current_time
        return max(0.0, remaining)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def is_expired(self) -> bool:
        """Check if the timeout has expired."""
        return self.get_remaining_time() <= 0


@asynccontextmanager
async def timeout_context(budget_minutes: float):
    """
    Context manager that enforces a hard timeout for async operations.
    
    Usage:
        async with timeout_context(5.0) as timeout_mgr:
            # Your operation here
            timeout_mgr.check_timeout()  # Call periodically
    """
    timeout_mgr = TimeoutManager(budget_minutes)
    timeout_mgr.start()
    
    try:
        yield timeout_mgr
    except TimeoutError:
        print(f"⏰ Operation timed out after {timeout_mgr.get_elapsed_time():.1f}s (budget: {budget_minutes:.1f}min)")
        raise


def with_timeout(budget_minutes: float):
    """
    Decorator that adds timeout enforcement to async functions.
    
    Usage:
        @with_timeout(5.0)
        async def my_operation():
            # Your code here
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with timeout_context(budget_minutes) as timeout_mgr:
                # Inject timeout manager into function if it accepts it
                import inspect
                sig = inspect.signature(func)
                if 'timeout_mgr' in sig.parameters:
                    kwargs['timeout_mgr'] = timeout_mgr
                
                return await func(*args, **kwargs)
        return wrapper
    return decorator


class AsyncTimeoutChecker:
    """
    Helper class for checking timeouts in long-running async operations.
    Automatically checks timeout every few operations.
    """
    
    def __init__(self, timeout_mgr: TimeoutManager, check_interval: int = 10):
        self.timeout_mgr = timeout_mgr
        self.check_interval = check_interval
        self.operation_count = 0
    
    def tick(self):
        """Call this on each iteration/operation. Checks timeout periodically."""
        self.operation_count += 1
        if self.operation_count % self.check_interval == 0:
            self.timeout_mgr.check_timeout()
    
    async def async_tick(self):
        """Async version that yields control and checks timeout."""
        self.tick()
        await asyncio.sleep(0)  # Yield control 
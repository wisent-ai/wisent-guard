"""
Task management system for wisent-guard.

This module provides comprehensive task discovery, validation, and loading
functionality for lm-evaluation-harness tasks.
"""

from .task_manager import (
    TaskManager,
    load_available_tasks,
    load_docs,
    get_available_tasks,
    is_valid_task,
    resolve_task_name
)

from .task_selector import (
    TaskSelector,
    find_relevant_tasks_for_issue_type,
    select_best_tasks_for_training
)

__all__ = [
    'TaskManager',
    'TaskSelector', 
    'load_available_tasks',
    'load_docs',
    'get_available_tasks',
    'is_valid_task',
    'resolve_task_name',
    'find_relevant_tasks_for_issue_type',
    'select_best_tasks_for_training'
] 
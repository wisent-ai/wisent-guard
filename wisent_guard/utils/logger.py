"""
Logging utilities for Wisent Guard
"""

import logging
import os
import sys
from typing import Optional

# Define log levels
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

# ANSI color codes for terminal output
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}

# Global logger instance
_logger = None

def get_logger(name: str = "wisent_guard", level: str = "info", 
               log_file: Optional[str] = None, use_colors: bool = True) -> logging.Logger:
    """
    Get or create a logger with the specified configuration.
    
    Args:
        name: Logger name
        level: Log level ('debug', 'info', 'warning', 'error', 'critical')
        log_file: Optional file path to write logs to
        use_colors: Whether to use colored terminal output
        
    Returns:
        Configured logger instance
    """
    global _logger
    
    if _logger is not None:
        return _logger
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level from string
    log_level = LOG_LEVELS.get(level.lower(), logging.INFO)
    logger.setLevel(log_level)
    
    # Create handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Format with colors if enabled
    if use_colors:
        console_formatter = logging.Formatter(
            f"{COLORS['bold']}{COLORS['blue']}[%(asctime)s]{COLORS['reset']} "
            f"{COLORS['bold']}%(levelname)s:{COLORS['reset']} "
            f"{COLORS['cyan']}[%(name)s]{COLORS['reset']} %(message)s",
            datefmt="%H:%M:%S"
        )
    else:
        console_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: [%(name)s] %(message)s",
            datefmt="%H:%M:%S"
        )
    
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Add handlers to logger
    for handler in handlers:
        logger.addHandler(handler)
    
    # Store logger globally
    _logger = logger
    return logger

def set_log_level(level: str) -> None:
    """
    Set the log level for the existing logger.
    
    Args:
        level: Log level ('debug', 'info', 'warning', 'error', 'critical')
    """
    global _logger
    if _logger is not None:
        log_level = LOG_LEVELS.get(level.lower(), logging.INFO)
        _logger.setLevel(log_level)
        for handler in _logger.handlers:
            handler.setLevel(log_level) 
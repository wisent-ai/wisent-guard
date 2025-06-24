"""
Wisent-Guard MCP (Model Control Protocol) Server

This module provides MCP tools for models to perform self-reflection and behavior editing
using wisent-guard capabilities.
"""

from .server import WisentGuardMCPServer
from .tools import (
    SelfReflectionTool,
    HallucinationDetectionTool,
    BehaviorEditingTool,
    ResponseAnalysisTool
)

__all__ = [
    'WisentGuardMCPServer',
    'SelfReflectionTool',
    'HallucinationDetectionTool', 
    'BehaviorEditingTool',
    'ResponseAnalysisTool'
] 
"""
Tools package.

Public API — import these names; never import from sub-modules directly.
"""

from .base import BaseTool, ToolError, ToolResult
from .calculator import CalculatorInput, CalculatorTool
from .doc_search import SUPPORTED_SOURCES, DocSearchInput, DocSearchTool
from .registry import ToolRegistry
from .runner import arun_with_tools, run_with_tools

__all__ = [
    "SUPPORTED_SOURCES",
    "BaseTool",
    "CalculatorInput",
    "CalculatorTool",
    "DocSearchInput",
    "DocSearchTool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "arun_with_tools",
    "run_with_tools",
]

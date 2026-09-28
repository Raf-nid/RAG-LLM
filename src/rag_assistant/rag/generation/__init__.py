"""
Generation module for grounded RAG with citations.

This module provides the generation layer that takes retrieved chunks
and produces grounded answers with explicit source citations.
"""

from .generator import (
    GroundedGenerator,
    create_generator,
)
from .prompts import (
    RAGPromptBuilder,
)

__all__ = [
    "GroundedGenerator",
    "RAGPromptBuilder",
    "create_generator",
]

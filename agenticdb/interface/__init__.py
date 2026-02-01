"""
Interface layer for AgenticDB.

Provides the main client SDK for interacting with AgenticDB.
"""

from agenticdb.interface.client import AgenticDB, BranchHandle

__all__ = [
    "AgenticDB",
    "BranchHandle",
]

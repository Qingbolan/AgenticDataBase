# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Interface layer for AgenticDB.

Provides the main client SDK for interacting with AgenticDB,
including the Intent-Aware Transaction API.
"""

from agenticdb.interface.client import AgenticDB, BranchHandle
from agenticdb.interface.query_result import QueryResult
from agenticdb.interface.builders import TransactionBuilder, StoreBuilder, FluentQueryBuilder
from agenticdb.interface.session import Session
from agenticdb.interface.async_client import AsyncAgenticDB, AsyncSession

__all__ = [
    # Core client
    "AgenticDB",
    "BranchHandle",
    # Query results
    "QueryResult",
    # Builders
    "TransactionBuilder",
    "StoreBuilder",
    "FluentQueryBuilder",
    # Session
    "Session",
    # Async
    "AsyncAgenticDB",
    "AsyncSession",
]

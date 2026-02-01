"""
Query layer for AgenticDB.

Provides the query engine and operators that make AgenticDB different
from traditional databases:

- why(x): Trace the causal chain that led to X
- impact(x): Find everything that depends on X
"""

from agenticdb.query.engine import QueryEngine
from agenticdb.query.operators import WhyQuery, ImpactQuery, TraceQuery

__all__ = [
    "QueryEngine",
    "WhyQuery",
    "ImpactQuery",
    "TraceQuery",
]

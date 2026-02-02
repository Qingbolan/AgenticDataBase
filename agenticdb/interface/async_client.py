# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Async client for AgenticDB.

Provides async/await support for the Intent-Aware Transaction API.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import AgenticDB
    from .query_result import QueryResult


class AsyncAgenticDB:
    """
    Async wrapper for AgenticDB.

    Provides async/await versions of the main API methods by
    running them in a thread pool executor.

    Example:
        async_db = AsyncAgenticDB(db)
        result = await async_db.query("show orders from last week")
    """

    def __init__(
        self,
        db: "AgenticDB",
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        """
        Initialize the async wrapper.

        Args:
            db: AgenticDB instance
            executor: Optional ThreadPoolExecutor
        """
        self._db = db
        self._executor = executor or ThreadPoolExecutor(max_workers=4)

    async def query(
        self,
        query_text: str,
        bindings: Optional[Dict[str, Any]] = None,
        branch_id: Optional[str] = None,
    ) -> "QueryResult":
        """
        Execute a query asynchronously.

        Args:
            query_text: Natural language query
            bindings: Optional explicit bindings
            branch_id: Branch to query

        Returns:
            QueryResult with operation outcome
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._db.query(query_text, bindings, branch_id),
        )

    async def bind(
        self,
        transaction_id: str,
        **bindings: Any,
    ) -> "QueryResult":
        """
        Provide bindings asynchronously.

        Args:
            transaction_id: Transaction ID
            **bindings: Named bindings

        Returns:
            QueryResult with updated status
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._db.bind(transaction_id, **bindings),
        )

    async def confirm(
        self,
        transaction_id: str,
        yes: bool = True,
    ) -> "QueryResult":
        """
        Confirm asynchronously.

        Args:
            transaction_id: Transaction ID
            yes: Confirmation decision

        Returns:
            QueryResult with final status
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._db.confirm(transaction_id, yes),
        )

    async def store(
        self,
        event_type: str,
        data: Dict[str, Any],
        branch_id: Optional[str] = None,
    ) -> "QueryResult":
        """
        Store an event asynchronously.

        Args:
            event_type: Event type
            data: Event data
            branch_id: Branch to store in

        Returns:
            QueryResult with store outcome
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._db.store(event_type, data, branch_id),
        )

    def close(self) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)

    async def __aenter__(self) -> "AsyncAgenticDB":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        self.close()


class AsyncSession:
    """
    Async session for multi-turn interactions.

    Example:
        async with async_db.session() as session:
            r1 = await session.query("show me users")
            r2 = await session.bind(target="customers")
    """

    def __init__(
        self,
        async_db: AsyncAgenticDB,
    ):
        """
        Initialize the async session.

        Args:
            async_db: AsyncAgenticDB instance
        """
        self._async_db = async_db
        self._bindings: Dict[str, Any] = {}
        self._pending_transaction_id: Optional[str] = None

    async def query(
        self,
        query_text: str,
        bindings: Optional[Dict[str, Any]] = None,
    ) -> "QueryResult":
        """
        Execute a query asynchronously.

        Args:
            query_text: Natural language query
            bindings: Optional bindings

        Returns:
            QueryResult
        """
        merged = {**self._bindings}
        if bindings:
            merged.update(bindings)

        result = await self._async_db.query(query_text, merged)

        if result.status == "pending_binding":
            self._pending_transaction_id = result.transaction_id

        return result

    async def bind(self, **bindings: Any) -> "QueryResult":
        """
        Provide bindings asynchronously.

        Args:
            **bindings: Named bindings

        Returns:
            QueryResult
        """
        self._bindings.update(bindings)

        if self._pending_transaction_id:
            result = await self._async_db.bind(
                self._pending_transaction_id,
                **bindings,
            )
            if result.status != "pending_binding":
                self._pending_transaction_id = None
            return result

        from .query_result import QueryResult
        return QueryResult.success(data={"bindings": bindings})

    async def confirm(self, yes: bool = True) -> "QueryResult":
        """
        Confirm asynchronously.

        Args:
            yes: Confirmation decision

        Returns:
            QueryResult
        """
        if not self._pending_transaction_id:
            from .query_result import QueryResult
            return QueryResult.error("No pending transaction")

        result = await self._async_db.confirm(self._pending_transaction_id, yes)
        self._pending_transaction_id = None
        return result

    async def __aenter__(self) -> "AsyncSession":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        pass

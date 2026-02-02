# coding: utf-8
"""
Author: Silan Hu(silan.hu@u.nus.edu)
Main client interface for AgenticDB.

This module provides the primary SDK for interacting with AgenticDB,
offering a clean, intuitive API for agent developers.

Usage:
    ```python
    from agenticdb import AgenticDB, Event, Claim, Action

    # Initialize
    db = AgenticDB()

    # Create a branch for your workflow
    branch = db.create_branch("my-workflow")

    # Record events, claims, and actions
    event = branch.record(Event(type="UserSignup", data={...}))
    claim = branch.record(Claim(subject="user.risk", value=0.3, source="model"))
    action = branch.execute(Action(type="Approve", agent_id="agent-1", depends_on=[event.id]))

    # Query: Why did this happen?
    chain = db.why(action.id)

    # Query: What's affected if this changes?
    impact = db.impact(claim.id)
    ```
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Optional, overload

from agenticdb.core.models import Entity, EntityType, Event, Claim, Action
from agenticdb.core.version import Branch, Version, Snapshot
from agenticdb.core.dependency import DependencyGraph, DependencyEdge, EdgeType
from agenticdb.storage.engine import StorageEngine, InMemoryStorage
from agenticdb.query.engine import QueryEngine
from agenticdb.query.operators import CausalChain, ImpactResult
from agenticdb.runtime.cache import DependencyAwareCache
from agenticdb.runtime.subscription import (
    SubscriptionManager,
    Subscription,
    SubscriptionCallback,
)

# Type imports for annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agenticdb.ingestion.compiler import IngestResult


class BranchHandle:
    """
    A handle to a specific branch for recording and querying.

    BranchHandle provides the main API for working with agent state:
    - Recording events, claims, and actions
    - Executing actions with dependency tracking
    - Querying state and causal chains

    Thread Safety:
        BranchHandle operations are thread-safe when using the
        default InMemoryStorage backend.
    """

    def __init__(
        self,
        branch: Branch,
        storage: StorageEngine,
        graph: DependencyGraph,
        cache: DependencyAwareCache,
        subscriptions: SubscriptionManager,
    ):
        """
        Initialize a branch handle.

        Args:
            branch: The branch model
            storage: Storage backend
            graph: Dependency graph
            cache: Cache instance
            subscriptions: Subscription manager
        """
        self._branch = branch
        self._storage = storage
        self._graph = graph
        self._cache = cache
        self._subscriptions = subscriptions
        self._query_engine = QueryEngine(storage, graph, branch.id)

    @property
    def id(self) -> str:
        """Get the branch ID."""
        return self._branch.id

    @property
    def name(self) -> str:
        """Get the branch name."""
        return self._branch.name

    @property
    def version(self) -> int:
        """Get the current version number."""
        return self._branch.head_version

    # =========================================================================
    # Recording Entities
    # =========================================================================

    def record(self, entity: Entity) -> Entity:
        """
        Record an entity (Event or Claim) to this branch.

        Args:
            entity: Entity to record

        Returns:
            The recorded entity with ID and version set

        Example:
            ```python
            event = branch.record(Event(
                event_type="UserRegistered",
                data={"user_id": "u123"}
            ))
            ```
        """
        # Store the entity
        version = self._storage.store(entity, self._branch.id)

        # Add to dependency graph
        self._graph.add_entity(entity.id)

        # Handle derived_from for Claims
        if isinstance(entity, Claim) and entity.derived_from:
            for dep_id in entity.derived_from:
                self._graph.add_edge(
                    entity.id,
                    dep_id,
                    EdgeType.DERIVED_FROM,
                    version=version.version_number,
                )

        # Notify subscribers
        self._subscriptions.notify(entity, "create")

        return entity

    def execute(self, action: Action) -> Action:
        """
        Execute an action with dependency tracking.

        This is the primary way to record agent behaviors. The action's
        depends_on list is used to build the dependency graph.

        Args:
            action: Action to execute

        Returns:
            The executed action with ID and version set

        Example:
            ```python
            action = branch.execute(Action(
                action_type="ApproveOrder",
                agent_id="approval-agent",
                inputs={"order_id": "o123"},
                depends_on=[event.id, claim.id]
            ))
            ```
        """
        # Mark as running
        action.start()

        # Store the action
        version = self._storage.store(action, self._branch.id)

        # Add to dependency graph with explicit dependencies
        self._graph.add_entity(action.id)

        for dep_id in action.depends_on:
            self._graph.add_edge(
                action.id,
                dep_id,
                EdgeType.DEPENDS_ON,
                version=version.version_number,
            )

        # Mark as completed
        action.complete()

        # Invalidate cache entries that depend on action's dependencies
        for dep_id in action.depends_on:
            self._cache.invalidate_dependents(dep_id)

        # Notify subscribers
        self._subscriptions.notify(action, "create")

        return action

    # =========================================================================
    # Semantic Ingestion (Text → State)
    # =========================================================================

    def ingest(
        self,
        text: str,
        mode: str = "description",
        extractor: Optional[Any] = None,
    ) -> "IngestResult":
        """
        Ingest text and compile it into semantic objects.

        This is the high-level "product interface" — give it text,
        get structured state with automatic dependency inference.

        Args:
            text: Input text (agent trace, log, description)
            mode: Ingestion mode - "agent_trace", "log", "description"
            extractor: Optional custom extractor

        Returns:
            IngestResult with extracted entities and schema proposals

        Example:
            ```python
            result = branch.ingest('''
                User u123 registered with alice@example.com.
                Risk model v2 computed score = 0.15.
                User approved because score < threshold.
            ''', mode="agent_trace")

            print(result.events)   # [Event(UserRegistered)]
            print(result.claims)   # [Claim(risk_score=0.15)]
            print(result.actions)  # [Action(ApproveUser)]
            ```
        """
        from agenticdb.ingestion.compiler import (
            TraceCompiler,
            IngestionMode,
            IngestResult,
        )

        # Create compiler
        compiler = TraceCompiler(extractor=extractor)

        # Parse mode
        try:
            ingestion_mode = IngestionMode(mode)
        except ValueError:
            ingestion_mode = IngestionMode.DESCRIPTION

        # Compile text to semantic objects
        start_version = self.version
        compilation = compiler.compile(text, mode=ingestion_mode)

        # Record all extracted entities
        event_ids = []
        for event in compilation.events:
            self.record(event)
            event_ids.append(event.id)

        claim_ids = []
        for claim in compilation.claims:
            self.record(claim)
            claim_ids.append(claim.id)

        action_ids = []
        for action in compilation.actions:
            self.execute(action)
            action_ids.append(action.id)

        return IngestResult(
            compilation=compilation,
            event_ids=event_ids,
            claim_ids=claim_ids,
            action_ids=action_ids,
            start_version=start_version,
            end_version=self.version,
        )

    # =========================================================================
    # Querying
    # =========================================================================

    def get(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self._query_engine.get(entity_id)

    def events(
        self,
        event_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Event]:
        """Query events."""
        return self._query_engine.events(event_type, limit)

    def claims(
        self,
        subject: Optional[str] = None,
        source: Optional[str] = None,
        active_only: bool = True,
        limit: Optional[int] = None,
    ) -> Iterator[Claim]:
        """Query claims."""
        return self._query_engine.claims(subject, source, active_only, limit)

    def actions(
        self,
        action_type: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Action]:
        """Query actions."""
        return self._query_engine.actions(action_type, agent_id, limit)

    # =========================================================================
    # Causal Queries
    # =========================================================================

    def why(
        self,
        entity_id: str,
        max_depth: Optional[int] = None,
    ) -> CausalChain:
        """
        Trace the causal chain that led to an entity.

        Args:
            entity_id: Entity to trace
            max_depth: Maximum traversal depth

        Returns:
            CausalChain showing the causal path
        """
        return self._query_engine.why(entity_id, max_depth)

    def impact(
        self,
        entity_id: str,
        max_depth: Optional[int] = None,
        auto_invalidate: bool = False,
    ) -> ImpactResult:
        """
        Find all entities affected by a change to this entity.

        Args:
            entity_id: Entity to analyze
            max_depth: Maximum traversal depth
            auto_invalidate: Mark affected claims as invalidated

        Returns:
            ImpactResult showing downstream dependencies
        """
        return self._query_engine.impact(entity_id, max_depth, auto_invalidate)

    # =========================================================================
    # Time Travel
    # =========================================================================

    def at(self, version: int) -> Snapshot:
        """
        Get a snapshot of state at a specific version.

        Args:
            version: Version number

        Returns:
            Snapshot of state at that version
        """
        return self._query_engine.at_version(version)

    def history(self, entity_id: str) -> list[Entity]:
        """
        Get the history of an entity across versions.

        Args:
            entity_id: Entity to trace

        Returns:
            List of entity states across versions
        """
        return self._query_engine.history(entity_id)

    # =========================================================================
    # Branching
    # =========================================================================

    def fork(self, name: str, description: Optional[str] = None) -> Branch:
        """
        Create a new branch forked from current state.

        Args:
            name: Name for the new branch
            description: Optional description

        Returns:
            The new branch (use db.branch() to get a handle)
        """
        return self._branch.fork(name, description)


class AgenticDB:
    """
    Main entry point for AgenticDB.

    AgenticDB is an agent-native data runtime where actions, reasoning,
    and state evolution are first-class queryable semantics.

    Usage:
        ```python
        db = AgenticDB()

        # Create a branch
        branch = db.create_branch("my-workflow")

        # Record state
        event = branch.record(Event(...))
        claim = branch.record(Claim(...))
        action = branch.execute(Action(...))

        # Query
        chain = db.why(action.id)
        impact = db.impact(claim.id)
        ```
    """

    def __init__(
        self,
        storage: Optional[StorageEngine] = None,
    ):
        """
        Initialize AgenticDB.

        Args:
            storage: Storage backend (default: InMemoryStorage)
        """
        self._storage = storage or InMemoryStorage()
        self._graphs: dict[str, DependencyGraph] = {}
        self._caches: dict[str, DependencyAwareCache] = {}
        self._subscriptions: dict[str, SubscriptionManager] = {}
        self._branch_handles: dict[str, BranchHandle] = {}

        # Create default "main" branch
        self._create_default_branch()

    def _create_default_branch(self) -> None:
        """Create the default main branch."""
        main_branch = Branch(name="main", description="Default branch")
        self._storage.create_branch(main_branch)

        graph = DependencyGraph(branch_id=main_branch.id)
        self._graphs[main_branch.id] = graph

        cache = DependencyAwareCache(graph)
        self._caches[main_branch.id] = cache

        subscriptions = SubscriptionManager(graph)
        self._subscriptions[main_branch.id] = subscriptions

    # =========================================================================
    # Branch Management
    # =========================================================================

    def create_branch(
        self,
        name: str,
        description: Optional[str] = None,
        parent_branch: Optional[str] = None,
    ) -> BranchHandle:
        """
        Create a new branch.

        Args:
            name: Branch name
            description: Optional description
            parent_branch: Parent branch to fork from (default: main)

        Returns:
            BranchHandle for the new branch
        """
        # Create branch model
        branch = Branch(
            name=name,
            description=description,
        )

        if parent_branch:
            parent = self._storage.get_branch(parent_branch)
            if parent:
                branch.parent_branch_id = parent.id
                branch.fork_version = parent.head_version

        # Store branch
        self._storage.create_branch(branch)

        # Create supporting structures
        graph = DependencyGraph(branch_id=branch.id)
        self._graphs[branch.id] = graph

        cache = DependencyAwareCache(graph)
        self._caches[branch.id] = cache

        subscriptions = SubscriptionManager(graph)
        self._subscriptions[branch.id] = subscriptions

        # Create and cache handle
        handle = BranchHandle(
            branch=branch,
            storage=self._storage,
            graph=graph,
            cache=cache,
            subscriptions=subscriptions,
        )
        self._branch_handles[branch.id] = handle

        return handle

    def branch(self, branch_id: Optional[str] = None) -> BranchHandle:
        """
        Get a handle to a branch.

        Args:
            branch_id: Branch ID (default: main branch)

        Returns:
            BranchHandle for the branch
        """
        if branch_id is None:
            # Get main branch
            for bid, handle in self._branch_handles.items():
                if handle.name == "main":
                    return handle
            # Create handle for main if not cached
            main_branch = None
            for bid in self._graphs:
                branch = self._storage.get_branch(bid)
                if branch and branch.name == "main":
                    main_branch = branch
                    break
            if main_branch is None:
                raise ValueError("Main branch not found")
            branch_id = main_branch.id

        # Return cached handle or create new one
        if branch_id in self._branch_handles:
            return self._branch_handles[branch_id]

        branch = self._storage.get_branch(branch_id)
        if branch is None:
            raise ValueError(f"Branch not found: {branch_id}")

        handle = BranchHandle(
            branch=branch,
            storage=self._storage,
            graph=self._graphs[branch_id],
            cache=self._caches[branch_id],
            subscriptions=self._subscriptions[branch_id],
        )
        self._branch_handles[branch_id] = handle
        return handle

    def list_branches(self) -> list[Branch]:
        """List all branches."""
        branches = []
        for branch_id in self._graphs:
            branch = self._storage.get_branch(branch_id)
            if branch:
                branches.append(branch)
        return branches

    # =========================================================================
    # Global Queries (across branches)
    # =========================================================================

    def why(
        self,
        entity_id: str,
        branch_id: Optional[str] = None,
        max_depth: Optional[int] = None,
    ) -> CausalChain:
        """
        Trace the causal chain that led to an entity.

        Args:
            entity_id: Entity to trace
            branch_id: Branch to search (default: main)
            max_depth: Maximum traversal depth

        Returns:
            CausalChain showing the causal path
        """
        return self.branch(branch_id).why(entity_id, max_depth)

    def impact(
        self,
        entity_id: str,
        branch_id: Optional[str] = None,
        max_depth: Optional[int] = None,
        auto_invalidate: bool = False,
    ) -> ImpactResult:
        """
        Find all entities affected by a change to this entity.

        Args:
            entity_id: Entity to analyze
            branch_id: Branch to search (default: main)
            max_depth: Maximum traversal depth
            auto_invalidate: Mark affected claims as invalidated

        Returns:
            ImpactResult showing downstream dependencies
        """
        return self.branch(branch_id).impact(entity_id, max_depth, auto_invalidate)

    # =========================================================================
    # Subscriptions
    # =========================================================================

    def subscribe(
        self,
        entity_id: str,
        callback: SubscriptionCallback,
        branch_id: Optional[str] = None,
    ) -> Subscription:
        """
        Subscribe to changes on an entity.

        Args:
            entity_id: Entity to watch
            callback: Function to call on changes
            branch_id: Branch to watch (default: main)

        Returns:
            Subscription object
        """
        handle = self.branch(branch_id)
        return self._subscriptions[handle.id].subscribe_entity(entity_id, callback)

    def subscribe_type(
        self,
        entity_type: EntityType,
        callback: SubscriptionCallback,
        branch_id: Optional[str] = None,
    ) -> Subscription:
        """
        Subscribe to all entities of a type.

        Args:
            entity_type: Type to watch
            callback: Function to call on changes
            branch_id: Branch to watch (default: main)

        Returns:
            Subscription object
        """
        handle = self.branch(branch_id)
        return self._subscriptions[handle.id].subscribe_type(entity_type, callback)

    # =========================================================================
    # Utilities
    # =========================================================================

    def clear(self) -> None:
        """Clear all data (for testing)."""
        if isinstance(self._storage, InMemoryStorage):
            self._storage.clear()
        self._graphs.clear()
        self._caches.clear()
        self._subscriptions.clear()
        self._branch_handles.clear()
        self._create_default_branch()

    # =========================================================================
    # Intent-Aware Transaction API
    # =========================================================================

    def query(
        self,
        query_text: str,
        bindings: Optional[dict[str, Any]] = None,
        branch_id: Optional[str] = None,
    ) -> "QueryResult":
        """
        Execute a natural language query with Intent-aware processing.

        This is the primary interface for the Intent-Aware Transaction Pipeline.
        The query is parsed into Intent IR, bindings are resolved, and the
        operation is executed (or returned with pending state if needed).

        Args:
            query_text: Natural language query
            bindings: Optional explicit bindings for parameters
            branch_id: Branch to query (default: main)

        Returns:
            QueryResult with status and data

        Example:
            ```python
            # Complete query - executes directly
            result = db.query("show orders from last week")
            print(result.data)

            # Partial query - returns pending state
            result = db.query("show records from last month")
            if result.status == "pending_binding":
                result = db.bind(result.transaction_id, target="orders")
            ```
        """
        from .query_result import QueryResult
        from agenticdb.core.agents.transaction import CoordinatorAgent
        from agenticdb.core.agents.base.base_agent import AgentContext

        # Get coordinator (create if needed)
        if not hasattr(self, "_coordinator"):
            self._coordinator = CoordinatorAgent(
                available_tables=self._get_available_tables()
            )

        # Create context
        ctx = AgentContext(
            branch_id=branch_id or self.branch().id,
        )

        # Process through coordinator
        coordination_result = self._coordinator.run(
            ctx,
            query_text,
            bindings=bindings,
        )

        return QueryResult.from_coordination_result(coordination_result)

    def bind(
        self,
        transaction_id: str,
        **bindings: Any,
    ) -> "QueryResult":
        """
        Provide bindings for a pending transaction.

        Args:
            transaction_id: ID of the pending transaction
            **bindings: Named bindings to apply

        Returns:
            QueryResult with updated status

        Example:
            ```python
            result = db.query("show records from last month")
            if result.status == "pending_binding":
                result = db.bind(result.transaction_id, target="orders")
            ```
        """
        from .query_result import QueryResult
        from agenticdb.core.agents.base.base_agent import AgentContext

        if not hasattr(self, "_coordinator"):
            return QueryResult.error("No active coordinator")

        ctx = AgentContext(branch_id=self.branch().id)
        coordination_result = self._coordinator.bind(ctx, transaction_id, bindings)
        return QueryResult.from_coordination_result(coordination_result)

    def confirm(
        self,
        transaction_id: str,
        yes: bool = True,
    ) -> "QueryResult":
        """
        Confirm or reject a pending confirmation.

        Args:
            transaction_id: ID of the pending transaction
            yes: True to confirm, False to reject

        Returns:
            QueryResult with final status

        Example:
            ```python
            result = db.query("delete all inactive users")
            if result.status == "pending_confirmation":
                result = db.confirm(result.transaction_id, yes=True)
            ```
        """
        from .query_result import QueryResult
        from agenticdb.core.agents.base.base_agent import AgentContext

        if not hasattr(self, "_coordinator"):
            return QueryResult.error("No active coordinator")

        ctx = AgentContext(branch_id=self.branch().id)
        coordination_result = self._coordinator.confirm(ctx, transaction_id, yes)
        return QueryResult.from_coordination_result(coordination_result)

    def store(
        self,
        event_type: str,
        data: dict[str, Any],
        branch_id: Optional[str] = None,
    ) -> "QueryResult":
        """
        Store an event with automatic schema inference.

        Args:
            event_type: Type of the event (e.g., "UserRegistered")
            data: Event data payload
            branch_id: Branch to store in (default: main)

        Returns:
            QueryResult with stored event info

        Example:
            ```python
            result = db.store("UserRegistered", {
                "name": "Alice",
                "email": "alice@example.com"
            })
            ```
        """
        from .query_result import QueryResult

        try:
            event = Event(
                event_type=event_type,
                data=data,
            )

            handle = self.branch(branch_id)
            stored_event = handle.record(event)

            return QueryResult.success(
                data={"event_id": stored_event.id},
                affected_rows=1,
            )
        except Exception as e:
            return QueryResult.error(str(e))

    def transaction(self) -> "TransactionBuilder":
        """
        Start a transaction builder for fluent API.

        Returns:
            TransactionBuilder for chaining operations

        Example:
            ```python
            result = (
                db.transaction()
                  .query("show orders from last week")
                  .where(total__gt=100)
                  .limit(10)
                  .execute()
            )
            ```
        """
        from .builders import TransactionBuilder
        return TransactionBuilder(self)

    def session(self) -> "Session":
        """
        Create a session for multi-turn interactions.

        Returns:
            Session context manager

        Example:
            ```python
            with db.session() as session:
                r1 = session.query("show me users")
                r2 = session.bind(target="customers")
                r3 = session.query("filter by active")
            ```
        """
        from .session import Session
        return Session(self)

    def _get_available_tables(self) -> list[str]:
        """Get list of available tables for Intent parsing."""
        # This would typically query the schema
        # For now, return common event types
        tables = set()

        try:
            # Get event types from recorded events
            for event in self.branch().events(limit=100):
                tables.add(event.event_type.lower())
        except Exception:
            pass

        return list(tables) or ["events", "claims", "actions"]

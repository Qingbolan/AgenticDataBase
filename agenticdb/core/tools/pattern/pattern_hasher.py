"""
Pattern hashing tool for structural comparison.

Pure computation - no LLM calls. Computes structural hashes
for query patterns to enable fast matching.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ...models import (
    Intent,
    OperationType,
    Predicate,
    Pattern,
)


@dataclass
class PatternStructure:
    """
    Structural representation of a query pattern.

    Attributes:
        operation: The operation type
        table: Target table (or None if parameterized)
        predicate_structure: Structure of predicates
        has_limit: Whether LIMIT is present
        has_order: Whether ORDER BY is present
        parameter_positions: Where parameters appear
    """

    operation: OperationType
    table: Optional[str] = None
    predicate_structure: List[str] = field(default_factory=list)
    has_limit: bool = False
    has_order: bool = False
    parameter_positions: Dict[str, str] = field(default_factory=dict)

    def to_hash_string(self) -> str:
        """Convert to a string for hashing."""
        parts = [
            f"op:{self.operation.value}",
            f"table:{self.table or '?'}",
            f"predicates:{','.join(sorted(self.predicate_structure))}",
            f"limit:{self.has_limit}",
            f"order:{self.has_order}",
        ]
        return "|".join(parts)


class PatternHasher:
    """
    Compute structural hashes for query patterns.

    This is a pure computation tool (no LLM) that extracts
    structural features and computes hashes for fast matching.

    Hash Properties:
        - Similar queries produce similar hashes
        - Different structures produce different hashes
        - Parameters are normalized (values don't affect hash)

    Example:
        hasher = PatternHasher()
        hash1 = hasher.hash_intent(intent1)
        hash2 = hasher.hash_intent(intent2)
        # Same structure → same hash
    """

    def __init__(
        self,
        normalize_values: bool = True,
        include_table_in_hash: bool = False,
    ):
        """
        Initialize the pattern hasher.

        Args:
            normalize_values: Replace values with placeholders
            include_table_in_hash: Whether table name affects hash
        """
        self.normalize_values = normalize_values
        self.include_table_in_hash = include_table_in_hash

    def hash_intent(self, intent: Intent) -> str:
        """
        Compute structural hash for an Intent.

        Args:
            intent: The Intent to hash

        Returns:
            Structural hash string
        """
        structure = self.extract_structure(intent)
        return self._compute_hash(structure)

    def hash_sql(self, sql: str) -> str:
        """
        Compute structural hash for SQL.

        Args:
            sql: The SQL query

        Returns:
            Structural hash string
        """
        normalized = self._normalize_sql(sql)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def extract_structure(self, intent: Intent) -> PatternStructure:
        """
        Extract structural information from an Intent.

        Args:
            intent: The Intent to analyze

        Returns:
            PatternStructure with extracted information
        """
        # Get predicate structure
        predicate_structure = []
        for pred in intent.predicates:
            # Structure is: field.operator
            struct = f"{pred.field}.{pred.operator}"
            if pred.negate:
                struct = f"NOT({struct})"
            predicate_structure.append(struct)

        # Check for limit and order
        has_limit = "limit" in intent.bindings or intent.metadata.get("limit") is not None
        has_order = "order_by" in intent.metadata

        # Get parameter positions
        param_positions = {}
        if isinstance(intent.target, str):
            param_positions["target"] = "fixed"
        else:
            param_positions["target"] = "parameter"

        for i, pred in enumerate(intent.predicates):
            if hasattr(pred.value, "is_bound"):
                param_positions[f"pred_{i}"] = "parameter"
            else:
                param_positions[f"pred_{i}"] = "fixed"

        return PatternStructure(
            operation=intent.operation,
            table=intent.get_target_name() if self.include_table_in_hash else None,
            predicate_structure=predicate_structure,
            has_limit=has_limit,
            has_order=has_order,
            parameter_positions=param_positions,
        )

    def create_pattern(
        self,
        intent: Intent,
        sql_template: str,
    ) -> Pattern:
        """
        Create a Pattern from an Intent and SQL template.

        Args:
            intent: The source Intent
            sql_template: SQL template with {placeholders}

        Returns:
            Pattern object
        """
        structural_hash = self.hash_intent(intent)

        # Extract parameter slots from template
        parameter_slots = re.findall(r"\{(\w+)\}", sql_template)

        return Pattern(
            template=sql_template,
            structural_hash=structural_hash,
            operation=intent.operation,
            target_table=intent.get_target_name(),
            parameter_slots=parameter_slots,
            example_queries=[intent.raw_input] if intent.raw_input else [],
            confidence=intent.confidence,
        )

    def are_structurally_equivalent(
        self,
        intent1: Intent,
        intent2: Intent,
    ) -> bool:
        """
        Check if two Intents are structurally equivalent.

        Args:
            intent1: First Intent
            intent2: Second Intent

        Returns:
            True if structurally equivalent
        """
        return self.hash_intent(intent1) == self.hash_intent(intent2)

    def _compute_hash(self, structure: PatternStructure) -> str:
        """Compute hash from structure."""
        hash_string = structure.to_hash_string()
        return hashlib.sha256(hash_string.encode()).hexdigest()[:16]

    def _normalize_sql(self, sql: str) -> str:
        """
        Normalize SQL for structural comparison.

        Replaces values with placeholders while preserving structure.
        """
        normalized = sql.upper()

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        if self.normalize_values:
            # Replace string literals with placeholder
            normalized = re.sub(r"'[^']*'", "'?'", normalized)

            # Replace numeric literals with placeholder
            normalized = re.sub(r"\b\d+\.?\d*\b", "?", normalized)

            # Replace parameter placeholders with generic
            normalized = re.sub(r":\w+", ":?", normalized)
            normalized = re.sub(r"\$\d+", "?", normalized)

        # Remove table names if not included in hash
        if not self.include_table_in_hash:
            # This is a simplification - real implementation would parse SQL
            pass

        return normalized.strip()

    def extract_template(self, sql: str, parameters: Dict[str, Any]) -> str:
        """
        Extract a template from SQL with known parameters.

        Args:
            sql: The SQL with parameter placeholders
            parameters: The parameter values

        Returns:
            SQL template with {name} placeholders
        """
        template = sql

        # Replace named parameters (:name) with {name}
        for name, value in parameters.items():
            # Handle both :name and the value directly
            template = re.sub(f":{name}\\b", f"{{{name}}}", template)

            # Also try to find the value in the SQL (for positional params)
            if isinstance(value, str):
                template = template.replace(f"'{value}'", f"{{{name}}}")
            elif isinstance(value, (int, float)):
                # Only replace if it's a standalone number
                template = re.sub(f"\\b{value}\\b", f"{{{name}}}", template)

        return template

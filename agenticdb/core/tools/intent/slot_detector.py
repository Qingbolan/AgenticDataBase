"""
Slot detection tool for Intent processing.

Pure computation - no LLM calls. Detects unbound parameter slots
in natural language queries using pattern matching.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ...models import (
    Intent,
    ParameterSlot,
    SlotType,
    OperationType,
)


@dataclass
class DetectedSlot:
    """
    A slot detected in the input.

    Attributes:
        name: Slot identifier
        slot_type: Type of slot
        position: Position in input (start, end)
        raw_text: Original text that indicates this slot
        suggested_values: Possible values if determinable
        confidence: Detection confidence [0, 1]
    """

    name: str
    slot_type: SlotType
    position: Tuple[int, int] = (0, 0)
    raw_text: str = ""
    suggested_values: List[Any] = field(default_factory=list)
    confidence: float = 1.0

    def to_parameter_slot(self) -> ParameterSlot:
        """Convert to ParameterSlot model."""
        return ParameterSlot(
            name=self.name,
            slot_type=self.slot_type,
            description=f"Detected from: {self.raw_text}",
        )


class SlotDetector:
    """
    Detect unbound parameter slots in natural language queries.

    This is a pure computation tool (no LLM) that uses pattern matching
    to identify slots that need binding.

    Slot Types Detected:
        - ENTITY: Target tables/entities ("records", "data", "items")
        - TEMPORAL: Time references ("last month", "yesterday", "recent")
        - NUMERIC: Numbers and limits ("top 10", "limit 100")
        - FILTER: Filter conditions ("active", "pending", "failed")

    Example:
        detector = SlotDetector()
        slots = detector.detect("show records from last month")
        # → [DetectedSlot(name="target", type=ENTITY), DetectedSlot(name="time_range", type=TEMPORAL)]
    """

    # Patterns for detecting ambiguous entity references
    AMBIGUOUS_ENTITY_PATTERNS = [
        r"\b(records?|data|items?|entries|results?|things?|stuff)\b",
        r"\b(everything|all)\b",
        r"\b(it|them|those|these|the ones?)\b",
    ]

    # Patterns for temporal references
    TEMPORAL_PATTERNS = [
        (r"\blast\s+(day|week|month|year|hour|minute)\b", "relative"),
        (r"\b(yesterday|today|tomorrow)\b", "named"),
        (r"\b(recent|recently|latest|newest|oldest)\b", "relative"),
        (r"\b(this|current)\s+(day|week|month|year)\b", "relative"),
        (r"\b(past|previous|next)\s+(\d+)\s+(days?|weeks?|months?|years?)\b", "relative"),
        (r"\b(since|after|before|until)\s+", "boundary"),
    ]

    # Patterns for numeric references
    NUMERIC_PATTERNS = [
        (r"\btop\s+(\d+)\b", "limit"),
        (r"\b(first|last)\s+(\d+)\b", "limit"),
        (r"\blimit\s+(\d+)\b", "limit"),
        (r"\b(\d+)\s+(results?|rows?|records?|items?)\b", "count"),
        (r"\bmore\s+than\s+(\d+)\b", "threshold"),
        (r"\bless\s+than\s+(\d+)\b", "threshold"),
    ]

    # Patterns for filter conditions
    FILTER_PATTERNS = [
        (r"\b(active|inactive)\b", "status"),
        (r"\b(pending|completed|failed|cancelled)\b", "status"),
        (r"\b(new|old|updated|created|deleted)\b", "state"),
        (r"\b(enabled|disabled)\b", "toggle"),
        (r"\bwhere\s+(\w+)\s*[=<>!]+", "condition"),
        (r"\bwith\s+(\w+)\s*[=<>!]+", "condition"),
    ]

    # Operation keywords
    OPERATION_KEYWORDS = {
        OperationType.QUERY: [
            "show", "display", "get", "find", "list", "retrieve",
            "select", "fetch", "search", "lookup", "view",
        ],
        OperationType.STORE: [
            "store", "save", "insert", "add", "create", "record",
            "put", "write", "log",
        ],
        OperationType.UPDATE: [
            "update", "modify", "change", "edit", "set", "alter",
            "patch", "adjust",
        ],
        OperationType.DELETE: [
            "delete", "remove", "drop", "clear", "purge", "erase",
            "destroy", "wipe",
        ],
    }

    def __init__(self, known_entities: Optional[Set[str]] = None):
        """
        Initialize the slot detector.

        Args:
            known_entities: Set of known entity/table names for reference resolution
        """
        self.known_entities = known_entities or set()

    def detect(self, text: str) -> List[DetectedSlot]:
        """
        Detect unbound slots in the input text.

        Args:
            text: Natural language input

        Returns:
            List of detected slots
        """
        text_lower = text.lower()
        slots = []

        # Detect entity slots
        entity_slots = self._detect_entity_slots(text, text_lower)
        slots.extend(entity_slots)

        # Detect temporal slots
        temporal_slots = self._detect_temporal_slots(text, text_lower)
        slots.extend(temporal_slots)

        # Detect numeric slots
        numeric_slots = self._detect_numeric_slots(text, text_lower)
        slots.extend(numeric_slots)

        # Detect filter slots
        filter_slots = self._detect_filter_slots(text, text_lower)
        slots.extend(filter_slots)

        return slots

    def detect_operation(self, text: str) -> Optional[OperationType]:
        """
        Detect the operation type from the text.

        Args:
            text: Natural language input

        Returns:
            Detected operation type or None
        """
        text_lower = text.lower()

        for op_type, keywords in self.OPERATION_KEYWORDS.items():
            for keyword in keywords:
                if re.search(rf"\b{keyword}\b", text_lower):
                    return op_type

        return None

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Full analysis of the input text.

        Args:
            text: Natural language input

        Returns:
            Analysis result with operation, slots, and metadata
        """
        slots = self.detect(text)
        operation = self.detect_operation(text)

        # Check if target entity is resolved
        target_resolved = self._is_target_resolved(text)

        return {
            "operation": operation,
            "slots": slots,
            "target_resolved": target_resolved,
            "needs_binding": len([s for s in slots if s.slot_type == SlotType.ENTITY]) > 0 and not target_resolved,
            "slot_names": [s.name for s in slots],
        }

    def _detect_entity_slots(
        self, text: str, text_lower: str
    ) -> List[DetectedSlot]:
        """Detect entity reference slots."""
        slots = []

        for pattern in self.AMBIGUOUS_ENTITY_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                # Check if preceded by a specific entity name
                if not self._is_preceded_by_entity(text_lower, match.start()):
                    slots.append(
                        DetectedSlot(
                            name="target",
                            slot_type=SlotType.ENTITY,
                            position=(match.start(), match.end()),
                            raw_text=match.group(),
                            confidence=0.9 if match.group() in ["records", "data", "items"] else 0.7,
                        )
                    )

        return slots

    def _detect_temporal_slots(
        self, text: str, text_lower: str
    ) -> List[DetectedSlot]:
        """Detect temporal reference slots."""
        slots = []

        for pattern, category in self.TEMPORAL_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                slots.append(
                    DetectedSlot(
                        name=f"time_{category}",
                        slot_type=SlotType.TEMPORAL,
                        position=(match.start(), match.end()),
                        raw_text=match.group(),
                        confidence=0.95,
                    )
                )

        # Deduplicate - prefer the most specific match
        return self._deduplicate_slots(slots)

    def _detect_numeric_slots(
        self, text: str, text_lower: str
    ) -> List[DetectedSlot]:
        """Detect numeric slots."""
        slots = []

        for pattern, category in self.NUMERIC_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                # Extract the numeric value if present
                groups = match.groups()
                value = None
                for g in groups:
                    if g and g.isdigit():
                        value = int(g)
                        break

                slots.append(
                    DetectedSlot(
                        name=f"limit" if category == "limit" else f"numeric_{category}",
                        slot_type=SlotType.NUMERIC,
                        position=(match.start(), match.end()),
                        raw_text=match.group(),
                        suggested_values=[value] if value else [],
                        confidence=1.0 if value else 0.8,
                    )
                )

        return slots

    def _detect_filter_slots(
        self, text: str, text_lower: str
    ) -> List[DetectedSlot]:
        """Detect filter condition slots."""
        slots = []

        for pattern, category in self.FILTER_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                slots.append(
                    DetectedSlot(
                        name=f"filter_{category}",
                        slot_type=SlotType.FILTER,
                        position=(match.start(), match.end()),
                        raw_text=match.group(),
                        confidence=0.9,
                    )
                )

        return slots

    def _is_preceded_by_entity(self, text: str, position: int) -> bool:
        """Check if position is preceded by a known entity name."""
        if position == 0:
            return False

        # Look at the word before
        before = text[:position].strip()
        words = before.split()
        if not words:
            return False

        last_word = words[-1]
        return last_word in self.known_entities

    def _is_target_resolved(self, text: str) -> bool:
        """Check if the text has a resolved target entity."""
        text_lower = text.lower()

        # Check for known entities
        for entity in self.known_entities:
            if entity.lower() in text_lower:
                return True

        # Check for explicit "from X" pattern
        from_match = re.search(r"\bfrom\s+(\w+)", text_lower)
        if from_match:
            return True

        # Check for "in X" pattern
        in_match = re.search(r"\bin\s+(\w+)\s+(table|collection|database)", text_lower)
        if in_match:
            return True

        return False

    def _deduplicate_slots(self, slots: List[DetectedSlot]) -> List[DetectedSlot]:
        """Remove duplicate slots, preferring higher confidence."""
        seen_names: Dict[str, DetectedSlot] = {}

        for slot in slots:
            if slot.name not in seen_names:
                seen_names[slot.name] = slot
            elif slot.confidence > seen_names[slot.name].confidence:
                seen_names[slot.name] = slot

        return list(seen_names.values())

    def create_intent_slots(
        self, text: str, operation: Optional[OperationType] = None
    ) -> Tuple[List[ParameterSlot], bool]:
        """
        Create ParameterSlot objects from detected slots.

        Args:
            text: Natural language input
            operation: Optional pre-detected operation type

        Returns:
            Tuple of (slots, is_complete) where is_complete indicates
            if all necessary slots are bound
        """
        detected = self.detect(text)
        analysis = self.analyze(text)

        # If no operation detected, add operation slot
        if operation is None and analysis["operation"] is None:
            detected.insert(
                0,
                DetectedSlot(
                    name="operation",
                    slot_type=SlotType.STRING,
                    confidence=0.5,
                )
            )

        parameter_slots = [d.to_parameter_slot() for d in detected]
        is_complete = not analysis["needs_binding"]

        return parameter_slots, is_complete

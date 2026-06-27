from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

import numpy as np


@dataclass
class ContextBodyRecord:
    """
    Persistent record stored in the vector database.

    This is the only representation of a context body that ever leaves memory.
    It contains exactly what a vector DB needs: the centroid vector plus a flat
    set of scalar metadata fields. No relational links, no history, no membership.

    The centroid IS the identity of the body — similarity between records is
    computed at query time from vector distance, not from stored foreign keys.

    Relationships (parent, resonance) are recovered at query time:
        - Parent   → most similar record with an earlier created_at
        - Resonance → records within a cosine similarity threshold at query time

    Contrast with ContextBody (context_body.py), which is the ephemeral in-memory
    representation used during a conversation for clustering and force computation.
    That object is never persisted.
    """

    id: UUID = field(default_factory=uuid4)
    centroid: np.ndarray = field(default_factory=lambda: np.array([]))
    mass: float = 0.0
    stability: float = 0.0
    domain: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)

    # ------------------------------------------------------------------
    # Serialization helpers (for vector DB metadata payloads)
    # ------------------------------------------------------------------

    def to_metadata(self) -> dict:
        """Flat dict of scalar metadata — suitable for any vector DB payload."""
        return {
            "id": str(self.id),
            "mass": self.mass,
            "stability": self.stability,
            "domain": self.domain,
            "created_at": self.created_at.isoformat(),
            "last_seen": self.last_seen.isoformat(),
        }

    @classmethod
    def from_metadata(cls, centroid: np.ndarray, metadata: dict) -> ContextBodyRecord:
        """Reconstruct a record from a vector DB search result."""
        return cls(
            id=UUID(metadata["id"]),
            centroid=centroid,
            mass=float(metadata.get("mass", 0.0)),
            stability=float(metadata.get("stability", 0.0)),
            domain=str(metadata.get("domain", "")),
            created_at=datetime.fromisoformat(metadata["created_at"])
            if "created_at" in metadata
            else datetime.utcnow(),
            last_seen=datetime.fromisoformat(metadata["last_seen"])
            if "last_seen" in metadata
            else datetime.utcnow(),
        )

from __future__ import annotations

import json
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
        - Parent    → most similar record with an earlier created_at
        - Resonance → partners recorded explicitly via ContextBodyStore.record_resonance()

    resonance_partners maps partner record ID (str UUID) → resonance score [0, 1].
    Score accumulates across sessions when two bodies are co-active and decays
    if they stop co-occurring. High-score pairs get a Lagrange midpoint force
    in GravitationalSampler in addition to their individual forces.

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

    # resonance_partners: partner_record_id → score in [0, 1]
    # populated by ContextBodyStore.record_resonance() and persisted as JSON
    resonance_partners: dict[str, float] = field(default_factory=dict)

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
            "resonance_partners": json.dumps(self.resonance_partners),
        }

    @classmethod
    def from_metadata(cls, centroid: np.ndarray, metadata: dict) -> ContextBodyRecord:
        """Reconstruct a record from a vector DB search result."""
        raw_partners = metadata.get("resonance_partners", "{}")
        try:
            resonance_partners = json.loads(raw_partners) if isinstance(raw_partners, str) else raw_partners
        except (json.JSONDecodeError, TypeError):
            resonance_partners = {}

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
            resonance_partners=resonance_partners,
        )

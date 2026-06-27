from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

import numpy as np


@dataclass
class ContextBody:
    """
    Ephemeral in-memory representation of a gravitational body.

    Lives only for the duration of a conversation. Created by IncrementalDBSCAN
    when a cluster stabilizes, used by GravitationalSampler to compute forces,
    and discarded at conversation end. Only the centroid + scalar metadata
    are ever written to persistent storage (via ContextBodyRecord).

    Contrast with ContextBodyRecord (context_body_record.py), which is the
    lightweight persistent form stored in the vector database.

    Mass hierarchy (for classify()):
        Black hole   — dominant, inescapable theme (e.g. system prompt constraint)
        Neutron star — rare but highly specific, dense context
        Planet       — moderate, stable topic
        Moon         — sub-topic orbiting a planet
        Asteroid     — passing mention, weak influence
    """

    # Identity
    id: UUID = field(default_factory=uuid4)
    domain: str = ""

    # Position in embedding space
    centroid: np.ndarray = field(default_factory=lambda: np.array([]))
    centroid_velocity: np.ndarray = field(default_factory=lambda: np.array([]))
    covariance: np.ndarray = field(default_factory=lambda: np.array([]))

    # Mass properties
    mass: float = 0.0       # derived from cluster density
    density: float = 0.0    # tokens per unit volume in embedding space
    stability: float = 0.0  # 0-1, how consistent the centroid has been

    # Ephemeral membership — used for clustering operations, never persisted
    member_tokens: set[int] = field(default_factory=set)
    orbital_radii: dict[int, float] = field(default_factory=dict)
    orbital_velocities: dict[int, np.ndarray] = field(default_factory=dict)

    # Lineage — ephemeral, tracks splits within a conversation session
    # Relationships across sessions are recovered at query time via similarity
    parent_ids: list[UUID] = field(default_factory=list)

    def accrete(self, token_id: int, token_embedding: np.ndarray) -> None:
        """
        Add a new token to this body, updating centroid and density.
        Analogous to a body gaining mass from nearby matter.
        """
        n = len(self.member_tokens)
        self.member_tokens.add(token_id)

        prev_centroid = self.centroid.copy()
        self.centroid = (self.centroid * n + token_embedding) / (n + 1)
        self.centroid_velocity = self.centroid - prev_centroid

        r = float(1 - np.dot(token_embedding, self.centroid) / (
            np.linalg.norm(token_embedding) * np.linalg.norm(self.centroid) + 1e-8
        ))
        self.orbital_radii[token_id] = r

    def classify(self) -> str:
        """
        Classify this body by mass tier.
        Thresholds are illustrative — tune per domain and embedding scale.
        """
        if self.mass > 1000:
            return "black_hole"
        elif self.mass > 500:
            return "neutron_star"
        elif self.mass > 100:
            return "planet"
        elif self.mass > 10:
            return "moon"
        else:
            return "asteroid"

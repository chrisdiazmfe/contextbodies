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
        Black hole   -- dominant, inescapable theme (e.g. system prompt constraint)
        Neutron star -- rare but highly specific, dense context
        Planet       -- moderate, stable topic
        Moon         -- sub-topic orbiting a planet
        Asteroid     -- passing mention, weak influence
    """

    # Identity
    id: UUID = field(default_factory=uuid4)
    domain: str = ""

    # Position in embedding space
    centroid: np.ndarray = field(default_factory=lambda: np.array([]))
    centroid_velocity: np.ndarray = field(default_factory=lambda: np.array([]))
    covariance: np.ndarray = field(default_factory=lambda: np.array([]))

    # Mass properties
    mass: float = 0.0
    density: float = 0.0
    stability: float = 0.0

    # Ephemeral membership
    member_tokens: set[int] = field(default_factory=set)
    orbital_radii: dict[int, float] = field(default_factory=dict)
    orbital_velocities: dict[int, np.ndarray] = field(default_factory=dict)

    # Lineage
    parent_ids: list = field(default_factory=list)

    def accrete(
        self,
        token_id: int,
        token_embedding: np.ndarray,
        token_mass: float = 1.0,
    ) -> None:
        """Add a new token to this body, updating centroid and mass."""
        n = len(self.member_tokens)
        self.member_tokens.add(token_id)

        if n == 0:
            self.centroid = token_embedding.copy().astype(float)
            self.centroid_velocity = np.zeros_like(self.centroid)
        else:
            prev_centroid = self.centroid.copy()
            self.centroid = (self.centroid * n + token_embedding) / (n + 1)
            self.centroid_velocity = self.centroid - prev_centroid

        self.mass += token_mass

        c_norm = np.linalg.norm(self.centroid)
        t_norm = np.linalg.norm(token_embedding)
        if c_norm > 1e-8 and t_norm > 1e-8:
            r = float(1.0 - np.dot(token_embedding, self.centroid) / (t_norm * c_norm))
        else:
            r = 0.0
        self.orbital_radii[token_id] = r

    def classify(self) -> str:
        """Classify this body by mass tier."""
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

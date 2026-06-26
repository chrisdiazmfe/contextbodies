from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

import numpy as np


@dataclass
class ContextBody:
    """
    Represents a gravitational body in semantic embedding space.

    Context bodies are emergent clusters of semantically related tokens
    that exert gravitational influence on token sampling. They are analogous
    to celestial bodies — black holes, planets, moons, asteroids — depending
    on their mass and density.

    Mass hierarchy:
        Black hole  — dominant, inescapable theme (e.g. system prompt constraint)
        Neutron star — rare but highly specific, dense context
        Planet       — moderate, stable topic
        Moon         — sub-topic orbiting a planet
        Asteroid     — passing mention, weak influence
    """

    # Identity
    id: UUID = field(default_factory=uuid4)
    domain: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)

    # Position in embedding space
    centroid: np.ndarray = field(default_factory=lambda: np.array([]))
    centroid_velocity: np.ndarray = field(default_factory=lambda: np.array([]))
    covariance: np.ndarray = field(default_factory=lambda: np.array([]))

    # Mass properties
    mass: float = 0.0          # derived from density × weight norms
    density: float = 0.0       # tokens per unit volume in embedding space
    stability: float = 0.0     # 0-1, how consistent centroid has been over time

    # Membership
    member_tokens: set[int] = field(default_factory=set)
    orbital_radii: dict[int, float] = field(default_factory=dict)
    orbital_velocities: dict[int, np.ndarray] = field(default_factory=dict)

    # Relationships (lineage + resonance)
    parent_ids: list[UUID] = field(default_factory=list)
    child_ids: list[UUID] = field(default_factory=list)
    resonant_ids: list[UUID] = field(default_factory=list)

    # History (for decay modeling and drift tracking)
    mass_history: list[tuple[datetime, float]] = field(default_factory=list)
    centroid_history: list[tuple[datetime, np.ndarray]] = field(default_factory=list)

    def accrete(self, token_id: int, token_embedding: np.ndarray) -> None:
        """
        Add a new token to this body, updating centroid, mass, and density.
        Analogous to a body gaining mass from nearby matter.
        """
        n = len(self.member_tokens)
        self.member_tokens.add(token_id)

        # update centroid incrementally
        prev_centroid = self.centroid.copy()
        self.centroid = (self.centroid * n + token_embedding) / (n + 1)
        self.centroid_velocity = self.centroid - prev_centroid

        # update orbital radius for this token
        r = float(1 - np.dot(token_embedding, self.centroid) / (
            np.linalg.norm(token_embedding) * np.linalg.norm(self.centroid) + 1e-8
        ))
        self.orbital_radii[token_id] = r

        # record history
        now = datetime.utcnow()
        self.last_seen = now
        self.mass_history.append((now, self.mass))
        self.centroid_history.append((now, self.centroid.copy()))

    def apply_decay(self, decay_rate: float, as_of: datetime) -> None:
        """
        Reduce mass based on time elapsed since last seen.
        Bodies not reinforced by recent context lose influence over time.
        """
        elapsed = (as_of - self.last_seen).total_seconds()
        self.mass *= max(0.0, 1.0 - decay_rate * elapsed)

    def classify(self) -> str:
        """
        Classify this body based on its mass.
        Thresholds are illustrative and should be tuned per domain.
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

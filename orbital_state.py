from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OrbitalState:
    """
    Tracks the trajectory of the current context vector in embedding space.

    Analogous to an object moving through a gravitational field:
        position     — where in semantic space the context currently sits
        velocity     — the direction and speed the context is moving
        acceleration — how rapidly the trajectory is changing

    High momentum (sustained velocity in one direction) means the context
    resists gravitational deflection from new context bodies.
    Sudden acceleration indicates a topic shift.
    """

    position: np.ndarray      # current position in embedding space [D]
    velocity: np.ndarray      # change in position per token [D]
    acceleration: np.ndarray  # change in velocity per token [D]

    @classmethod
    def initialize(cls, embedding: np.ndarray) -> OrbitalState:
        zeros = np.zeros_like(embedding)
        return cls(position=embedding.copy(), velocity=zeros, acceleration=zeros)

    def update(self, new_position: np.ndarray) -> None:
        """
        Advance the orbital state given the embedding of the latest token.
        """
        new_velocity = new_position - self.position
        self.acceleration = new_velocity - self.velocity
        self.velocity = new_velocity
        self.position = new_position.copy()

    @property
    def momentum(self) -> float:
        """Scalar magnitude of current velocity — resistance to gravitational deflection."""
        return float(np.linalg.norm(self.velocity))

    @property
    def speed_of_change(self) -> float:
        """Scalar magnitude of acceleration — how rapidly the topic is shifting."""
        return float(np.linalg.norm(self.acceleration))

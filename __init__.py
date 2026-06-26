"""
contextbodies — gravitational sampling for LLMs

Replaces temperature-based token sampling with a physics-inspired gravitational
field model. Semantic clusters (context bodies) emerge from the token stream
and exert gravitational influence on the sampling distribution.

Core concepts:
    ContextBody         — a gravitational body in embedding space (black hole,
                          planet, moon, asteroid, etc.) with mass, centroid,
                          velocity, and orbital membership.
    ContextBodyStore    — persistent store for recorded bodies; two-layer
                          hot (FAISS) + cold (vector DB) architecture.
    OrbitalState        — tracks position, velocity, and acceleration of the
                          current context vector through embedding space.
    IncrementalDBSCAN   — online clustering that discovers emergent context
                          bodies as tokens arrive.
    GravitationalSampler — the sampling engine; replaces temperature with
                          gravitational force computed from active bodies.

Quickstart:
    from contextbodies import GravitationalSampler, ContextBodyStore
    from contextbodies.generate import generate

    store = ContextBodyStore(embedding_dim=768)
    sampler = GravitationalSampler(body_store=store, G=1.0, domain="ml")

    text = generate(model, tokenizer, prompt="Tell me about transformers",
                    sampler=sampler, max_tokens=200)
"""

from context_body import ContextBody
from context_body_store import ContextBodyStore
from gravitational_sampler import GravitationalSampler
from incremental_dbscan import IncrementalDBSCAN
from orbital_state import OrbitalState

__all__ = [
    "ContextBody",
    "ContextBodyStore",
    "GravitationalSampler",
    "IncrementalDBSCAN",
    "OrbitalState",
]

"""
contextbodies — gravitational sampling for LLMs

Replaces temperature-based token sampling with a physics-inspired gravitational
field model. Semantic clusters (context bodies) emerge from the token stream
and exert gravitational influence on the sampling distribution.

Core concepts:
    ContextBody         — ephemeral in-memory body used during a conversation
                          for clustering and force computation. Never persisted.
    ContextBodyRecord   — lightweight persistent record stored in the vector DB.
                          Centroid + scalar metadata only. No relational fields.
    ContextBodyStore    — thin wrapper around a VectorBackend. Three operations:
                          record(), query_nearby(), decay().
    VectorBackend       — protocol for swappable storage backends.
    FAISSBackend        — default in-memory backend using FAISS IndexIDMap.
    OrbitalState        — tracks position, velocity, and acceleration of the
                          current context vector through embedding space.
    IncrementalDBSCAN   — online clustering that discovers emergent context
                          bodies as tokens arrive.
    GravitationalSampler — the sampling engine; replaces temperature with
                          gravitational force computed from active bodies.
    DomainClassifier    — infers the active domain from the token embedding
                          stream via EMA context direction + nearest anchor.
    AdaptiveG           — adaptive gravitational constant. Adjusts G each step
                          from body mass normalization, escape rate feedback,
                          and per-domain multipliers.

Quickstart:
    from contextbodies import GravitationalSampler, ContextBodyStore
    from contextbodies.generate import generate

    store = ContextBodyStore(embedding_dim=768)
    sampler = GravitationalSampler(body_store=store, G=1.0, domain="ml")

    text = generate(model, tokenizer, prompt="Tell me about transformers",
                    sampler=sampler, max_tokens=200)

Swapping to Qdrant (production):
    from contextbodies import ContextBodyStore, QdrantBackend
    from qdrant_client import QdrantClient

    client = QdrantClient(host="localhost", port=6333)
    store = ContextBodyStore(embedding_dim=768, backend=QdrantBackend(client))
"""

from adaptive_g import AdaptiveG
from context_body import ContextBody
from context_body_record import ContextBodyRecord
from context_body_store import ContextBodyStore
from domain_classifier import DomainClassifier
from gravitational_sampler import GravitationalSampler
from incremental_dbscan import IncrementalDBSCAN
from orbital_state import OrbitalState
from vector_backend import VectorBackend, FAISSBackend, QdrantBackend

__all__ = [
    "AdaptiveG",
    "ContextBody",
    "ContextBodyRecord",
    "ContextBodyStore",
    "DomainClassifier",
    "FAISSBackend",
    "GravitationalSampler",
    "IncrementalDBSCAN",
    "OrbitalState",
    "QdrantBackend",
    "VectorBackend",
]

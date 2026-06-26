from __future__ import annotations

from datetime import datetime
from uuid import UUID

import numpy as np
import faiss

from context_body import ContextBody


class ContextBodyStore:
    """
    Persistent store for recorded context bodies.

    Two-layer architecture:
        Hot layer  — in-memory dict + FAISS index for active/recent bodies.
                     Used during inference for fast gravitational lookups.
        Cold layer — vector database (Pinecone, Weaviate, pgvector, etc.)
                     for the full historical corpus. Queried at conversation
                     start to seed the hot layer.

    Bodies flow:
        Emergent (DBSCAN) → stable → record() → hot layer → cold layer
        Cold layer → query_nearby() → hot layer → inference
    """

    def __init__(
        self,
        embedding_dim: int,
        decay_rate: float = 1e-5,
        extinction_threshold: float = 0.01,
    ):
        self.embedding_dim = embedding_dim
        self.decay_rate = decay_rate
        self.extinction_threshold = extinction_threshold

        # primary storage
        self.bodies: dict[UUID, ContextBody] = {}

        # FAISS index — inner product on L2-normalized vectors = cosine similarity
        self.embedding_index = faiss.IndexFlatIP(embedding_dim)
        self.index_id_map: dict[int, UUID] = {}   # faiss position → UUID
        self._next_index_pos: int = 0

        # domain partitioning
        self.domain_index: dict[str, list[UUID]] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def record(self, body: ContextBody) -> UUID:
        """
        Persist a newly stabilized emergent body.
        Checks for near-duplicates (cosine similarity > 0.95) before inserting.
        """
        if len(self.bodies) > 0:
            nearby = self.query_nearby(body.centroid, body.domain, k=1)
            if nearby and nearby[0][1] < 0.05:   # cosine distance < 0.05 = near-duplicate
                # merge into existing body instead of inserting
                existing_body, _ = nearby[0]
                for token_id, emb in zip(body.member_tokens,
                                         [body.centroid] * len(body.member_tokens)):
                    existing_body.accrete(token_id, emb)
                return existing_body.id

        # insert new body
        self.bodies[body.id] = body

        # add normalized centroid to FAISS index
        norm_centroid = body.centroid / (np.linalg.norm(body.centroid) + 1e-8)
        self.embedding_index.add(norm_centroid.reshape(1, -1).astype(np.float32))
        self.index_id_map[self._next_index_pos] = body.id
        self._next_index_pos += 1

        # update domain index
        self.domain_index.setdefault(body.domain, []).append(body.id)

        return body.id

    def query_nearby(
        self,
        embedding: np.ndarray,
        domain: str,
        k: int = 10,
        mass_threshold: float = 0.0,
    ) -> list[tuple[ContextBody, float]]:
        """
        Find k nearest recorded bodies to a given embedding.
        Returns (body, cosine_distance) pairs sorted by gravitational influence
        (mass / distance²).
        """
        if len(self.bodies) == 0:
            return []

        norm_emb = embedding / (np.linalg.norm(embedding) + 1e-8)
        k_search = min(k * 2, len(self.bodies))   # oversample, then filter by domain

        similarities, indices = self.embedding_index.search(
            norm_emb.reshape(1, -1).astype(np.float32), k_search
        )

        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0:
                continue
            body_id = self.index_id_map.get(int(idx))
            if body_id is None:
                continue
            body = self.bodies.get(body_id)
            if body is None:
                continue
            if domain and body.domain and body.domain != domain:
                continue
            if body.mass < mass_threshold:
                continue
            cosine_dist = float(1 - sim)
            results.append((body, cosine_dist))

        # sort by gravitational influence: mass / r²
        results.sort(key=lambda x: x[0].mass / (x[1] ** 2 + 1e-8), reverse=True)
        return results[:k]

    def update(self, body_id: UUID, new_tokens: list[tuple[int, np.ndarray]]) -> None:
        """
        Accrete new tokens onto an existing body.
        Updates centroid, mass, density, stability.
        """
        body = self.bodies.get(body_id)
        if body is None:
            return
        for token_id, embedding in new_tokens:
            body.accrete(token_id, embedding)
        body.density = len(body.member_tokens) / (
            np.trace(body.covariance) + 1e-8
        ) if body.covariance.size > 0 else float(len(body.member_tokens))

    def decay(self, as_of: datetime | None = None) -> list[UUID]:
        """
        Apply mass decay to all bodies. Returns IDs of extinct bodies.
        """
        as_of = as_of or datetime.utcnow()
        extinct = []
        for body_id, body in list(self.bodies.items()):
            body.apply_decay(self.decay_rate, as_of)
            if body.mass < self.extinction_threshold:
                extinct.append(body_id)
                del self.bodies[body_id]
        return extinct

    def merge(self, body_a_id: UUID, body_b_id: UUID) -> UUID:
        """
        Combine two bodies that have drifted together.
        Preserves lineage in child/parent relationships.
        """
        a = self.bodies[body_a_id]
        b = self.bodies[body_b_id]

        n_a = len(a.member_tokens)
        n_b = len(b.member_tokens)
        total = n_a + n_b

        merged = ContextBody(
            domain=a.domain,
            centroid=(a.centroid * n_a + b.centroid * n_b) / total,
            mass=a.mass + b.mass,
            density=(a.density + b.density) / 2,
            member_tokens=a.member_tokens | b.member_tokens,
            orbital_radii={**a.orbital_radii, **b.orbital_radii},
            parent_ids=[body_a_id, body_b_id],
        )
        merged.centroid_velocity = np.zeros_like(merged.centroid)

        a.child_ids.append(merged.id)
        b.child_ids.append(merged.id)

        self.record(merged)
        return merged.id

    def fragment(self, body_id: UUID, clusters: list[list[int]]) -> list[UUID]:
        """
        Split a body that has grown internally inconsistent into sub-bodies.
        Returns IDs of newly created child bodies.
        """
        parent = self.bodies[body_id]
        child_ids = []

        for cluster_tokens in clusters:
            child = ContextBody(
                domain=parent.domain,
                member_tokens=set(cluster_tokens),
                mass=parent.mass / len(clusters),
                parent_ids=[body_id],
            )
            child_ids.append(self.record(child))

        parent.child_ids.extend(child_ids)
        return child_ids

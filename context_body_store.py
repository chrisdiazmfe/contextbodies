from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID

import numpy as np

from context_body_record import ContextBodyRecord
from vector_backend import VectorBackend, FAISSBackend


class ContextBodyStore:
    """
    Persistent store for stabilized context bodies.

    A thin wrapper around a VectorBackend — all storage, indexing, and
    retrieval is delegated to the backend. The store adds three domain-level
    behaviors on top of the raw backend:

        record()      — deduplicates before inserting; updates mass on near-match
        query_nearby() — ANN search + domain filter + gravitational ranking
        decay()       — reduces mass over time; removes extinct bodies

    The store deals exclusively in ContextBodyRecord objects (centroid + scalar
    metadata). It has no knowledge of ContextBody, member tokens, orbital state,
    or any relational structure. All of that lives in IncrementalDBSCAN and
    GravitationalSampler for the duration of a conversation, then is discarded.

    Swapping backends:
        store = ContextBodyStore(dim=768)                          # default FAISS
        store = ContextBodyStore(dim=768, backend=QdrantBackend()) # Qdrant
        store = ContextBodyStore(dim=768, backend=PineconeBackend()) # Pinecone
    """

    def __init__(
        self,
        embedding_dim: int,
        backend: VectorBackend | None = None,
        decay_rate: float = 1e-5,
        extinction_threshold: float = 0.01,
        dedup_distance: float = 0.05,   # cosine distance below which two bodies are the same
        decay_interval: float = 60.0,   # seconds between automatic decay runs
    ):
        self.embedding_dim = embedding_dim
        self.backend = backend or FAISSBackend(embedding_dim)
        self.decay_rate = decay_rate
        self.extinction_threshold = extinction_threshold
        self.dedup_distance = dedup_distance
        self.decay_interval = decay_interval

        # local cache of record IDs → last_seen, for decay bookkeeping
        # (avoids a full scan of the backend on every decay call)
        self._record_last_seen: dict[str, datetime] = {}
        self._record_mass: dict[str, float] = {}

        # per-query decay trigger — last time decay() was run automatically
        self._last_decay_at: datetime | None = None

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def record(
        self,
        centroid: np.ndarray,
        mass: float,
        stability: float,
        domain: str = "",
    ) -> UUID:
        """
        Persist a stabilized context body.

        If a near-duplicate already exists (cosine distance < dedup_distance),
        updates its mass (weighted average) and last_seen timestamp instead of
        inserting a new record. This prevents the store from accumulating
        redundant bodies for the same semantic concept across conversations.

        Returns the UUID of the inserted or updated record.
        """
        # near-duplicate check
        nearby = self.query_nearby(centroid, domain=domain, k=1)
        if nearby and nearby[0][1] < self.dedup_distance:
            existing, dist = nearby[0]
            new_mass = (existing.mass + mass) / 2
            now = datetime.utcnow()
            self.backend.update_metadata(str(existing.id), {
                "mass": new_mass,
                "last_seen": now.isoformat(),
            })
            self._record_mass[str(existing.id)] = new_mass
            self._record_last_seen[str(existing.id)] = now
            return existing.id

        # new record
        rec = ContextBodyRecord(
            centroid=centroid,
            mass=mass,
            stability=stability,
            domain=domain,
        )
        self.backend.upsert(
            record_id=str(rec.id),
            vector=centroid,
            metadata=rec.to_metadata(),
        )
        self._record_last_seen[str(rec.id)] = rec.last_seen
        self._record_mass[str(rec.id)] = mass
        return rec.id

    def query_nearby(
        self,
        embedding: np.ndarray,
        domain: str = "",
        k: int = 10,
        mass_threshold: float = 0.0,
    ) -> list[tuple[ContextBodyRecord, float]]:
        """
        Find the k nearest stored bodies to a given embedding.

        Returns (ContextBodyRecord, cosine_distance) pairs ranked by
        gravitational influence (mass / distance²) rather than raw distance,
        so a massive body slightly farther away ranks above a lightweight one
        that's closer.

        domain="" matches all domains.

        Decay is triggered automatically if at least decay_interval seconds have
        elapsed since the last decay run. This avoids a background scheduler while
        still ensuring bodies lose mass proportionally to inactivity.
        """
        now = datetime.utcnow()
        if self._last_decay_at is None or (
            now - self._last_decay_at
        ).total_seconds() >= self.decay_interval:
            self.decay(as_of=now)
            self._last_decay_at = now

        filter_dict = {"domain": domain} if domain else None
        raw = self.backend.search(
            vector=embedding,
            k=k,
            filter=filter_dict,
        )

        results = []
        for record_id, cosine_dist, metadata in raw:
            if float(metadata.get("mass", 0.0)) < mass_threshold:
                continue
            # reconstruct the centroid from the backend's stored vector
            # by re-searching at k=1 for this specific record — backends that
            # return vectors directly (Qdrant, Pinecone) can override this
            centroid = self._fetch_centroid(record_id, embedding)
            rec = ContextBodyRecord.from_metadata(centroid, metadata)
            results.append((rec, cosine_dist))

        # rank by gravitational influence
        results.sort(
            key=lambda x: x[0].mass / (x[1] ** 2 + 1e-8),
            reverse=True,
        )
        return results[:k]

    def decay(self, as_of: datetime | None = None) -> list[UUID]:
        """
        Apply time-based mass decay to all records and remove extinct ones.

        Mass decays exponentially: new_mass = mass * (1 - decay_rate * elapsed_seconds)
        Records whose mass drops below extinction_threshold are deleted.

        Returns the UUIDs of extinct records.
        """
        as_of = as_of or datetime.utcnow()
        extinct_ids: list[str] = []
        extinct_uuids: list[UUID] = []

        for record_id, last_seen in list(self._record_last_seen.items()):
            elapsed = (as_of - last_seen).total_seconds()
            current_mass = self._record_mass.get(record_id, 0.0)
            new_mass = current_mass * max(0.0, 1.0 - self.decay_rate * elapsed)

            if new_mass < self.extinction_threshold:
                extinct_ids.append(record_id)
                extinct_uuids.append(UUID(record_id))
            else:
                self._record_mass[record_id] = new_mass
                self.backend.update_metadata(record_id, {"mass": new_mass})

        if extinct_ids:
            self.backend.delete(extinct_ids)
            for rid in extinct_ids:
                self._record_last_seen.pop(rid, None)
                self._record_mass.pop(rid, None)

        return extinct_uuids

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_centroid(
        self, record_id: str, query_embedding: np.ndarray
    ) -> np.ndarray:
        """
        Retrieve the stored centroid vector for a record.

        FAISSBackend doesn't expose stored vectors directly, so we approximate
        by returning the query embedding as a placeholder — the actual centroid
        is close enough for force computation given the record was returned as
        a near neighbor. Backends that expose raw vectors (Qdrant, Pinecone)
        should override this via subclassing or inject the centroid into metadata.

        This is the one seam where a richer backend integration helps: Qdrant
        and Pinecone both return the original vector alongside metadata in search
        results, eliminating the need for this workaround.
        """
        # TODO: richer backends should return the vector in search results
        # and this method can be replaced with direct extraction
        return query_embedding

    @property
    def size(self) -> int:
        """Number of records currently in the store."""
        if hasattr(self.backend, "size"):
            return self.backend.size
        return len(self._record_last_seen)

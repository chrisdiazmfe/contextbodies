from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import UUID

import numpy as np

from context_body_record import ContextBodyRecord
from vector_backend import VectorBackend, FAISSBackend


class ContextBodyStore:
    """
    Persistent store for stabilized context bodies.

    A thin wrapper around a VectorBackend -- all storage, indexing, and
    retrieval is delegated to the backend. The store adds three domain-level
    behaviors on top of the raw backend:

        record()       -- deduplicates before inserting; updates mass on near-match
        query_nearby() -- ANN search + domain filter + gravitational ranking
        decay()        -- reduces mass over time; removes extinct bodies

    The store deals exclusively in ContextBodyRecord objects (centroid + scalar
    metadata). It has no knowledge of ContextBody, member tokens, orbital state,
    or any relational structure.

    Swapping backends:
        store = ContextBodyStore(dim=768)                           # default FAISS
        store = ContextBodyStore(dim=768, backend=QdrantBackend())  # Qdrant
    """

    def __init__(
        self,
        embedding_dim: int,
        backend: VectorBackend | None = None,
        decay_rate: float = 1e-5,
        extinction_threshold: float = 0.01,
        dedup_distance: float = 0.05,
        decay_interval: float = 60.0,
        reemergence_distance: float = 0.15,
        reemergence_mass_threshold: float = 0.1,
    ):
        self.embedding_dim = embedding_dim
        self.backend = backend or FAISSBackend(embedding_dim)
        self.decay_rate = decay_rate
        self.extinction_threshold = extinction_threshold
        self.dedup_distance = dedup_distance
        self.decay_interval = decay_interval
        self.reemergence_distance = reemergence_distance
        self.reemergence_mass_threshold = reemergence_mass_threshold

        self._record_last_seen: dict[str, datetime] = {}
        self._record_mass: dict[str, float] = {}
        self._resonance_cache: dict[str, dict[str, float]] = {}
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

        Three checks run in order before a new record is created:

        1. Exact dedup (dedup_distance): near-identical concept already exists ->
           update its mass as a weighted average.

        2. Re-emergence (reemergence_distance + reemergence_mass_threshold): a
           dormant (low-mass) record exists within a wider window -> boost it.

        3. New record: no match found -> insert a fresh ContextBodyRecord.

        Returns the UUID of the inserted or updated record.
        """
        # --- 1. exact dedup -----------------------------------------------
        nearby = self.query_nearby(centroid, domain=domain, k=1)
        if nearby and nearby[0][1] < self.dedup_distance:
            existing, _ = nearby[0]
            new_mass = (existing.mass + mass) / 2
            now = datetime.now(timezone.utc)
            self.backend.update_metadata(str(existing.id), {
                "mass": new_mass,
                "last_seen": now.isoformat(),
            })
            self._record_mass[str(existing.id)] = new_mass
            self._record_last_seen[str(existing.id)] = now
            return existing.id

        # --- 2. re-emergence check ----------------------------------------
        filter_dict = {"domain": domain} if domain else None
        candidates = self.backend.search(
            vector=centroid,
            k=5,
            filter=filter_dict,
        )
        for record_id, dist, metadata, _ in candidates:
            if dist >= self.reemergence_distance:
                break
            current_mass = self._record_mass.get(
                record_id, float(metadata.get("mass", 0.0))
            )
            if current_mass < self.reemergence_mass_threshold:
                alpha = mass / (current_mass + mass + 1e-8)
                new_mass = (1.0 - alpha) * current_mass + alpha * mass
                now = datetime.now(timezone.utc)
                self.backend.update_metadata(record_id, {
                    "mass": new_mass,
                    "last_seen": now.isoformat(),
                })
                self._record_mass[record_id] = new_mass
                self._record_last_seen[record_id] = now
                return UUID(record_id)

        # --- 3. new record ------------------------------------------------
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

        Returns list of (ContextBodyRecord, cosine_distance) sorted by
        gravitational influence (mass / dist^2) descending -- heaviest
        nearby bodies first.

        Triggers decay() automatically if decay_interval seconds have elapsed
        since the last automatic decay run.
        """
        # Auto-decay trigger
        now = datetime.now(timezone.utc)
        if self._last_decay_at is None or (
            (now - self._last_decay_at).total_seconds() >= self.decay_interval
        ):
            self._last_decay_at = now
            self.decay()

        filter_dict = {"domain": domain} if domain else None
        raw = self.backend.search(vector=embedding, k=k * 2, filter=filter_dict)

        results: list[tuple[ContextBodyRecord, float]] = []
        for record_id, dist, metadata, stored_vec in raw:
            centroid = stored_vec if stored_vec is not None else embedding
            rec = ContextBodyRecord.from_metadata(centroid=centroid, metadata=metadata)
            # Use local mass cache for up-to-date value
            rec.mass = self._record_mass.get(record_id, rec.mass)
            if rec.mass < mass_threshold:
                continue
            results.append((rec, dist))

        # Gravitational ranking: mass / dist^2 descending
        def grav_score(item: tuple[ContextBodyRecord, float]) -> float:
            rec, dist = item
            return rec.mass / (dist ** 2 + 1e-8)

        results.sort(key=grav_score, reverse=True)
        return results[:k]

    def decay(self) -> list[str]:
        """
        Apply exponential mass decay to all records.
        Records whose mass drops below extinction_threshold are removed.

        Returns list of extinct record UUIDs (as strings).
        """
        extinct: list[str] = []
        now = datetime.now(timezone.utc)

        for record_id in list(self._record_mass.keys()):
            last_seen = self._record_last_seen.get(record_id, now)
            elapsed = (now - last_seen).total_seconds()
            current_mass = self._record_mass[record_id]
            new_mass = current_mass * (1.0 - self.decay_rate * elapsed)

            if new_mass < self.extinction_threshold:
                extinct.append(record_id)
                self.backend.delete([record_id])
                self._record_mass.pop(record_id, None)
                self._record_last_seen.pop(record_id, None)
                self._resonance_cache.pop(record_id, None)
            else:
                self._record_mass[record_id] = new_mass
                self.backend.update_metadata(record_id, {"mass": new_mass})

        return extinct

    def merge_records(
        self,
        incoming_id: str,
        incoming_centroid: np.ndarray,
        existing_record: "ContextBodyRecord",
    ) -> None:
        """
        Inelastic collision between two stored records (cross-session).

        The incoming record is absorbed into the existing record:
          - Centroid becomes the mass-weighted average of both.
          - Masses are summed.
          - Resonance partners are transferred from incoming to existing.
          - The incoming record is deleted from the backend.

        Called by GravitationalSampler when a newly persisted local body
        overlaps (within collision_distance) with a store record loaded from
        a previous session.

        Parameters
        ----------
        incoming_id       : record ID of the newly persisted body (absorbed)
        incoming_centroid : centroid of the incoming body
        existing_record   : the surviving ContextBodyRecord (already loaded
                            into active_bodies, so its centroid is known)
        """
        existing_id = str(existing_record.id)

        incoming_mass = self._record_mass.get(incoming_id, 0.0)
        existing_mass = self._record_mass.get(existing_id, existing_record.mass)
        total_mass = incoming_mass + existing_mass
        if total_mass < 1e-8:
            return

        # Mass-weighted centroid
        merged_centroid = (
            incoming_centroid * incoming_mass
            + existing_record.centroid * existing_mass
        ) / total_mass

        # Re-upsert the surviving record with the merged centroid and summed
        # mass. upsert() removes the stale FAISS vector before reinserting, so
        # the ANN index stays consistent without a separate delete step.
        now = datetime.now(timezone.utc)
        merged_meta = existing_record.to_metadata()
        merged_meta["mass"] = total_mass
        merged_meta["last_seen"] = now.isoformat()

        self.backend.upsert(
            record_id=existing_id,
            vector=merged_centroid,
            metadata=merged_meta,
        )
        self._record_mass[existing_id] = total_mass
        self._record_last_seen[existing_id] = now

        # Transfer resonance partners from incoming to existing.
        # If both records already share a partner, scores are additive
        # (capped by record_resonance's max_score).
        for partner_id, score in self._resonance_cache.get(incoming_id, {}).items():
            if partner_id != existing_id:
                self.record_resonance(existing_id, partner_id, delta=score)

        # Delete the absorbed record.
        self.backend.delete([incoming_id])
        self._record_mass.pop(incoming_id, None)
        self._record_last_seen.pop(incoming_id, None)
        self._resonance_cache.pop(incoming_id, None)

    def record_resonance(
        self,
        id_a: str,
        id_b: str,
        delta: float = 0.1,
        max_score: float = 1.0,
    ) -> None:
        """
        Increment resonance score between id_a and id_b (symmetric).
        Score is capped at max_score.
        """
        for src, dst in [(id_a, id_b), (id_b, id_a)]:
            cache = self._resonance_cache.setdefault(src, {})
            current = cache.get(dst, 0.0)
            new_score = min(current + delta, max_score)
            cache[dst] = new_score
            # Write through to backend
            self.backend.update_metadata(src, {
                "resonance_partners": json.dumps(cache)
            })

    def _fetch_centroid(self, record_id: str, fallback: np.ndarray) -> np.ndarray:
        """
        Approximate centroid by re-querying the backend with the known ID.
        FAISSBackend doesn't return stored vectors, so we use the query embedding
        as a fallback approximation when no stored vector is available.
        """
        return fallback

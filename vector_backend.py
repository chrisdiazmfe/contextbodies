from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable
from uuid import UUID

import numpy as np
import faiss


# ---------------------------------------------------------------------------
# Protocol — swap any backend without changing ContextBodyStore
# ---------------------------------------------------------------------------

@runtime_checkable
class VectorBackend(Protocol):
    """
    Minimal interface for a vector store backend.

    Implementations must support:
        upsert          — insert or update a record by ID
        search          — ANN search returning (id, cosine_distance, metadata, vector|None)
        delete          — remove records by ID
        update_metadata — patch scalar metadata on an existing record

    The optional fourth element of each search result is the stored centroid vector.
    Backends that return vectors (Qdrant, Pinecone) populate it; FAISSBackend returns
    None. ContextBodyStore uses it to eliminate the _fetch_centroid approximation when
    available.

    Intended backends: FAISSBackend (default), QdrantBackend, PineconeBackend, pgvector.
    Swap by passing a different backend to ContextBodyStore.__init__.
    """

    def upsert(
        self,
        record_id: str,
        vector: np.ndarray,
        metadata: dict,
    ) -> None: ...

    def search(
        self,
        vector: np.ndarray,
        k: int,
        filter: dict | None = None,
    ) -> list[tuple[str, float, dict, np.ndarray | None]]:
        """
        Returns up to k results as (record_id, cosine_distance, metadata, stored_vector)
        sorted by distance ascending (nearest first).
        stored_vector is None for backends that don't return vectors in search results.
        filter is a dict of metadata equality constraints, e.g. {"domain": "ml"}.
        """
        ...

    def delete(self, record_ids: list[str]) -> None: ...

    def update_metadata(self, record_id: str, metadata: dict) -> None: ...


# ---------------------------------------------------------------------------
# FAISSBackend — in-memory default, no external dependencies beyond faiss
# ---------------------------------------------------------------------------

class FAISSBackend:
    """
    In-memory vector backend backed by FAISS IndexIDMap.

    Uses IndexFlatIP (inner product) on L2-normalized vectors, which is
    equivalent to cosine similarity. IndexIDMap wraps it to support removal
    of individual records by ID without rebuilding the index.

    Metadata is stored in a parallel dict keyed by string record ID.

    Suitable for:
        - Development and testing
        - Single-session use where persistence isn't required
        - Small corpora (< ~1M bodies) where in-memory is acceptable

    For production cross-session persistence, replace with QdrantBackend,
    PineconeBackend, or PgvectorBackend — all share this same interface.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim

        # IndexIDMap lets us add and remove by int64 ID
        base = faiss.IndexFlatIP(embedding_dim)
        self.index = faiss.IndexIDMap(base)

        # metadata store: string_id → dict
        self.metadata: dict[str, dict] = {}

        # string_id ↔ int64_id mapping (FAISS requires int64)
        self._str_to_int: dict[str, int] = {}
        self._int_to_str: dict[int, str] = {}
        self._next_int_id: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-8)

    def _get_or_create_int_id(self, record_id: str) -> int:
        if record_id not in self._str_to_int:
            int_id = self._next_int_id
            self._next_int_id += 1
            self._str_to_int[record_id] = int_id
            self._int_to_str[int_id] = record_id
        return self._str_to_int[record_id]

    def _matches_filter(self, meta: dict, filter: dict | None) -> bool:
        if not filter:
            return True
        return all(meta.get(k) == v for k, v in filter.items())

    # ------------------------------------------------------------------
    # VectorBackend interface
    # ------------------------------------------------------------------

    def upsert(
        self,
        record_id: str,
        vector: np.ndarray,
        metadata: dict,
    ) -> None:
        """
        Insert or replace a record.
        If the record_id already exists, removes the old vector before inserting.
        """
        if record_id in self._str_to_int:
            # remove old vector so we don't have stale duplicates
            int_id = self._str_to_int[record_id]
            self.index.remove_ids(
                np.array([int_id], dtype=np.int64)
            )

        int_id = self._get_or_create_int_id(record_id)
        norm_vec = self._normalize(vector).reshape(1, -1).astype(np.float32)
        ids = np.array([int_id], dtype=np.int64)
        self.index.add_with_ids(norm_vec, ids)
        self.metadata[record_id] = dict(metadata)

    def search(
        self,
        vector: np.ndarray,
        k: int,
        filter: dict | None = None,
    ) -> list[tuple[str, float, dict, np.ndarray | None]]:
        """
        Return up to k nearest records as (record_id, cosine_distance, metadata, None).

        The fourth element is always None — FAISSBackend does not return stored vectors
        in search results. ContextBodyStore falls back to _fetch_centroid in this case.

        Oversamples by 2× then applies metadata filter, so the effective k
        may be lower than requested if many records are filtered out.
        """
        total = self.index.ntotal
        if total == 0:
            return []

        k_search = min(k * 2, total)
        norm_vec = self._normalize(vector).reshape(1, -1).astype(np.float32)
        similarities, int_ids = self.index.search(norm_vec, k_search)

        results = []
        for sim, int_id in zip(similarities[0], int_ids[0]):
            if int_id < 0:
                continue
            record_id = self._int_to_str.get(int(int_id))
            if record_id is None:
                continue
            meta = self.metadata.get(record_id, {})
            if not self._matches_filter(meta, filter):
                continue
            cosine_dist = float(1.0 - sim)
            results.append((record_id, cosine_dist, meta, None))
            if len(results) >= k:
                break

        return results  # already sorted nearest-first by FAISS

    def delete(self, record_ids: list[str]) -> None:
        """Remove records by string ID."""
        int_ids = [
            self._str_to_int[rid]
            for rid in record_ids
            if rid in self._str_to_int
        ]
        if int_ids:
            self.index.remove_ids(np.array(int_ids, dtype=np.int64))
        for rid in record_ids:
            self.metadata.pop(rid, None)
            int_id = self._str_to_int.pop(rid, None)
            if int_id is not None:
                self._int_to_str.pop(int_id, None)

    def update_metadata(self, record_id: str, metadata: dict) -> None:
        """Patch scalar metadata fields on an existing record."""
        if record_id in self.metadata:
            self.metadata[record_id].update(metadata)

    @property
    def size(self) -> int:
        return self.index.ntotal


# ---------------------------------------------------------------------------
# QdrantBackend — production backend backed by Qdrant
# ---------------------------------------------------------------------------

class QdrantBackend:
    """
    Vector backend backed by Qdrant.

    Accepts a pre-configured QdrantClient so any deployment variant works:
        Local server:  QdrantClient(host="localhost", port=6333)
        Qdrant Cloud:  QdrantClient(url="https://...", api_key="...")
        In-memory:     QdrantClient(":memory:")   ← useful for testing

    Unlike FAISSBackend, Qdrant returns stored vectors in search results, so
    ContextBodyStore can use the actual centroid instead of the query-vector
    approximation in _fetch_centroid. The fourth element of each search result
    is the stored np.ndarray rather than None.

    The collection is created automatically on first use if it does not exist.
    Cosine distance is used to match the rest of the system.

    Install the client:
        pip install qdrant-client
    """

    def __init__(
        self,
        client,                              # qdrant_client.QdrantClient
        collection_name: str = "context_bodies",
        embedding_dim: int = 768,
    ):
        self.client = client
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        from qdrant_client.models import Distance, VectorParams

        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    @staticmethod
    def _build_filter(filter: dict | None):
        """Convert a flat equality dict to a Qdrant Filter object."""
        if not filter:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        return Filter(
            must=[
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter.items()
            ]
        )

    # ------------------------------------------------------------------
    # VectorBackend interface
    # ------------------------------------------------------------------

    def upsert(
        self,
        record_id: str,
        vector: np.ndarray,
        metadata: dict,
    ) -> None:
        from qdrant_client.models import PointStruct

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=record_id,
                    vector=vector.astype(np.float32).tolist(),
                    payload=metadata,
                )
            ],
        )

    def search(
        self,
        vector: np.ndarray,
        k: int,
        filter: dict | None = None,
    ) -> list[tuple[str, float, dict, np.ndarray | None]]:
        """
        Return up to k nearest records as (record_id, cosine_distance, metadata, centroid).

        Qdrant returns cosine similarity in [-1, 1]; we convert to cosine distance
        via 1 - score so results are consistent with FAISSBackend.
        The stored centroid vector is returned directly, eliminating the
        _fetch_centroid approximation used by FAISSBackend.
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector.astype(np.float32).tolist(),
            query_filter=self._build_filter(filter),
            limit=k,
            with_payload=True,
            with_vectors=True,
        )

        return [
            (
                str(r.id),
                float(1.0 - r.score),
                r.payload or {},
                np.array(r.vector, dtype=np.float32) if r.vector is not None else None,
            )
            for r in results
        ]

    def delete(self, record_ids: list[str]) -> None:
        from qdrant_client.models import PointIdsList

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=record_ids),
        )

    def update_metadata(self, record_id: str, metadata: dict) -> None:
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=metadata,
            points=[record_id],
        )

    @property
    def size(self) -> int:
        return self.client.count(collection_name=self.collection_name).count

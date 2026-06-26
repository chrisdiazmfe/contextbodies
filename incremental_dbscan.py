from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

import numpy as np

from context_body import ContextBody


@dataclass
class IncrementalDBSCAN:
    """
    Incremental DBSCAN clustering over token embeddings.

    Standard DBSCAN isn't designed for online updates — this implementation
    maintains cluster state incrementally, processing one token at a time.

    Context bodies emerge when a dense enough region forms in embedding space.
    Sparse points are treated as noise (asteroids). Dense cores become bodies.

    Parameters:
        eps         — neighborhood radius in embedding space (cosine distance)
        min_samples — minimum tokens to form a core (body nucleus)
        dim         — embedding dimensionality
    """

    eps: float = 0.1
    min_samples: int = 5
    dim: int = 768

    # internal state
    embeddings: list[np.ndarray] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)       # -1 = noise
    clusters: dict[int, list[int]] = field(default_factory=dict)  # label → [indices]
    _next_label: int = 0

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _neighbors(self, idx: int) -> list[int]:
        """Return indices of all points within eps of point idx."""
        target = self.embeddings[idx]
        return [
            i for i, emb in enumerate(self.embeddings)
            if i != idx and self._cosine_distance(target, emb) <= self.eps
        ]

    def update(
        self, token_id: int, embedding: np.ndarray
    ) -> tuple[list[ContextBody], list[tuple[int, int]], list[int]]:
        """
        Add a new token and update cluster state.

        Returns:
            new_bodies  — newly stabilized ContextBody objects
            merged      — list of (old_label, new_label) merge events
            fragmented  — list of labels that split
        """
        self.embeddings.append(embedding)
        self.token_ids.append(token_id)
        idx = len(self.embeddings) - 1

        neighbors = self._neighbors(idx)

        # noise point — no dense neighborhood
        if len(neighbors) < self.min_samples:
            self.labels.append(-1)
            return [], [], []

        # find which existing clusters the neighbors belong to
        neighbor_labels = set(
            self.labels[n] for n in neighbors if self.labels[n] != -1
        )

        if not neighbor_labels:
            # new cluster nucleus
            label = self._next_label
            self._next_label += 1
            self.labels.append(label)
            self.clusters[label] = [idx] + neighbors
            for n in neighbors:
                if self.labels[n] == -1:
                    self.labels[n] = label
            new_bodies = self._build_bodies([label])
            return new_bodies, [], []

        elif len(neighbor_labels) == 1:
            # join existing cluster
            label = next(iter(neighbor_labels))
            self.labels.append(label)
            self.clusters.setdefault(label, []).append(idx)
            return [], [], []

        else:
            # merge multiple clusters
            labels_list = sorted(neighbor_labels)
            primary = labels_list[0]
            merged_events = []

            for other in labels_list[1:]:
                self.clusters[primary].extend(self.clusters.pop(other, []))
                for i, lbl in enumerate(self.labels):
                    if lbl == other:
                        self.labels[i] = primary
                merged_events.append((other, primary))

            self.labels.append(primary)
            self.clusters[primary].append(idx)
            return [], merged_events, []

    def _build_bodies(self, labels: list[int]) -> list[ContextBody]:
        """Build ContextBody objects for newly formed clusters."""
        bodies = []
        for label in labels:
            indices = self.clusters.get(label, [])
            if not indices:
                continue
            cluster_embeddings = np.array([self.embeddings[i] for i in indices])
            cluster_tokens = [self.token_ids[i] for i in indices]

            centroid = cluster_embeddings.mean(axis=0)
            density = len(indices) / (np.var(cluster_embeddings).sum() + 1e-8)

            body = ContextBody(
                centroid=centroid,
                centroid_velocity=np.zeros_like(centroid),
                covariance=np.cov(cluster_embeddings.T) if len(indices) > 1
                           else np.eye(cluster_embeddings.shape[1]),
                mass=density,   # initial mass = density; refined later with weight norms
                density=float(density),
                stability=0.5,
                member_tokens=set(cluster_tokens),
                orbital_radii={
                    tok: float(1 - np.dot(cluster_embeddings[i], centroid) /
                               (np.linalg.norm(cluster_embeddings[i]) *
                                np.linalg.norm(centroid) + 1e-8))
                    for i, tok in enumerate(cluster_tokens)
                },
            )
            bodies.append(body)
        return bodies

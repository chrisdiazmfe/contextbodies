from __future__ import annotations

from collections import defaultdict, deque
from typing import Literal

import numpy as np
import faiss

from context_body import ContextBody

PointType = Literal["core", "border", "noise"]


class IncrementalDBSCAN:
    """
    Incremental DBSCAN clustering over token embeddings.

    Processes one token at a time. Uses FAISS for O(log n) neighbor queries
    instead of O(n^2) brute force.

    DBSCAN point types:
        core   -- has >= min_samples points within eps (including itself)
        border -- within eps of a core point, but fewer than min_samples neighbors
        noise  -- neither core nor border (asteroids in context gravity terms)

    Incremental update rules when a new point p arrives:
        1. Query FAISS for all existing points within eps of p.
        2. If |N(p)| >= min_samples -> p is a core point:
               - No existing cluster neighbors -> create new cluster
               - One existing cluster       -> join it
               - Multiple existing clusters -> merge them all
               - Absorb any noise neighbors as border points
               - Recheck existing neighbors -- p may have tipped them to core
        3. If |N(p)| < min_samples -> p is border or noise:
               - Any core neighbor exists -> border point, join that cluster
               - No core neighbors        -> noise point

    Fragmentation -- two mechanisms:

        1. Connectivity fragmentation (structural):
           A cluster fragments when its internal eps-connectivity graph splits into
           multiple connected components -- meaning some core points can no longer
           reach others through eps-neighborhood chains within the cluster.

        2. Bimodality fragmentation (semantic):
           A cluster is bimodal when it contains two distinct density peaks along
           its principal axis of variance, separated by a low-density valley.

        Both checks are gated on stability < fragmentation_stability_threshold
        to avoid running them every token on healthy clusters.

    Parameters:
        eps                               -- neighborhood radius in cosine distance space
        min_samples                       -- minimum points (including self) to form a core point
        dim                               -- embedding dimensionality
        stability_window                  -- number of centroid snapshots used to compute stability
        faiss_k_cap                       -- max neighbors returned by FAISS per query.
        fragmentation_stability_threshold -- stability score below which fragmentation is checked.
        bimodality_min_cluster_size       -- minimum cluster size to attempt bimodality check.
        bimodality_elongation_threshold   -- ratio of PC1 variance to mean per-dimension variance.
        bimodality_valley_depth           -- valley height must be below this fraction of smaller peak.
        collision_detection_threshold     -- centroid-to-centroid cosine distance below which two
                                            clusters are reported as a collision_event in update().
                                            Set to 0.0 to disable. Default: 0.2.
    """

    def __init__(
        self,
        eps: float = 0.1,
        min_samples: int = 5,
        dim: int = 768,
        stability_window: int = 10,
        faiss_k_cap: int = 512,
        fragmentation_stability_threshold: float = 0.3,
        bimodality_min_cluster_size: int | None = None,
        bimodality_elongation_threshold: float = 2.0,
        bimodality_valley_depth: float = 0.5,
        collision_detection_threshold: float = 0.2,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.dim = dim
        self.stability_window = stability_window
        self.faiss_k_cap = faiss_k_cap
        self.fragmentation_stability_threshold = fragmentation_stability_threshold
        self.bimodality_min_cluster_size = bimodality_min_cluster_size or (2 * min_samples)
        self.bimodality_elongation_threshold = bimodality_elongation_threshold
        self.bimodality_valley_depth = bimodality_valley_depth
        self.collision_detection_threshold = collision_detection_threshold

        # Per-point state (parallel lists indexed by insertion order)
        self.embeddings: list[np.ndarray] = []
        self.token_ids: list[int] = []
        self.labels: list[int] = []          # -1 = noise
        self.point_types: list[PointType] = []

        # Per-cluster state
        self.clusters: dict[int, set[int]] = {}         # label -> set of point indices
        self.centroids: dict[int, np.ndarray] = {}      # label -> centroid vector
        self.centroid_history: dict[int, list[np.ndarray]] = defaultdict(list)
        self.stabilities: dict[int, float] = {}         # label -> [0,1]
        self._point_mass: dict[int, float] = {}  # point_index -> token_mass

        # Noise set (point indices)
        self._noise_indices: set[int] = set()

        # FAISS index (IndexFlatIP on normalized vectors = cosine similarity)
        base = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap(base)

        self._next_label: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        token_id: int,
        embedding: np.ndarray,
        token_mass: float = 1.0,
    ) -> tuple[list[ContextBody], list[tuple], list[tuple], list[tuple]]:
        """
        Process one new token embedding.

        Returns a 4-tuple:
            new_bodies       -- list[ContextBody] for newly stabilized clusters
            merged_events    -- list of (old_label_a, old_label_b) 2-tuples
            fragmented_events -- list of (old_label, [ContextBody, ...]) tuples
            collision_events -- list of (label_a, label_b, dist) 3-tuples
        """
        new_bodies: list[ContextBody] = []
        merged_events: list[tuple] = []
        fragmented_events: list[tuple] = []

        idx = len(self.embeddings)
        self.embeddings.append(embedding)
        self.token_ids.append(token_id)
        self.labels.append(-1)
        self.point_types.append("noise")

        # Record this token's mass by index immediately (before cluster assignment)
        self._record_point_mass(idx, token_mass)

        # Add to FAISS
        norm_emb = self._normalize(embedding).reshape(1, -1).astype(np.float32)
        self.index.add_with_ids(norm_emb, np.array([idx], dtype=np.int64))

        # Find neighbors within eps
        neighbors = self._range_query(embedding)
        # Include self
        all_neighbors = neighbors | {idx}

        if len(all_neighbors) >= self.min_samples:
            # p is a core point
            self.point_types[idx] = "core"

            # Find which clusters the neighbors belong to
            neighbor_clusters: set[int] = set()
            for n_idx in neighbors:
                lbl = self.labels[n_idx]
                if lbl != -1:
                    neighbor_clusters.add(lbl)

            if not neighbor_clusters:
                # Create new cluster
                new_label = self._new_label()
                self._assign_to_cluster(idx, new_label, "core")
                # Absorb noise neighbors as border
                for n_idx in neighbors:
                    if self.labels[n_idx] == -1:
                        self._assign_to_cluster(n_idx, new_label, "border")
                        self._noise_indices.discard(n_idx)
                self._update_centroid(new_label)
                # Emit new body when cluster first forms
                body = self._make_body(new_label)
                new_bodies.append(body)

            elif len(neighbor_clusters) == 1:
                # Join existing cluster
                existing_label = next(iter(neighbor_clusters))
                self._assign_to_cluster(idx, existing_label, "core")
                # Absorb noise neighbors
                for n_idx in neighbors:
                    if self.labels[n_idx] == -1:
                        self._assign_to_cluster(n_idx, existing_label, "border")
                        self._noise_indices.discard(n_idx)
                self._update_centroid(existing_label)

            else:
                # Merge multiple clusters
                sorted_labels = sorted(neighbor_clusters)
                target_label = sorted_labels[0]
                for other_label in sorted_labels[1:]:
                    merged_events.append((other_label, target_label))
                    self._merge_clusters(other_label, target_label)
                self._assign_to_cluster(idx, target_label, "core")
                for n_idx in neighbors:
                    if self.labels[n_idx] == -1:
                        self._assign_to_cluster(n_idx, target_label, "border")
                        self._noise_indices.discard(n_idx)
                self._update_centroid(target_label)

            # Recheck existing neighbors -- they may now have enough neighbors to be core
            assigned_label = self.labels[idx]
            for n_idx in neighbors:
                if self.point_types[n_idx] != "core":
                    n_neighbors = self._range_query(self.embeddings[n_idx])
                    if len(n_neighbors | {n_idx}) >= self.min_samples:
                        self.point_types[n_idx] = "core"

            # Fragmentation check on the cluster
            if assigned_label != -1:
                frag_labels, frag_bodies = self._maybe_fragment(assigned_label)
                if frag_labels:
                    fragmented_events.append(
                        (assigned_label if assigned_label in self.clusters else frag_labels[0],
                         frag_bodies)
                    )
                    # assigned_label may be gone now; emit new_bodies for fragments
                    for fb in frag_bodies:
                        new_bodies.append(fb)
        else:
            # p is border or noise
            core_neighbor_label = None
            for n_idx in neighbors:
                if self.point_types[n_idx] == "core" and self.labels[n_idx] != -1:
                    core_neighbor_label = self.labels[n_idx]
                    break

            if core_neighbor_label is not None:
                self.point_types[idx] = "border"
                self._assign_to_cluster(idx, core_neighbor_label, "border")
                self._update_centroid(core_neighbor_label)
            else:
                self.point_types[idx] = "noise"
                self.labels[idx] = -1
                self._noise_indices.add(idx)

        # Collision detection: check all cluster pairs for close centroids
        collision_events = self._detect_collisions()

        return new_bodies, merged_events, fragmented_events, collision_events

    def noise_points(self) -> list[int]:
        """Return token_ids of all noise points."""
        return [self.token_ids[i] for i in self._noise_indices]

    @property
    def num_clusters(self) -> int:
        return len(self.clusters)

    @property
    def num_noise(self) -> int:
        return len(self._noise_indices)

    # ------------------------------------------------------------------
    # Internal: cluster management
    # ------------------------------------------------------------------

    def _new_label(self) -> int:
        lbl = self._next_label
        self._next_label += 1
        return lbl

    def _assign_to_cluster(self, idx: int, label: int, point_type: PointType) -> None:
        old_label = self.labels[idx]
        if old_label != -1 and old_label != label:
            self.clusters[old_label].discard(idx)
        self.labels[idx] = label
        self.point_types[idx] = point_type
        if label not in self.clusters:
            self.clusters[label] = set()
            self.stabilities[label] = 0.0
        self.clusters[label].add(idx)

    def _merge_clusters(self, src_label: int, dst_label: int) -> None:
        """Merge src_label into dst_label."""
        if src_label not in self.clusters:
            return
        for idx in self.clusters[src_label]:
            self.labels[idx] = dst_label
            self.clusters[dst_label].add(idx)
        # _point_mass is per-index, nothing to merge per-label
        del self.clusters[src_label]
        self.stabilities.pop(src_label, None)
        self.centroids.pop(src_label, None)
        self.centroid_history.pop(src_label, None)

    def _update_centroid(self, label: int) -> None:
        if label not in self.clusters or not self.clusters[label]:
            return
        embs = np.stack([self.embeddings[i] for i in self.clusters[label]])
        centroid = embs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-8:
            centroid = centroid / norm
        self.centroids[label] = centroid

        history = self.centroid_history[label]
        history.append(centroid.copy())
        if len(history) > self.stability_window:
            history.pop(0)

        if len(history) >= 2:
            drifts = [
                float(1.0 - np.dot(history[i], history[i + 1]))
                for i in range(len(history) - 1)
            ]
            mean_drift = float(np.mean(drifts))
            self.stabilities[label] = float(np.exp(-10.0 * mean_drift))
        else:
            self.stabilities[label] = 0.0

    def _record_point_mass(self, idx: int, mass: float) -> None:
        self._point_mass[idx] = mass

    def _make_body(self, label: int) -> ContextBody:
        """Construct a ContextBody snapshot for a cluster."""
        body = ContextBody()
        indices = self.clusters.get(label, set())
        for idx in sorted(indices):
            m = self._point_mass.get(idx, 1.0)
            body.accrete(
                token_id=self.token_ids[idx],
                token_embedding=self.embeddings[idx],
                token_mass=m,
            )
        body.stability = self.stabilities.get(label, 0.0)
        return body

    # ------------------------------------------------------------------
    # Internal: neighbor queries
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / (n + 1e-8)

    def _range_query(self, embedding: np.ndarray, exclude_self: bool = True) -> set[int]:
        """Return indices of all points within eps cosine distance of embedding."""
        if self.index.ntotal == 0:
            return set()
        k = min(self.faiss_k_cap, self.index.ntotal)
        norm_emb = self._normalize(embedding).reshape(1, -1).astype(np.float32)
        sims, ids = self.index.search(norm_emb, k)
        result = set()
        for sim, int_id in zip(sims[0], ids[0]):
            if int_id < 0:
                continue
            dist = float(1.0 - sim)
            if dist <= self.eps:
                result.add(int(int_id))
        return result

    # ------------------------------------------------------------------
    # Internal: fragmentation
    # ------------------------------------------------------------------

    def _maybe_fragment(self, label: int) -> tuple[list[int], list[ContextBody]]:
        """
        Check if cluster `label` should fragment and trigger it if so.
        Returns (new_labels, new_bodies). Empty lists if no fragmentation.
        """
        if label not in self.clusters:
            return [], []
        if self.stabilities.get(label, 0.0) >= self.fragmentation_stability_threshold:
            return [], []

        # 1. Connectivity check
        components = self._connected_components(label)
        if len(components) > 1:
            return self._fragment_cluster(label, components)

        # 2. Bimodality check
        if self._check_bimodality(label):
            return self._split_bimodal(label)

        return [], []

    def _connected_components(self, label: int) -> list[set[int]]:
        """
        BFS over the eps-connectivity graph of core points in cluster `label`.
        Border points are attached to the component of their nearest core neighbor.
        Returns list of sets of point indices (one per component).
        """
        if label not in self.clusters:
            return [set()]

        members = self.clusters[label]
        core_indices = {i for i in members if self.point_types[i] == "core"}

        if len(core_indices) < 2:
            return [set(members)]

        visited: set[int] = set()
        components: list[set[int]] = []

        for start in core_indices:
            if start in visited:
                continue
            component: set[int] = set()
            queue = deque([start])
            while queue:
                curr = queue.popleft()
                if curr in visited:
                    continue
                visited.add(curr)
                if curr in members:
                    component.add(curr)
                # Only expand from core points
                if self.point_types[curr] == "core":
                    neighbors = self._range_query(self.embeddings[curr])
                    for n in neighbors:
                        if n in core_indices and n not in visited:
                            queue.append(n)
                        # Border points within eps of this core point join this component
                        if n in members and self.point_types[n] == "border" and n not in visited:
                            component.add(n)
                            visited.add(n)
            if component:
                components.append(component)

        # Any members not yet visited (shouldn't happen, but be safe)
        remainder = members - visited
        if remainder:
            if components:
                components[-1].update(remainder)
            else:
                components.append(remainder)

        return components

    def _fragment_cluster(
        self, label: int, components: list[set[int]]
    ) -> tuple[list[int], list[ContextBody]]:
        """
        Split cluster `label` into len(components) new clusters.
        Returns (new_labels, new_bodies).
        """
        old_history = list(self.centroid_history.get(label, []))
        old_indices_sorted = sorted(self.clusters.get(label, set()))
        mass_by_idx = {idx: self._point_mass.get(idx, 1.0) for idx in old_indices_sorted}

        # Remove old cluster
        del self.clusters[label]
        self.stabilities.pop(label, None)
        self.centroids.pop(label, None)
        self.centroid_history.pop(label, None)

        new_labels: list[int] = []
        new_bodies: list[ContextBody] = []

        for component in components:
            new_label = self._new_label()
            new_labels.append(new_label)
            self.clusters[new_label] = set(component)
            self.stabilities[new_label] = 0.0
            # _point_mass already has per-index masses; no per-label list needed
            for idx in component:
                self.labels[idx] = new_label
            # Inherit centroid history from parent
            self.centroid_history[new_label] = list(old_history)
            self._update_centroid(new_label)

            body = ContextBody()
            for i, idx in enumerate(sorted(component)):
                m = mass_by_idx.get(idx, 1.0)
                body.accrete(
                    token_id=self.token_ids[idx],
                    token_embedding=self.embeddings[idx],
                    token_mass=m,
                )
            body.parent_ids = [label]
            body.stability = self.stabilities.get(new_label, 0.0)
            new_bodies.append(body)

        return new_labels, new_bodies

    # ------------------------------------------------------------------
    # Internal: bimodality detection
    # ------------------------------------------------------------------

    def _check_bimodality(self, label: int) -> bool:
        """
        Return True if cluster `label` exhibits bimodal distribution.
        Three-stage gate: size, elongation, valley.
        """
        if label not in self.clusters:
            return False
        members = self.clusters[label]
        if len(members) < self.bimodality_min_cluster_size:
            return False

        embs = np.stack([self.embeddings[i] for i in members])
        X = embs - embs.mean(axis=0)

        pc1 = self._first_principal_component(X)

        # Elongation gate
        pc1_var = float(np.var(X @ pc1))
        per_dim_var = float(np.mean(np.var(X, axis=0)))
        if per_dim_var < 1e-10:
            return False
        elongation = pc1_var / per_dim_var
        if elongation < self.bimodality_elongation_threshold:
            return False

        # Valley gate
        projections = X @ pc1
        return self._has_valley(projections)

    def _first_principal_component(self, X: np.ndarray) -> np.ndarray:
        """Power iteration to find dominant eigenvector of X^T X."""
        rng = np.random.default_rng(0)
        v = rng.standard_normal(X.shape[1])
        v = v / (np.linalg.norm(v) + 1e-8)
        for _ in range(20):
            v = X.T @ (X @ v)
            norm = np.linalg.norm(v)
            if norm < 1e-10:
                break
            v = v / norm
        return v

    def _has_valley(self, projections: np.ndarray) -> bool:
        """
        Return True if the 1D distribution of projections has a significant valley
        between two peaks. Uses top-2 bins strategy: find the two highest-count bins,
        then check whether the region between them is genuinely depleted.
        """
        n = len(projections)
        if n < 4:
            return False

        n_bins = max(6, int(np.sqrt(n)))
        counts, _ = np.histogram(projections, bins=n_bins)

        if np.max(counts) == 0:
            return False

        # Find the two highest-count bins
        top2 = np.argsort(counts)[-2:]
        p1, p2 = int(np.min(top2)), int(np.max(top2))

        if p1 == p2 or p2 - p1 < 2:
            return False

        smaller_peak = float(min(counts[p1], counts[p2]))
        if smaller_peak < 1:
            return False

        # Valley = bins strictly between the two peaks
        valley = counts[p1 + 1:p2]
        valley_min = float(np.min(valley))
        valley_mean = float(np.mean(valley))
        threshold = self.bimodality_valley_depth * smaller_peak

        # Both the minimum and the mean of the valley must be below threshold
        # to avoid noise-level fluctuations triggering a false positive
        return bool(valley_min < threshold and valley_mean < threshold)

    def _split_bimodal(self, label: int) -> tuple[list[int], list[ContextBody]]:
        """
        Split cluster along PC1 at the largest projection gap.
        Abort if either half < min_samples.
        Returns (new_labels, new_bodies), or ([], []) if aborted.
        """
        if label not in self.clusters:
            return [], []

        members = sorted(self.clusters[label])
        embs = np.stack([self.embeddings[i] for i in members])
        X = embs - embs.mean(axis=0)
        pc1 = self._first_principal_component(X)
        projections = X @ pc1

        # Split at the largest gap in the sorted projection values
        sort_order = np.argsort(projections)
        sorted_proj = projections[sort_order]
        gaps = np.diff(sorted_proj)
        valley_idx = int(np.argmax(gaps))
        split_val = float((sorted_proj[valley_idx] + sorted_proj[valley_idx + 1]) / 2.0)

        side_a = {members[i] for i, p in enumerate(projections) if p <= split_val}
        side_b = {members[i] for i, p in enumerate(projections) if p > split_val}

        if len(side_a) < self.min_samples or len(side_b) < self.min_samples:
            return [], []

        return self._fragment_cluster(label, [side_a, side_b])

    # ------------------------------------------------------------------
    # Internal: collision detection
    # ------------------------------------------------------------------

    def _detect_collisions(self) -> list[tuple[int, int, float]]:
        """
        Check all cluster pairs for centroids within collision_detection_threshold.
        Returns list of (label_a, label_b, dist) 3-tuples.
        """
        if self.collision_detection_threshold <= 0.0:
            return []

        labels = list(self.clusters.keys())
        events = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                la, lb = labels[i], labels[j]
                ca = self.centroids.get(la)
                cb = self.centroids.get(lb)
                if ca is None or cb is None:
                    continue
                dist = float(1.0 - np.dot(self._normalize(ca), self._normalize(cb)))
                if dist < self.collision_detection_threshold:
                    events.append((la, lb, dist))
        return events

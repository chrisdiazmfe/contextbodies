from __future__ import annotations

from typing import Literal

import numpy as np
import faiss

from context_body import ContextBody

PointType = Literal["core", "border", "noise"]


class IncrementalDBSCAN:
    """
    Incremental DBSCAN clustering over token embeddings.

    Processes one token at a time. Uses FAISS for O(log n) neighbor queries
    instead of O(n²) brute force.

    DBSCAN point types:
        core   — has >= min_samples points within eps (including itself)
        border — within eps of a core point, but fewer than min_samples neighbors
        noise  — neither core nor border (asteroids in context gravity terms)

    Incremental update rules when a new point p arrives:
        1. Query FAISS for all existing points within eps of p.
        2. If |N(p)| >= min_samples → p is a core point:
               - No existing cluster neighbors → create new cluster
               - One existing cluster       → join it
               - Multiple existing clusters → merge them all
               - Absorb any noise neighbors as border points
               - Recheck existing neighbors — p may have tipped them to core
        3. If |N(p)| < min_samples → p is border or noise:
               - Any core neighbor exists → border point, join that cluster
               - No core neighbors        → noise point

    Fragmentation — two mechanisms:

        1. Connectivity fragmentation (structural):
           A cluster fragments when its internal eps-connectivity graph splits into
           multiple connected components — meaning some core points can no longer
           reach others through eps-neighborhood chains within the cluster.

        2. Bimodality fragmentation (semantic):
           A cluster is bimodal when it contains two distinct density peaks along
           its principal axis of variance, separated by a low-density valley.
           This catches the common case where two topics co-occur in embedding
           space (close enough to cluster together initially) but accumulate
           separately as the conversation develops — forming a dumbbell shape
           that remains eps-connected but is semantically incoherent.

           Detection: project cluster members onto their first principal component
           (via cheap power iteration), then look for a significant valley between
           two peaks in the 1D histogram. If found, split at the valley midpoint.

        Both checks are gated on stability < `fragmentation_stability_threshold`
        to avoid running them every token on healthy clusters.

    Parameters:
        eps                               — neighborhood radius in cosine distance space
        min_samples                       — minimum points (including self) to form a core point
        dim                               — embedding dimensionality
        stability_window                  — number of centroid snapshots used to compute stability
        faiss_k_cap                       — max neighbors returned by FAISS per query.
                                            512 is safe for typical LLM context lengths at eps=0.1
                                            in high-dimensional space; raise if needed.
        fragmentation_stability_threshold — stability score below which fragmentation is checked.
                                            Lower = less sensitive. Higher = more aggressive.
        bimodality_min_cluster_size       — minimum cluster size to attempt bimodality check.
                                            Default: 2 * min_samples.
        bimodality_elongation_threshold   — ratio of PC1 variance to mean per-dimension variance.
                                            Clusters below this ratio are roughly spherical and
                                            skipped. Default: 2.0.
        bimodality_valley_depth           — valley height must be below this fraction of the
                                            smaller peak height. Default: 0.5 (valley < 50% of
                                            smaller peak). Lower = less sensitive.
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
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.dim = dim
        self.stability_window = stability_window
        self.faiss_k_cap = faiss_k_cap
        self.fragmentation_stability_threshold = fragmentation_stability_threshold
        self.bimodality_min_cluster_size = (
            bimodality_min_cluster_size
            if bimodality_min_cluster_size is not None
            else min_samples * 2
        )
        self.bimodality_elongation_threshold = bimodality_elongation_threshold
        self.bimodality_valley_depth = bimodality_valley_depth

        # per-point state
        self.embeddings: list[np.ndarray] = []
        self.token_ids: list[int] = []
        self.token_masses: list[float] = []  # weight-norm-derived mass per token
        self.labels: list[int] = []          # -1 = noise / unassigned
        self.point_types: list[PointType] = []

        # per-cluster state
        self.clusters: dict[int, set[int]] = {}   # label → set of point indices
        self.centroids: dict[int, np.ndarray] = {}
        self.centroid_history: dict[int, list[np.ndarray]] = {}
        self.stabilities: dict[int, float] = {}
        self._next_label: int = 0

        # FAISS — inner product on L2-normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(dim)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-8)

    def _cosine_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - np.dot(self._normalize(a), self._normalize(b)))

    def _find_neighbors(self, embedding: np.ndarray, exclude_idx: int = -1) -> list[int]:
        """
        Return indices of all existing points within eps cosine distance.
        Uses FAISS; caps at faiss_k_cap results.
        """
        n = len(self.embeddings)
        if n == 0:
            return []

        k = min(n, self.faiss_k_cap)
        norm_emb = self._normalize(embedding).reshape(1, -1).astype(np.float32)
        similarities, indices = self.index.search(norm_emb, k)

        neighbors = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0 or idx == exclude_idx:
                continue
            if (1.0 - float(sim)) <= self.eps:   # convert similarity → distance
                neighbors.append(int(idx))
        return neighbors

    def _is_core(self, idx: int) -> bool:
        """Check if an existing point qualifies as a core point."""
        neighbors = self._find_neighbors(self.embeddings[idx], exclude_idx=idx)
        return (len(neighbors) + 1) >= self.min_samples  # +1 for self

    def _update_centroid(self, label: int) -> None:
        """
        Recompute centroid for a cluster and update stability.
        Stability = 1 / (1 + variance_of_recent_centroid_drift).
        """
        indices = list(self.clusters[label])
        embs = np.array([self.embeddings[i] for i in indices])
        new_centroid = embs.mean(axis=0)

        history = self.centroid_history.setdefault(label, [])
        old = self.centroids.get(label)
        if old is not None:
            history.append(old.copy())
            if len(history) > self.stability_window:
                history.pop(0)

        self.centroids[label] = new_centroid

        if len(history) >= 2:
            drifts = [
                np.linalg.norm(history[i + 1] - history[i])
                for i in range(len(history) - 1)
            ]
            variance = float(np.var(drifts))
            self.stabilities[label] = 1.0 / (1.0 + variance * 100.0)
        else:
            self.stabilities[label] = 0.5

    def _absorb_noise_neighbors(self, label: int, neighbors: list[int]) -> None:
        """Promote noise/unassigned neighbors into a cluster as border points."""
        for n in neighbors:
            if self.labels[n] == -1:
                self.labels[n] = label
                self.clusters[label].add(n)
                self.point_types[n] = "border"

    def _merge_clusters(
        self, labels: set[int]
    ) -> tuple[int, list[tuple[int, int]]]:
        """
        Merge all clusters in `labels` into the lowest-numbered one.
        Returns (primary_label, [(absorbed_label, primary_label), ...]).
        """
        primary, *others = sorted(labels)
        merge_events = []

        for other in others:
            for idx in self.clusters[other]:
                self.labels[idx] = primary
            self.clusters[primary].update(self.clusters.pop(other))
            self.centroid_history.pop(other, None)
            self.centroids.pop(other, None)
            self.stabilities.pop(other, None)
            merge_events.append((other, primary))

        self._update_centroid(primary)
        return primary, merge_events

    def _new_cluster(self, idx: int, neighbors: list[int]) -> int:
        """Create a new cluster seeded by point idx and its neighbors."""
        label = self._next_label
        self._next_label += 1
        self.labels[idx] = label
        self.clusters[label] = {idx}
        self._absorb_noise_neighbors(label, neighbors)
        self._update_centroid(label)
        return label

    # ------------------------------------------------------------------
    # Fragmentation
    # ------------------------------------------------------------------

    def _connected_components(self, label: int) -> list[set[int]]:
        """
        Find connected components in the eps-neighborhood graph restricted to
        the cluster's core points.

        Two core points are connected if they are within eps of each other.
        Border points are not part of the graph — they are assigned afterward
        to the nearest component centroid.

        Returns a list of point-index sets, one per component.
        If len == 1, the cluster is still fully connected (no fragmentation).
        """
        members = list(self.clusters[label])
        member_set = set(members)
        core_members = [i for i in members if self.point_types[i] == "core"]

        if len(core_members) < 2:
            # can't split with fewer than 2 core points
            return [member_set]

        # build adjacency list among core members using FAISS
        core_set = set(core_members)
        adjacency: dict[int, list[int]] = {i: [] for i in core_members}

        for idx in core_members:
            neighbors = self._find_neighbors(self.embeddings[idx], exclude_idx=idx)
            for n in neighbors:
                if n in core_set:
                    adjacency[idx].append(n)

        # BFS over core members to find connected components
        visited: set[int] = set()
        core_components: list[set[int]] = []

        for start in core_members:
            if start in visited:
                continue
            component: set[int] = set()
            queue = [start]
            while queue:
                node = queue.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                queue.extend(adjacency[node])
            core_components.append(component)

        if len(core_components) == 1:
            return [member_set]

        # assign border points to the nearest component centroid
        component_centroids = [
            np.array([self.embeddings[i] for i in comp]).mean(axis=0)
            for comp in core_components
        ]

        full_components: list[set[int]] = [set(c) for c in core_components]
        border_members = [i for i in members if self.point_types[i] != "core"]

        for border_idx in border_members:
            emb = self.embeddings[border_idx]
            distances = [self._cosine_dist(emb, c) for c in component_centroids]
            nearest = int(np.argmin(distances))
            full_components[nearest].add(border_idx)

        return full_components

    def _fragment_cluster(
        self,
        label: int,
        components: list[set[int]],
    ) -> tuple[list[int], list[ContextBody]]:
        """
        Split a cluster into sub-clusters, one per connected component.

        The original cluster is removed. New labels are assigned to each
        component. Centroid history is inherited from the parent so stability
        continues to be tracked without resetting.

        Returns (new_labels, new_bodies) — the ContextBody for each fragment,
        with parent_ids set to [label] for lineage tracking.
        """
        new_labels: list[int] = []
        new_bodies: list[ContextBody] = []
        parent_history = list(self.centroid_history.get(label, []))

        for component in components:
            new_label = self._next_label
            self._next_label += 1
            new_labels.append(new_label)

            self.clusters[new_label] = component
            for idx in component:
                self.labels[idx] = new_label

            # inherit centroid history so stability doesn't hard-reset
            self.centroid_history[new_label] = list(parent_history)
            self._update_centroid(new_label)

            body = self._build_body(new_label)
            body.parent_ids = [label]   # preserve lineage
            new_bodies.append(body)

        # remove the now-fragmented cluster
        del self.clusters[label]
        self.centroids.pop(label, None)
        self.centroid_history.pop(label, None)
        self.stabilities.pop(label, None)

        return new_labels, new_bodies

    def _maybe_fragment(
        self, label: int
    ) -> tuple[list[int], list[ContextBody]]:
        """
        Check whether a cluster should fragment and execute if so.

        Runs two checks in order, only when stability < threshold:
            1. Structural connectivity (BFS) — fast, catches hard splits
            2. Semantic bimodality (PCA + valley) — catches soft dumbbell splits

        Returns (new_labels, new_bodies) if fragmented, ([], []) if not.
        """
        if label not in self.clusters:
            return [], []

        stability = self.stabilities.get(label, 1.0)
        if stability > self.fragmentation_stability_threshold:
            return [], []

        # --- structural check -------------------------------------------
        components = self._connected_components(label)
        if len(components) > 1:
            return self._fragment_cluster(label, components)

        # --- bimodality check -------------------------------------------
        if self._check_bimodality(label):
            return self._split_bimodal(label)

        return [], []

    # ------------------------------------------------------------------
    # Bimodality detection
    # ------------------------------------------------------------------

    def _first_principal_component(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the first principal component via power iteration.

        O(n * D * iterations) — much cheaper than full eigendecomposition
        O(D³) for large embedding dimensions (768, 1536, etc.).

        Converges in ~20 iterations for typical embedding matrices.
        """
        rng = np.random.default_rng(seed=0)   # deterministic
        v = rng.standard_normal(X.shape[1])
        v /= np.linalg.norm(v) + 1e-8

        for _ in range(20):
            v = X.T @ (X @ v)
            norm = float(np.linalg.norm(v))
            if norm < 1e-8:
                break
            v /= norm

        return v

    def _has_valley(self, projections: np.ndarray) -> bool:
        """
        Detect a significant valley between two peaks in a 1D distribution.

        Uses an adaptive bin count and light smoothing to avoid false
        positives from histogram noise. The valley must be below
        `bimodality_valley_depth` * (height of the smaller peak).

        Returns True if a bimodal structure is present.
        """
        n_bins = max(10, len(projections) // 3)
        hist, _ = np.histogram(projections, bins=n_bins)

        # light 3-point smoothing to reduce histogram noise
        smoothed = np.convolve(
            hist.astype(float), [0.25, 0.5, 0.25], mode="same"
        )

        peaks = [
            i for i in range(1, len(smoothed) - 1)
            if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]
        ]
        valleys = [
            i for i in range(1, len(smoothed) - 1)
            if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]
        ]

        if len(peaks) < 2:
            return False

        # two tallest peaks
        p1, p2 = sorted(peaks, key=lambda p: smoothed[p], reverse=True)[:2]
        lo, hi = min(p1, p2), max(p1, p2)

        between = [v for v in valleys if lo < v < hi]
        if not between:
            return False

        deepest = min(between, key=lambda v: smoothed[v])
        valley_h = smoothed[deepest]
        min_peak_h = min(smoothed[p1], smoothed[p2])

        return bool(valley_h < min_peak_h * self.bimodality_valley_depth)

    def _check_bimodality(self, label: int) -> bool:
        """
        Check if a cluster is bimodal.

        Three-stage pipeline — each stage is a cheap gate before the next:
            1. Size gate        — skip clusters smaller than min size
            2. Elongation gate  — skip roughly spherical clusters
                                  (PC1 variance not dominant → unlikely bimodal)
            3. Valley detection — check for two peaks with a valley in between
                                  in the PC1 projection
        """
        indices = list(self.clusters[label])
        if len(indices) < self.bimodality_min_cluster_size:
            return False

        embs = np.array([self.embeddings[i] for i in indices])
        centered = embs - embs.mean(axis=0)

        # elongation gate
        pc1 = self._first_principal_component(centered)
        pc1_variance = float(np.var(centered @ pc1))
        total_variance = float(np.var(centered))

        if total_variance < 1e-8:
            return False  # degenerate / all-same cluster

        # compare PC1 variance to the average per-dimension variance
        mean_dim_variance = total_variance / centered.shape[1]
        elongation = pc1_variance / (mean_dim_variance + 1e-8)

        if elongation < self.bimodality_elongation_threshold:
            return False  # spherical — skip valley check

        return self._has_valley(centered @ pc1)

    def _split_bimodal(self, label: int) -> tuple[list[int], list[ContextBody]]:
        """
        Split a bimodal cluster at the valley of its PC1 projection.

        Reprojects the cluster onto PC1, finds the deepest valley between the
        two tallest peaks, and partitions points on each side of the valley
        midpoint into separate clusters via `_fragment_cluster`.

        If either partition is smaller than `min_samples`, the split is aborted
        and ([], []) is returned — better to leave an impure cluster than create
        a cluster too small to be meaningful.
        """
        indices = list(self.clusters[label])
        embs = np.array([self.embeddings[i] for i in indices])
        centered = embs - embs.mean(axis=0)

        pc1 = self._first_principal_component(centered)
        projections = centered @ pc1

        n_bins = max(10, len(projections) // 3)
        hist, bin_edges = np.histogram(projections, bins=n_bins)
        smoothed = np.convolve(
            hist.astype(float), [0.25, 0.5, 0.25], mode="same"
        )

        peaks = [
            i for i in range(1, len(smoothed) - 1)
            if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]
        ]
        valleys = [
            i for i in range(1, len(smoothed) - 1)
            if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]
        ]

        p1, p2 = sorted(peaks, key=lambda p: smoothed[p], reverse=True)[:2]
        lo, hi = min(p1, p2), max(p1, p2)
        between = [v for v in valleys if lo < v < hi]
        deepest = min(between, key=lambda v: smoothed[v])

        split_value = float(
            (bin_edges[deepest] + bin_edges[deepest + 1]) / 2
        )

        group_a = {indices[i] for i, p in enumerate(projections) if p <= split_value}
        group_b = {indices[i] for i, p in enumerate(projections) if p > split_value}

        # abort if either side is too small to be a valid cluster
        if len(group_a) < self.min_samples or len(group_b) < self.min_samples:
            return [], []

        return self._fragment_cluster(label, [group_a, group_b])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        token_id: int,
        embedding: np.ndarray,
        token_mass: float = 1.0,
    ) -> tuple[list[ContextBody], list[tuple[int, int]], list[tuple[int, list[ContextBody]]]]:
        """
        Add a new token and update cluster state.

        token_mass — physical mass of this token, derived from model weight norms
                     via ||W[token_id]|| / G. Defaults to 1.0 when the weight
                     matrix is unavailable (e.g. prompt seeding in initialize()).
                     Body mass is the sum of its constituent token masses, so
                     bodies built from high-weight tokens are proportionally heavier.

        Returns:
            new_bodies        — ContextBody objects for any newly formed clusters
            merged_events     — [(absorbed_label, surviving_label), ...]
            fragmented_events — [(old_label, [new_body, ...]), ...]
                                One entry per cluster that split this step.
        """
        idx = len(self.embeddings)
        self.embeddings.append(embedding)
        self.token_ids.append(token_id)
        self.token_masses.append(token_mass)
        self.labels.append(-1)            # placeholder
        self.point_types.append("noise")  # placeholder

        # add to FAISS index
        norm_emb = self._normalize(embedding).reshape(1, -1).astype(np.float32)
        self.index.add(norm_emb)

        neighbors = self._find_neighbors(embedding, exclude_idx=idx)
        neighbor_count = len(neighbors) + 1   # +1 for self

        new_bodies: list[ContextBody] = []
        merged_events: list[tuple[int, int]] = []
        fragmented_events: list[tuple[int, list[ContextBody]]] = []

        # track which cluster labels were touched this step so we can check
        # fragmentation once per affected cluster at the end
        touched_labels: set[int] = set()

        if neighbor_count >= self.min_samples:
            # ---- new point is a core point --------------------------------
            self.point_types[idx] = "core"

            neighbor_labels = {
                self.labels[n] for n in neighbors if self.labels[n] != -1
            }

            if not neighbor_labels:
                label = self._new_cluster(idx, neighbors)
                new_bodies.append(self._build_body(label))
                touched_labels.add(label)

            elif len(neighbor_labels) == 1:
                label = next(iter(neighbor_labels))
                self.labels[idx] = label
                self.clusters[label].add(idx)
                self._absorb_noise_neighbors(label, neighbors)
                self._update_centroid(label)
                touched_labels.add(label)

            else:
                # new point bridges multiple clusters — merge them
                self.labels[idx] = -2   # temp sentinel during merge
                label, merged_events = self._merge_clusters(neighbor_labels)
                self.labels[idx] = label
                self.clusters[label].add(idx)
                self._absorb_noise_neighbors(label, neighbors)
                self._update_centroid(label)
                touched_labels.add(label)

            # arriving core point may tip existing neighbors to core status
            for n in neighbors:
                if self.point_types[n] != "core" and self._is_core(n):
                    self.point_types[n] = "core"
                    if self.labels[n] != -1:
                        touched_labels.add(self.labels[n])

        else:
            # ---- new point is border or noise -----------------------------
            core_labels = {
                self.labels[n]
                for n in neighbors
                if self.labels[n] != -1 and self.point_types[n] == "core"
            }

            if core_labels:
                label = next(iter(core_labels))
                self.labels[idx] = label
                self.clusters[label].add(idx)
                self.point_types[idx] = "border"
                self._update_centroid(label)
                touched_labels.add(label)
            # else: remains noise (-1)

        # check fragmentation for every cluster touched this step
        for touched_label in list(touched_labels):
            frag_labels, frag_bodies = self._maybe_fragment(touched_label)
            if frag_labels:
                fragmented_events.append((touched_label, frag_bodies))

        return new_bodies, merged_events, fragmented_events

    def _build_body(self, label: int) -> ContextBody:
        """
        Construct a ContextBody snapshot from a cluster's current state.

        mass    — sum of constituent token masses (||W[token_id]|| / G).
                  Falls back to 1.0 per token for points seeded without a weight
                  matrix (e.g. prompt tokens in initialize()).
        density — geometric density: token count / total embedding variance.
                  Kept separate from mass so both signals remain available.
        """
        indices = list(self.clusters[label])
        embs = np.array([self.embeddings[i] for i in indices])
        tokens = [self.token_ids[i] for i in indices]
        centroid = self.centroids[label]
        n = len(indices)

        density = float(n / (np.var(embs).sum() + 1e-8))

        # body mass = sum of constituent token masses
        # falls back to 1.0 for tokens added before token_masses was populated
        # (e.g. via seed_cluster in tests or legacy callers)
        mass = sum(
            self.token_masses[i] if i < len(self.token_masses) else 1.0
            for i in indices
        )

        return ContextBody(
            centroid=centroid.copy(),
            centroid_velocity=np.zeros_like(centroid),
            covariance=(
                np.cov(embs.T) if n > 1 else np.eye(embs.shape[1])
            ),
            mass=mass,
            density=density,
            stability=float(self.stabilities.get(label, 0.5)),
            member_tokens=set(tokens),
            orbital_radii={
                tok: self._cosine_dist(embs[i], centroid)
                for i, tok in enumerate(tokens)
            },
        )

    def get_body(self, label: int) -> ContextBody | None:
        """Return the current ContextBody for a cluster label, or None."""
        if label not in self.clusters:
            return None
        return self._build_body(label)

    def all_bodies(self) -> list[ContextBody]:
        """Return ContextBody snapshots for all active clusters."""
        return [self._build_body(label) for label in self.clusters]

    def noise_points(self) -> list[tuple[int, np.ndarray]]:
        """
        Return all current noise points as (token_id, embedding) pairs.
        These are the asteroids — isolated tokens with no cluster affinity.
        """
        return [
            (self.token_ids[i], self.embeddings[i])
            for i, lbl in enumerate(self.labels)
            if lbl == -1
        ]

    @property
    def num_clusters(self) -> int:
        return len(self.clusters)

    @property
    def num_noise(self) -> int:
        return sum(1 for lbl in self.labels if lbl == -1)

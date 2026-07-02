from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class Universe:
    """
    The pre-computed gravitational universe for a language model.

    Attributes
    ----------
    centroids : [n_bodies, D] -- cluster centroids, L2-normalized
    masses    : [n_bodies]   -- gravitational mass of each body
    labels    : [vocab_size] -- cluster assignment for each token
    n_bodies  : number of universe bodies
    """

    def __init__(
        self,
        centroids: np.ndarray,
        masses: np.ndarray,
        labels: np.ndarray,
    ):
        self.centroids = centroids
        self.masses = masses
        self.labels = labels
        self.n_bodies = len(centroids)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        self._centroids_norm = centroids / (norms + 1e-8)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        np.savez(path, centroids=self.centroids, masses=self.masses, labels=self.labels)

    @classmethod
    def load(cls, path) -> "Universe":
        data = np.load(path)
        return cls(centroids=data["centroids"], masses=data["masses"], labels=data["labels"])

    # ------------------------------------------------------------------
    # Force computation
    # ------------------------------------------------------------------

    def compute_field(self, embs_norm, G, token_mass=1.0, context_pos=None,
                      top_k_bodies: int = 16):
        """Numpy/CPU path. Prefer compute_field_torch when GPU is available."""
        cos_sims = embs_norm @ self._centroids_norm.T   # [vocab, n_bodies]
        r = np.maximum(1.0 - cos_sims, 0.1)
        if context_pos is not None:
            ctx_norm = context_pos / (np.linalg.norm(context_pos) + 1e-8)
            affinities = np.maximum(self._centroids_norm @ ctx_norm, 0.0)
            k = min(top_k_bodies, self.n_bodies)
            if k < self.n_bodies:
                topk_idx = np.argpartition(affinities, -k)[-k:]
                mask = np.zeros(self.n_bodies, dtype=np.float32)
                mask[topk_idx] = 1.0
                affinities = affinities * mask
        else:
            affinities = np.ones(self.n_bodies, dtype=np.float32)
        effective_masses = self.masses * affinities
        forces = G * token_mass * effective_masses / (r ** 2)
        k_active = min(top_k_bodies, self.n_bodies)
        return forces.sum(axis=1) / k_active

    def compute_field_torch(self, embs_norm, G, token_mass=1.0, context_pos=None,
                            top_k_bodies: int = 16):
        """
        GPU-accelerated field computation. Returns CPU numpy array [vocab_size].

        top_k_bodies : int
            Only the top-k bodies most aligned with the current context contribute
            force. Concentrating force on relevant bodies makes the semantic
            geometry meaningful — random/shuffled centroids will align with
            different (less relevant) bodies and diverge from the real universe.
            Set to n_bodies to use all bodies (old behaviour).
        """
        device = embs_norm.device
        dtype = embs_norm.dtype
        if (
            not hasattr(self, "_torch_centroids")
            or self._torch_device != str(device)
            or self._torch_dtype != str(dtype)
        ):
            self._torch_centroids = torch.tensor(self._centroids_norm, dtype=dtype, device=device)
            self._torch_masses = torch.tensor(self.masses, dtype=dtype, device=device)
            self._torch_device = str(device)
            self._torch_dtype = str(dtype)
        with torch.no_grad():
            cos_sims = embs_norm @ self._torch_centroids.T   # [vocab, n_bodies]
            r = torch.clamp(1.0 - cos_sims, min=0.1)

            if context_pos is not None:
                ctx_norm = context_pos / (np.linalg.norm(context_pos) + 1e-8)
                ctx_t = torch.tensor(ctx_norm, dtype=dtype, device=device)
                affinities = torch.clamp(self._torch_centroids @ ctx_t, min=0.0)  # [n_bodies]

                # Concentrate force on the top-k most contextually relevant bodies.
                # This is the key change: with all bodies active, the force field is
                # nearly uniform (shuffled ≈ real). With top-k, only semantically
                # aligned bodies contribute — so geometry matters.
                k = min(top_k_bodies, self.n_bodies)
                if k < self.n_bodies:
                    topk_vals, topk_idx = torch.topk(affinities, k=k)
                    mask = torch.zeros(self.n_bodies, dtype=dtype, device=device)
                    mask[topk_idx] = 1.0
                    affinities = affinities * mask
            else:
                affinities = torch.ones(self.n_bodies, dtype=dtype, device=device)

            effective_masses = self._torch_masses * affinities
            forces = G * token_mass * effective_masses / (r ** 2)
            # Normalise by k (active bodies) not n_bodies so force scale is
            # consistent regardless of how many bodies are selected.
            k_active = min(top_k_bodies, self.n_bodies)
            field = forces.sum(dim=1) / k_active
        return field.cpu().numpy()

    # ------------------------------------------------------------------
    # Ablation variants
    # ------------------------------------------------------------------

    def shuffled(self, seed=None) -> "Universe":
        """
        Return a copy with centroid positions randomly permuted.

        Destroys semantic geometry while preserving mass distribution and
        token->cluster labels. Primary geometry ablation:
            real universe  ->  semantic geometry + mass + IDF
            shuffled       ->  random geometry   + mass + IDF
        """
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.n_bodies)
        return Universe(
            centroids=self.centroids[perm].copy(),
            masses=self.masses.copy(),
            labels=self.labels.copy(),
        )

    def with_random_centroids(self, seed=None) -> "Universe":
        """
        Return a copy with fully random centroids on the unit sphere.

        Destroys both semantic geometry and cluster structure:
            real universe    ->  semantic geometry + mass + IDF
            random_centroids ->  no geometry       + mass + IDF
        """
        rng = np.random.default_rng(seed)
        D = self.centroids.shape[1]
        random_c = rng.standard_normal((self.n_bodies, D)).astype(np.float32)
        norms = np.linalg.norm(random_c, axis=1, keepdims=True)
        random_c = random_c / (norms + 1e-8)
        return Universe(
            centroids=random_c,
            masses=self.masses.copy(),
            labels=self.labels.copy(),
        )

    def with_uniform_mass(self) -> "Universe":
        """Return a copy with all body masses = 1.0 (removes IDF/size mass effect)."""
        return Universe(
            centroids=self.centroids.copy(),
            masses=np.ones(self.n_bodies, dtype=np.float32),
            labels=self.labels.copy(),
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def nearest_tokens(self, tokenizer, body_idx, top_k=10):
        """Return top_k member tokens of a universe body (for interpretability check)."""
        member_ids = [int(tid) for tid, lbl in enumerate(self.labels) if lbl == body_idx]
        results = []
        for tid in member_ids[:top_k * 4]:
            try:
                results.append((tokenizer.decode([tid]), float(tid)))
            except Exception:
                pass
        return results[:top_k]

    def top_bodies_for_context(self, context_pos, top_k=10):
        """
        Return top_k universe bodies most relevant to the current context.
        Returns list of (body_idx, affinity, mass) tuples sorted by affinity desc.
        """
        ctx_norm = context_pos / (np.linalg.norm(context_pos) + 1e-8)
        affinities = self._centroids_norm @ ctx_norm
        top_indices = np.argsort(affinities)[-top_k:][::-1]
        return [(int(i), float(affinities[i]), float(self.masses[i])) for i in top_indices]

    def semantic_coverage(self, token_ids, threshold=0.1):
        """
        Fraction of universe bodies visited by a generated sequence.
        A body is visited if at least one generated token belongs to its cluster.
        """
        if not token_ids or self.n_bodies == 0:
            return 0.0
        visited = set()
        for tid in token_ids:
            if 0 <= tid < len(self.labels):
                visited.add(int(self.labels[tid]))
        return len(visited) / self.n_bodies


class UniverseBuilder:
    """
    Builds a Universe by clustering a language model's vocabulary embeddings.

    Algorithm
    ---------
    1. Extract the model's input embedding matrix [vocab_size, D].
    2. L2-normalize each embedding (so k-means minimizes cosine distance).
    3. Run MiniBatchKMeans to find n_clusters semantic cluster centroids.
    4. Assign a gravitational mass to each cluster.
    5. Return a Universe object.

    Mass schemes
    ------------
    "uniform" : all bodies have equal mass (1.0). Simple baseline.
    "size"    : mass proportional to cluster size (tokens/cluster).
    "idf"     : mass = mean IDF weight of member tokens.
    """

    def build(
        self,
        model: torch.nn.Module,
        n_clusters: int = 256,
        mass_scheme: str = "idf",
        random_state: int = 42,
        batch_size: int = 2048,
        n_init: int = 5,
        verbose: bool = True,
    ) -> Universe:
        """Cluster the model's vocabulary embeddings and return a Universe."""
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            raise ImportError(
                "scikit-learn is required to build a universe. "
                "Install it with: pip install scikit-learn --break-system-packages"
            )

        if verbose:
            print(f"Building universe: {n_clusters} clusters, mass_scheme='{mass_scheme}'")

        token_embeddings = model.get_input_embeddings().weight
        embs = token_embeddings.detach().cpu().numpy()
        vocab_size, D = embs.shape

        if verbose:
            print(f"  Vocabulary: {vocab_size} tokens, embedding dim: {D}")

        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs_norm = embs / (norms + 1e-8)

        if verbose:
            print(f"  Running MiniBatchKMeans (n_clusters={n_clusters})...")

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=batch_size,
            n_init=n_init,
            verbose=0,
        )
        labels = kmeans.fit_predict(embs_norm)
        centroids = kmeans.cluster_centers_

        if verbose:
            sizes = np.bincount(labels, minlength=n_clusters)
            print(f"  Cluster sizes: min={sizes.min()}, max={sizes.max()}, "
                  f"mean={sizes.mean():.1f}")

        masses = self._compute_masses(
            model=model, labels=labels, n_clusters=n_clusters,
            scheme=mass_scheme, verbose=verbose,
        )

        if verbose:
            print(f"  Body masses: min={masses.min():.3f}, max={masses.max():.3f}, "
                  f"mean={masses.mean():.3f}")
            print("Universe built.")

        return Universe(centroids=centroids, masses=masses, labels=labels)

    # ------------------------------------------------------------------
    # Internal: mass computation
    # ------------------------------------------------------------------

    def _compute_masses(self, model, labels, n_clusters, scheme, verbose):
        if scheme == "uniform":
            return np.ones(n_clusters, dtype=np.float32)

        elif scheme == "size":
            counts = np.bincount(labels, minlength=n_clusters).astype(np.float32)
            return counts / (counts.max() + 1e-8)

        elif scheme == "idf":
            if verbose:
                print("  Computing IDF weights from unconditional distribution...")
            device = next(model.parameters()).device
            bos_id = getattr(model.config, "bos_token_id", None) or 0
            input_ids = torch.tensor([[bos_id]], device=device)
            with torch.no_grad():
                outputs = model(input_ids)
                probs = torch.softmax(outputs.logits[0, -1, :], dim=-1).cpu().numpy()
            idf = -np.log(probs + 1e-8)
            idf_min, idf_max = idf.min(), idf.max()
            idf_norm = (idf - idf_min) / (idf_max - idf_min + 1e-8)
            masses = np.zeros(n_clusters, dtype=np.float32)
            counts = np.zeros(n_clusters, dtype=np.float32)
            for token_id, label in enumerate(labels):
                if token_id < len(idf_norm):
                    masses[label] += idf_norm[token_id]
                    counts[label] += 1.0
            return masses / (counts + 1e-8)

        else:
            raise ValueError(
                f"Unknown mass_scheme '{scheme}'. "
                "Choose from: 'uniform', 'size', 'idf'."
            )

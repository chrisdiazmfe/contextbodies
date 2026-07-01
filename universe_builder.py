from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class Universe:
    """
    The pre-computed gravitational universe for a language model.

    A Universe is a set of semantic bodies derived from clustering the model's
    full vocabulary embedding space. Unlike context bodies (which form dynamically
    from generated tokens), universe bodies are permanent, covering every semantic
    region expressible by the model.

    This provides the background gravitational field that all generation occurs
    within. Any idea that can be expressed using the model's vocabulary has a
    location in its universe — new concepts are new trajectories through the
    universe, not new locations.

    Attributes
    ----------
    centroids : [n_bodies, D] — cluster centroids, L2-normalized
    masses    : [n_bodies]   — gravitational mass of each body
    labels    : [vocab_size] — cluster assignment for each token
    n_bodies  : number of universe bodies
    """

    def __init__(
        self,
        centroids: np.ndarray,   # [n_bodies, D]
        masses: np.ndarray,      # [n_bodies]
        labels: np.ndarray,      # [vocab_size] token → cluster assignment
    ):
        self.centroids = centroids
        self.masses = masses
        self.labels = labels
        self.n_bodies = len(centroids)

        # Pre-compute L2-normalized centroids for efficient cosine similarity.
        # These never change, so computing once and caching saves work every step.
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        self._centroids_norm = centroids / (norms + 1e-8)   # [n_bodies, D]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save universe to a .npz file."""
        np.savez(
            path,
            centroids=self.centroids,
            masses=self.masses,
            labels=self.labels,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Universe":
        """Load universe from a .npz file."""
        data = np.load(path)
        return cls(
            centroids=data["centroids"],
            masses=data["masses"],
            labels=data["labels"],
        )

    # ------------------------------------------------------------------
    # Force computation
    # ------------------------------------------------------------------

    def compute_field(
        self,
        embs_norm: np.ndarray,   # [vocab_size, D] row-normalized token embeddings
        G: float,
        token_mass: float = 1.0,
    ) -> np.ndarray:
        """
        Compute the total gravitational field strength from all universe bodies
        at every token position in the vocabulary.

        Uses a single batched matrix multiply rather than one loop per body:

            cos_sims  = embs_norm @ centroids_norm.T    [vocab, n_bodies]
            r         = 1 - cos_sims                    [vocab, n_bodies]
            forces    = G * m_token * masses / r²       [vocab, n_bodies]
            field     = forces.sum(axis=1)              [vocab]

        Parameters
        ----------
        embs_norm   : L2-normalized vocabulary embedding matrix [vocab_size, D]
        G           : gravitational constant
        token_mass  : mass of the candidate token (default 1.0)

        Returns
        -------
        field : [vocab_size] total gravitational field strength per token
        """
        # [vocab_size, n_bodies] cosine similarities
        cos_sims = embs_norm @ self._centroids_norm.T

        # cosine distance, clipped to avoid 1/0 at identical vectors
        r = np.maximum(1.0 - cos_sims, 1e-6)          # [vocab_size, n_bodies]

        # F = G * m_token * m_body / r²  for each (token, body) pair
        forces = G * token_mass * self.masses / (r ** 2)  # [vocab_size, n_bodies]

        return forces.sum(axis=1)                       # [vocab_size]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def semantic_coverage(self, token_ids: list[int], threshold: float = 0.1) -> float:
        """
        Fraction of universe bodies that were meaningfully visited by a
        generated sequence.

        A body is "visited" if at least one generated token is assigned to it
        (via cluster label) or is within threshold cosine distance of its centroid.

        Parameters
        ----------
        token_ids : list of generated token ids
        threshold : cosine distance within which a token "visits" a body

        Returns
        -------
        coverage : float in [0, 1]
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
    "size"    : mass proportional to cluster size (tokens/cluster). Denser
                semantic regions exert more force — gravity follows population.
    "idf"     : mass = mean IDF weight of member tokens. Rarer, semantically
                specific clusters get higher mass; common-word clusters get lower
                mass. Matches the IDF philosophy used in the sampler's output field.
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
        """
        Cluster the model's vocabulary embeddings and return a Universe.

        Parameters
        ----------
        model        : HuggingFace causal LM
        n_clusters   : number of universe bodies (default 256)
        mass_scheme  : "uniform", "size", or "idf"
        random_state : k-means seed for reproducibility
        batch_size   : MiniBatchKMeans batch size
        n_init       : number of k-means random restarts
        verbose      : print progress

        Returns
        -------
        Universe
        """
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            raise ImportError(
                "scikit-learn is required to build a universe. "
                "Install it with: pip install scikit-learn --break-system-packages"
            )

        if verbose:
            print(f"Building universe: {n_clusters} clusters, mass_scheme='{mass_scheme}'")

        # --- Extract and normalize vocabulary embeddings ---
        token_embeddings = model.get_input_embeddings().weight
        embs = token_embeddings.detach().cpu().numpy()          # [vocab_size, D]
        vocab_size, D = embs.shape

        if verbose:
            print(f"  Vocabulary: {vocab_size} tokens, embedding dim: {D}")

        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs_norm = embs / (norms + 1e-8)                       # L2-normalized

        # --- Cluster ---
        if verbose:
            print(f"  Running MiniBatchKMeans (n_clusters={n_clusters})...")

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=batch_size,
            n_init=n_init,
            verbose=0,
        )
        labels = kmeans.fit_predict(embs_norm)                  # [vocab_size]
        centroids = kmeans.cluster_centers_                     # [n_clusters, D]

        if verbose:
            sizes = np.bincount(labels, minlength=n_clusters)
            print(f"  Cluster sizes: min={sizes.min()}, max={sizes.max()}, "
                  f"mean={sizes.mean():.1f}")

        # --- Compute masses ---
        masses = self._compute_masses(
            model=model,
            labels=labels,
            n_clusters=n_clusters,
            scheme=mass_scheme,
            verbose=verbose,
        )

        if verbose:
            print(f"  Body masses: min={masses.min():.3f}, max={masses.max():.3f}, "
                  f"mean={masses.mean():.3f}")
            print("Universe built.")

        return Universe(
            centroids=centroids,
            masses=masses,
            labels=labels,
        )

    # ------------------------------------------------------------------
    # Internal: mass computation
    # ------------------------------------------------------------------

    def _compute_masses(
        self,
        model: torch.nn.Module,
        labels: np.ndarray,
        n_clusters: int,
        scheme: str,
        verbose: bool,
    ) -> np.ndarray:
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
                probs = torch.softmax(
                    outputs.logits[0, -1, :], dim=-1
                ).cpu().numpy()                                  # [vocab_size]

            idf = -np.log(probs + 1e-8)
            idf_min, idf_max = idf.min(), idf.max()
            idf_norm = (idf - idf_min) / (idf_max - idf_min + 1e-8)  # [0, 1]

            # Mean IDF weight per cluster
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

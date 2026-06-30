from __future__ import annotations

import numpy as np


class AdaptiveDBSCAN:
    """
    Adaptive DBSCAN parameters that auto-calibrate to the embedding space
    and maintain a target number of active context bodies during generation.

    Two mechanisms:

    --- Geometry-based eps initialization ---

    At initialize() time, pairwise cosine distances among the prompt's context
    embeddings are computed and eps is set to the eps_percentile-th percentile
    of that distribution. This makes cluster radius proportional to the actual
    spread of the model's embedding space rather than a fixed hand-tuned value,
    so it transfers across model families and embedding dimensions automatically.

    --- Target body count feedback ---

    At each post_step(), the number of active bodies is compared to target_bodies.
    If too few bodies are forming, eps is nudged up (looser clustering → more
    clusters). If too many, eps is nudged down (tighter clustering → fewer).
    The adjustment is capped to eps_min/eps_max to stay in a sensible range.

    Parameters
    ----------
    target_bodies    : desired number of simultaneously active context bodies.
    eps_percentile   : percentile of pairwise context distances used to seed eps.
                       Lower = tighter initial clusters. 15-25 works well.
    eps_min          : hard floor on eps. Prevents clusters from becoming trivially
                       small (single-token bodies).
    eps_max          : hard ceiling on eps. Prevents the entire vocabulary from
                       collapsing into one cluster.
    min_samples      : minimum tokens to form a cluster core. Fixed; not adapted.
    adjustment_rate  : how much to move eps per step when body count is off-target.
                       Smaller = smoother but slower to react.
    """

    def __init__(
        self,
        target_bodies: int = 3,
        eps_percentile: float = 20.0,
        eps_min: float = 0.05,
        eps_max: float = 0.8,
        min_samples: int = 3,
        adjustment_rate: float = 0.01,
    ):
        self.target_bodies = target_bodies
        self.eps_percentile = eps_percentile
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.min_samples = min_samples
        self.adjustment_rate = adjustment_rate

        self.current_eps: float = 0.3   # overwritten by initialize_eps()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize_eps(self, context_embeddings: np.ndarray) -> float:
        """
        Derive eps from the pairwise cosine distance distribution of
        the prompt's context embeddings.

        Parameters
        ----------
        context_embeddings : [context_len, D] float array (CPU numpy)

        Returns
        -------
        eps : the initial DBSCAN epsilon to use for this generation
        """
        n = len(context_embeddings)
        if n < 2:
            # Can't compute pairwise distances with one token; keep default.
            return self.current_eps

        norms = np.linalg.norm(context_embeddings, axis=1, keepdims=True)
        embs_norm = context_embeddings / (norms + 1e-8)
        cos_sim = embs_norm @ embs_norm.T          # [n, n]
        cos_dist = 1.0 - cos_sim                   # cosine distance
        upper_tri = cos_dist[np.triu_indices(n, k=1)]

        eps = float(np.percentile(upper_tri, self.eps_percentile))
        self.current_eps = float(np.clip(eps, self.eps_min, self.eps_max))
        return self.current_eps

    def update(self, active_body_count: int) -> float:
        """
        Adjust eps toward the target number of active bodies.

        Call once per post_step() after active_bodies has been rebuilt.
        The returned eps should be written to clustering.eps so future
        DBSCAN updates use the new radius.

        Parameters
        ----------
        active_body_count : len(sampler.active_bodies) after this step

        Returns
        -------
        eps : updated epsilon
        """
        if active_body_count < self.target_bodies:
            # Too few bodies → loosen clustering radius
            self.current_eps = min(
                self.current_eps + self.adjustment_rate, self.eps_max
            )
        elif active_body_count > self.target_bodies:
            # Too many bodies → tighten clustering radius
            self.current_eps = max(
                self.current_eps - self.adjustment_rate, self.eps_min
            )
        # If exactly on target, leave eps unchanged.
        return self.current_eps

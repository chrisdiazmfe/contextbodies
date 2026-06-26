from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from context_body import ContextBody
from context_body_store import ContextBodyStore
from incremental_dbscan import IncrementalDBSCAN
from orbital_state import OrbitalState


class GravitationalSampler:
    """
    Replaces temperature-based sampling with a gravitational field model.

    At each token generation step:
        1. Compute gravitational force on every candidate token from all active
           context bodies (recorded + emergent).
        2. Add the force magnitudes as a bias to the raw logits.
        3. Sample from the adjusted distribution via torch.multinomial.
        4. Update orbital state (position, velocity, acceleration).
        5. Update incremental DBSCAN clustering with the new token.
        6. Record any newly stabilized emergent bodies to the store.

    Key parameters:
        G                  — gravitational constant; primary tuning knob.
                             Larger G = stronger context pull, less diversity.
        escape_threshold   — minimum force magnitude to influence sampling.
                             Tokens below this threshold are unaffected
                             (they have "escape velocity" from all bodies).
        stability_threshold — minimum stability score for a body to be persisted.
    """

    def __init__(
        self,
        body_store: ContextBodyStore,
        G: float = 1.0,
        escape_threshold: float = 0.01,
        stability_threshold: float = 0.8,
        domain: str = "",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.body_store = body_store
        self.G = G
        self.escape_threshold = escape_threshold
        self.stability_threshold = stability_threshold
        self.domain = domain
        self.device = device

        self.orbital_state: OrbitalState | None = None
        # (cluster_label, body, distance) — label=-1 for store-loaded bodies
        self.active_bodies: list[tuple[int, ContextBody, float]] = []
        self.clustering: IncrementalDBSCAN | None = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(
        self,
        context_embeddings: torch.Tensor,   # [context_len, D]
        embedding_dim: int,
    ) -> None:
        """
        Seed the sampler from the prompt context.
        Loads relevant recorded bodies and initializes orbital state + clustering.
        """
        initial_pos = context_embeddings[0].cpu().numpy()

        self.orbital_state = OrbitalState.initialize(initial_pos)

        self.clustering = IncrementalDBSCAN(
            eps=0.1,
            min_samples=5,
            dim=embedding_dim,
        )

        # seed clustering with all prompt tokens
        for i in range(context_embeddings.shape[0]):
            emb = context_embeddings[i].cpu().numpy()
            self.clustering.update(token_id=-i, embedding=emb)

        # load relevant recorded bodies from the store (label=-1 = store-sourced)
        self.active_bodies = [
            (-1, body, dist)
            for body, dist in self.body_store.query_nearby(
                embedding=initial_pos,
                domain=self.domain,
                k=20,
            )
        ]

    # ------------------------------------------------------------------
    # Force computation
    # ------------------------------------------------------------------

    def _compute_token_mass(
        self, token_id: int, weight_matrix: torch.Tensor
    ) -> float:
        """
        Derive token mass from model weight norms.
            m = W / G
        where W is the L2 norm of the token's row in the output embedding matrix.
        """
        weight_norm = torch.norm(weight_matrix[token_id]).item()
        return weight_norm / self.G

    def _gravitational_force(
        self,
        token_embedding: np.ndarray,
        token_mass: float,
        body: ContextBody,
    ) -> np.ndarray:
        """
        Compute gravitational force vector on a token from a single context body.

            F = G * (m_token * m_body) / r²  ·  r̂

        Distance r is cosine distance (angle-based), appropriate for embedding
        spaces where semantics live in direction rather than magnitude.
        Force vector points toward the body centroid.
        """
        r_vec = token_embedding - body.centroid

        # cosine distance as scalar r
        r = float(1 - np.dot(token_embedding, body.centroid) / (
            np.linalg.norm(token_embedding) * np.linalg.norm(body.centroid) + 1e-8
        ))
        r = max(r, 1e-8)  # avoid singularity at r = 0

        f_magnitude = self.G * (token_mass * body.mass) / (r ** 2)

        # unit vector pointing toward body centroid
        r_norm = np.linalg.norm(r_vec)
        r_hat = -r_vec / (r_norm + 1e-8)

        return f_magnitude * r_hat

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        logits: torch.Tensor,           # [vocab_size]
        token_embeddings: torch.Tensor,  # [vocab_size, D]
        weight_matrix: torch.Tensor,     # [vocab_size, D]
    ) -> int:
        """
        Apply gravitational field to logits and sample next token.

        Gravity bias is added to logits (not multiplied to probabilities) to
        preserve the relative shape of the base distribution while steering it.
        This avoids distribution collapse that could occur with multiplicative bias.

        Returns the sampled token id.
        """
        vocab_size = logits.shape[0]
        gravity_bias = np.zeros(vocab_size, dtype=np.float32)

        for token_id in range(vocab_size):
            token_emb = token_embeddings[token_id].cpu().numpy()
            token_mass = self._compute_token_mass(token_id, weight_matrix)

            total_force = np.zeros_like(token_emb)
            for _, body, _ in self.active_bodies:
                total_force += self._gravitational_force(token_emb, token_mass, body)

            force_magnitude = float(np.linalg.norm(total_force))

            # only apply bias if token exceeds escape threshold
            if force_magnitude > self.escape_threshold:
                gravity_bias[token_id] = force_magnitude

        # add gravity bias to logits and sample
        bias_tensor = torch.tensor(gravity_bias, dtype=torch.float32).to(self.device)
        adjusted_logits = logits + bias_tensor
        adjusted_probs = F.softmax(adjusted_logits, dim=-1)
        next_token = int(torch.multinomial(adjusted_probs, num_samples=1).item())

        # update orbital state with sampled token's embedding
        next_emb = token_embeddings[next_token].cpu().numpy()
        self.orbital_state.update(next_emb)

        # update clustering and record any stable emergent bodies
        self._update_clustering(next_token, next_emb)

        return next_token

    # ------------------------------------------------------------------
    # Clustering maintenance
    # ------------------------------------------------------------------

    def _update_clustering(self, token_id: int, embedding: np.ndarray) -> None:
        """
        Incrementally update DBSCAN clustering with the newly sampled token.
        Keeps active_bodies in sync by tracking cluster labels alongside bodies.

        Three events to handle:
            new_bodies        — append with their cluster label
            merged_events     — remove absorbed labels; surviving label stays
            fragmented_events — remove old label; append new fragment labels
        """
        new_bodies, merged_events, fragmented_events = self.clustering.update(
            token_id, embedding
        )

        # new cluster formed — append
        for body in new_bodies:
            body.domain = self.domain
            label = next(
                (lbl for lbl in self.clustering.clusters
                 if body.member_tokens <= self.clustering.clusters[lbl]),
                -1,
            )
            if body.stability >= self.stability_threshold:
                self.body_store.record(body)
            self.active_bodies.append((label, body, 0.0))

        # merge — remove absorbed labels (surviving label's body centroid
        # was updated in-place so it remains correct in active_bodies)
        if merged_events:
            absorbed = {old for old, _ in merged_events}
            self.active_bodies = [
                (lbl, b, d) for lbl, b, d in self.active_bodies
                if lbl not in absorbed
            ]

        # fragmentation — remove old label, add fragment bodies
        for old_label, frag_bodies in fragmented_events:
            self.active_bodies = [
                (lbl, b, d) for lbl, b, d in self.active_bodies
                if lbl != old_label
            ]
            for frag_body in frag_bodies:
                frag_body.domain = self.domain
                frag_label = next(
                    (lbl for lbl in self.clustering.clusters
                     if frag_body.member_tokens <= self.clustering.clusters[lbl]),
                    -1,
                )
                if frag_body.stability >= self.stability_threshold:
                    self.body_store.record(frag_body)
                self.active_bodies.append((frag_label, frag_body, 0.0))

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from context_body import ContextBody
from context_body_record import ContextBodyRecord
from context_body_store import ContextBodyStore
from domain_classifier import DomainClassifier
from incremental_dbscan import IncrementalDBSCAN
from orbital_state import OrbitalState

# Both ContextBody and ContextBodyRecord have .centroid and .mass,
# so force computation works on either without a union type.
_GravitySource = ContextBody | ContextBodyRecord


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
        resonance_threshold — minimum resonance score for a Lagrange midpoint force.
        domain             — static domain label used when no domain_classifier
                             is provided. Ignored if domain_classifier is set.
        domain_classifier  — optional DomainClassifier that infers the domain
                             from the token stream. When provided, self.domain
                             is updated automatically on every generated token.
    """

    def __init__(
        self,
        body_store: ContextBodyStore,
        G: float = 1.0,
        escape_threshold: float = 0.01,
        stability_threshold: float = 0.8,
        resonance_threshold: float = 0.3,
        domain: str = "",
        domain_classifier: DomainClassifier | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.body_store = body_store
        self.G = G
        self.escape_threshold = escape_threshold
        self.stability_threshold = stability_threshold
        self.resonance_threshold = resonance_threshold
        self.domain = domain
        self.domain_classifier = domain_classifier
        self.device = device

        self.orbital_state: OrbitalState | None = None
        # (cluster_label, body, distance)
        # label=-1 for store-loaded ContextBodyRecord objects
        # label>=0 for emergent ContextBody objects from IncrementalDBSCAN
        self.active_bodies: list[tuple[int, _GravitySource, float]] = []
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

        If a domain_classifier was provided, seeds it from the prompt embeddings
        so the initial domain is inferred rather than hard-coded.
        """
        embs_np = context_embeddings.cpu().numpy()   # [context_len, D]
        initial_pos = embs_np[0]

        # infer domain from the full prompt if a classifier is available
        if self.domain_classifier is not None:
            self.domain = self.domain_classifier.seed(embs_np)

        self.orbital_state = OrbitalState.initialize(initial_pos)

        self.clustering = IncrementalDBSCAN(
            eps=0.1,
            min_samples=5,
            dim=embedding_dim,
        )

        # seed clustering with all prompt tokens
        for i in range(context_embeddings.shape[0]):
            self.clustering.update(token_id=-i, embedding=embs_np[i])

        # load relevant recorded bodies from the store using the (now inferred) domain
        # query_nearby returns ContextBodyRecord objects — label=-1 marks them
        # as store-sourced so _update_clustering doesn't try to remove them
        self.active_bodies = [
            (-1, record, dist)
            for record, dist in self.body_store.query_nearby(
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

    def _compute_resonance_forces(
        self,
        token_embedding: np.ndarray,
        token_mass: float,
    ) -> np.ndarray:
        """
        Compute Lagrange midpoint forces for co-active resonant body pairs.

        When two ContextBodyRecord bodies in active_bodies have a mutual resonance
        score above resonance_threshold, they define a Lagrange midpoint:

            midpoint = (centroid_A + centroid_B) / 2

        The force toward this midpoint uses the geometric mean of the two body
        masses as the effective joint mass:

            F = G * m_token * sqrt(m_A * m_B) * score / r²  ·  r̂_midpoint

        This steers sampling toward tokens that bridge both resonant topics —
        the semantic region between them — with strength proportional to how
        consistently they've co-occurred across sessions.

        Only ContextBodyRecord pairs are checked; ephemeral ContextBody objects
        have no cross-session resonance history.
        """
        force = np.zeros_like(token_embedding)
        checked_pairs: set[frozenset] = set()

        for i, (_, body_a, _) in enumerate(self.active_bodies):
            if not isinstance(body_a, ContextBodyRecord):
                continue
            if not body_a.resonance_partners:
                continue

            for j, (_, body_b, _) in enumerate(self.active_bodies):
                if i >= j:
                    continue
                pair = frozenset({str(body_a.id), str(getattr(body_b, "id", None))})
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                partner_id = str(getattr(body_b, "id", None))
                if not partner_id or partner_id not in body_a.resonance_partners:
                    continue

                score = body_a.resonance_partners[partner_id]
                if score < self.resonance_threshold:
                    continue

                # Lagrange midpoint between the two resonant body centroids
                midpoint = (body_a.centroid + body_b.centroid) / 2.0
                mid_norm = np.linalg.norm(midpoint)
                if mid_norm < 1e-8:
                    continue

                # cosine distance from token to midpoint
                r = float(1.0 - np.dot(token_embedding, midpoint) / (
                    np.linalg.norm(token_embedding) * mid_norm + 1e-8
                ))
                r = max(r, 1e-8)

                # joint mass = geometric mean; preserves units and scales naturally
                joint_mass = float(np.sqrt(body_a.mass * body_b.mass))

                f_magnitude = self.G * token_mass * joint_mass * score / (r ** 2)

                direction = midpoint - token_embedding
                d_norm = np.linalg.norm(direction)
                if d_norm > 1e-8:
                    force += f_magnitude * (direction / d_norm)

        return force

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
            total_force += self._compute_resonance_forces(token_emb, token_mass)

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

        # update domain inference from the new token embedding
        if self.domain_classifier is not None:
            self.domain = self.domain_classifier.update(next_emb)

        # compute token mass from weight norms, then update clustering
        next_token_mass = self._compute_token_mass(next_token, weight_matrix)
        self._update_clustering(next_token, next_emb, next_token_mass)

        return next_token

    # ------------------------------------------------------------------
    # Clustering maintenance
    # ------------------------------------------------------------------

    def _update_clustering(
        self, token_id: int, embedding: np.ndarray, token_mass: float = 1.0
    ) -> None:
        """
        Incrementally update DBSCAN clustering with the newly sampled token.
        Keeps active_bodies in sync by tracking cluster labels alongside bodies.

        token_mass — derived from ||W[token_id]|| / G; propagated into the
                     cluster so body mass accumulates correctly from weight norms.

        Three events to handle:
            new_bodies        — append with their cluster label
            merged_events     — remove absorbed labels; surviving label stays
            fragmented_events — remove old label; append new fragment labels
        """
        new_bodies, merged_events, fragmented_events = self.clustering.update(
            token_id, embedding, token_mass=token_mass
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
                new_id = self.body_store.record(
                    centroid=body.centroid,
                    mass=body.mass,
                    stability=body.stability,
                    domain=body.domain,
                )
                # record resonance with all already-persisted co-active bodies
                for _, other_body, _ in self.active_bodies:
                    if isinstance(other_body, ContextBodyRecord):
                        self.body_store.record_resonance(
                            str(new_id), str(other_body.id)
                        )
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
                    frag_id = self.body_store.record(
                        centroid=frag_body.centroid,
                        mass=frag_body.mass,
                        stability=frag_body.stability,
                        domain=frag_body.domain,
                    )
                    # record resonance with all already-persisted co-active bodies
                    for _, other_body, _ in self.active_bodies:
                        if isinstance(other_body, ContextBodyRecord):
                            self.body_store.record_resonance(
                                str(frag_id), str(other_body.id)
                            )
                self.active_bodies.append((frag_label, frag_body, 0.0))

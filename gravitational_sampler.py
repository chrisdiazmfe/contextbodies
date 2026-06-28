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
        amplification_threshold: float = 0.2,
        collision_distance: float = 0.1,
        domain: str = "",
        domain_classifier: DomainClassifier | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.body_store = body_store
        self.G = G
        self.escape_threshold = escape_threshold
        self.stability_threshold = stability_threshold
        self.resonance_threshold = resonance_threshold
        self.amplification_threshold = amplification_threshold
        self.collision_distance = collision_distance
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
        body_centroid: np.ndarray,
        body_mass: float,
    ) -> np.ndarray:
        """
        Compute gravitational force vector on a token from a body (or virtual body).

            F = G * (m_token * m_body) / r²  ·  r̂

        Accepts centroid and mass directly so the same formula works for both
        individual active bodies and mass-weighted virtual groups produced by
        _group_active_bodies().

        Distance r is cosine distance (angle-based), appropriate for embedding
        spaces where semantics live in direction rather than magnitude.
        Force vector points toward the body centroid.
        """
        r_vec = token_embedding - body_centroid

        r = float(1.0 - np.dot(token_embedding, body_centroid) / (
            np.linalg.norm(token_embedding) * np.linalg.norm(body_centroid) + 1e-8
        ))
        r = max(r, 1e-8)

        f_magnitude = self.G * (token_mass * body_mass) / (r ** 2)

        r_norm = np.linalg.norm(r_vec)
        r_hat = -r_vec / (r_norm + 1e-8)

        return f_magnitude * r_hat

    def _group_active_bodies(self) -> list[tuple[np.ndarray, float]]:
        """
        Group active bodies within amplification_threshold of each other into
        virtual bodies, each with a mass-weighted centroid and summed mass.

        This prevents gravitational amplification — the unintended gravity well
        that forms when multiple bodies cluster near the same concept and their
        forces sum independently. A single body of mass Σmᵢ at the mass-weighted
        centroid is the physically correct representation.

        Uses greedy single-pass grouping: each body joins the first existing group
        whose centroid is within amplification_threshold (cosine distance), or
        starts a new group if none qualifies. O(n²) in active body count, which
        is typically small (<50).

        Returns (virtual_centroid, total_mass) pairs — one per group.
        """
        groups: list[list] = []  # [[centroid, mass], ...]

        for _, body, _ in self.active_bodies:
            centroid = body.centroid
            mass = body.mass
            placed = False

            for group in groups:
                g_centroid, g_mass = group
                r = float(1.0 - np.dot(centroid, g_centroid) / (
                    np.linalg.norm(centroid) * np.linalg.norm(g_centroid) + 1e-8
                ))
                if r < self.amplification_threshold:
                    # mass-weighted centroid update
                    total = g_mass + mass
                    group[0] = (g_centroid * g_mass + centroid * mass) / total
                    group[1] = total
                    placed = True
                    break

            if not placed:
                groups.append([centroid.copy(), mass])

        return [(np.array(g[0]), g[1]) for g in groups]

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

        # group nearby bodies into virtual bodies once per step — O(n²) in
        # active body count but n is small; avoids recomputing for every token
        virtual_bodies = self._group_active_bodies()

        for token_id in range(vocab_size):
            token_emb = token_embeddings[token_id].cpu().numpy()
            token_mass = self._compute_token_mass(token_id, weight_matrix)

            total_force = np.zeros_like(token_emb)
            for v_centroid, v_mass in virtual_bodies:
                total_force += self._gravitational_force(
                    token_emb, token_mass, v_centroid, v_mass
                )
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
    # Collision detection
    # ------------------------------------------------------------------

    def _check_collisions(self) -> None:
        """
        Detect and resolve inelastic collisions between active in-session bodies.

        Scans all pairs of ephemeral ContextBody objects in active_bodies. When
        two bodies are within collision_distance (cosine), the lighter is absorbed
        by the heavier using conservation of momentum:

            centroid_merged  = (m_A * c_A + m_B * c_B) / (m_A + m_B)
            velocity_merged  = (m_A * v_A + m_B * v_B) / (m_A + m_B)
            mass_merged      = m_A + m_B

        This handles the case where two cluster centroids converge semantically
        without a bridging token — a gap DBSCAN merge cannot close on its own.
        The merged body replaces both originals in active_bodies with label=-1
        (treated as a non-DBSCAN body for label-tracking purposes).

        If the merged body is stable enough, it is persisted to the store.
        DBSCAN's internal cluster state is left intact — the original clusters
        continue to receive new tokens, but force computation uses the merged body.
        """
        # collect only in-session ContextBody entries (label >= 0)
        body_entries: list[tuple[int, int, ContextBody]] = [
            (active_idx, label, body)
            for active_idx, (label, body, _) in enumerate(self.active_bodies)
            if isinstance(body, ContextBody) and label >= 0
        ]

        absorbed: set[int] = set()   # active_bodies indices to remove
        merged_bodies: list[tuple[int, ContextBody, float]] = []

        for i in range(len(body_entries)):
            for j in range(i + 1, len(body_entries)):
                active_idx_a, _, body_a = body_entries[i]
                active_idx_b, _, body_b = body_entries[j]

                if active_idx_a in absorbed or active_idx_b in absorbed:
                    continue

                r = float(1.0 - np.dot(body_a.centroid, body_b.centroid) / (
                    np.linalg.norm(body_a.centroid) * np.linalg.norm(body_b.centroid) + 1e-8
                ))

                if r >= self.collision_distance:
                    continue

                # inelastic collision — lighter absorbed by heavier
                heavy, light = (
                    (body_a, body_b) if body_a.mass >= body_b.mass
                    else (body_b, body_a)
                )
                heavy_active_idx = (
                    active_idx_a if body_a.mass >= body_b.mass else active_idx_b
                )
                light_active_idx = (
                    active_idx_b if body_a.mass >= body_b.mass else active_idx_a
                )

                total_mass = heavy.mass + light.mass

                merged = ContextBody(
                    domain=heavy.domain,
                    centroid=(
                        heavy.centroid * heavy.mass + light.centroid * light.mass
                    ) / total_mass,
                    centroid_velocity=(
                        heavy.centroid_velocity * heavy.mass
                        + light.centroid_velocity * light.mass
                    ) / total_mass,
                    mass=total_mass,
                    density=heavy.density + light.density,
                    # stability is the minimum of the two — a fresh collision
                    # is less stable than either parent alone
                    stability=min(heavy.stability, light.stability),
                    member_tokens=heavy.member_tokens | light.member_tokens,
                    parent_ids=[heavy.id, light.id],
                )

                if merged.stability >= self.stability_threshold:
                    merged_id = self.body_store.record(
                        centroid=merged.centroid,
                        mass=merged.mass,
                        stability=merged.stability,
                        domain=merged.domain,
                    )
                    for _, other_body, _ in self.active_bodies:
                        if isinstance(other_body, ContextBodyRecord):
                            self.body_store.record_resonance(
                                str(merged_id), str(other_body.id)
                            )

                absorbed.add(heavy_active_idx)
                absorbed.add(light_active_idx)
                # label=-1: merged body is not tracked by DBSCAN label logic
                merged_bodies.append((-1, merged, 0.0))

        if absorbed:
            self.active_bodies = [
                entry for active_idx, entry in enumerate(self.active_bodies)
                if active_idx not in absorbed
            ]
            self.active_bodies.extend(merged_bodies)

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

        # check for inelastic collisions among in-session bodies after all
        # DBSCAN events have been applied this step
        self._check_collisions()

from __future__ import annotations

from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_g import AdaptiveG
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
        G                  -- gravitational constant; primary tuning knob.
        escape_threshold   -- minimum force magnitude to influence sampling.
        stability_threshold -- minimum stability score for a body to be persisted.
        resonance_threshold -- minimum resonance score for a Lagrange midpoint force.
        recency_decay_lambda -- time-decay rate lambda for persisted bodies.
        adaptive_g         -- optional AdaptiveG instance.
        domain             -- static domain label used when no domain_classifier.
        domain_classifier  -- optional DomainClassifier.
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
        recency_decay_lambda: float = 0.0,
        adaptive_g: AdaptiveG | None = None,
        domain: str = "",
        domain_classifier: DomainClassifier | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dbscan_eps: float = 0.3,
        dbscan_min_samples: int = 3,
    ):
        self.body_store = body_store
        self.G = G
        self.escape_threshold = escape_threshold
        self.stability_threshold = stability_threshold
        self.resonance_threshold = resonance_threshold
        self.amplification_threshold = amplification_threshold
        self.collision_distance = collision_distance
        self.recency_decay_lambda = recency_decay_lambda
        self.adaptive_g = adaptive_g
        self.domain = domain
        self.domain_classifier = domain_classifier
        self.device = device

        self.orbital_state: OrbitalState | None = None
        # (cluster_label, body, distance)
        # label=-1 for store-loaded ContextBodyRecord objects
        # label>=0 for emergent ContextBody objects from IncrementalDBSCAN
        self.active_bodies: list[tuple[int, _GravitySource, float]] = []
        self.clustering: IncrementalDBSCAN | None = None

        # maps DBSCAN cluster label -> UUID of the persisted ContextBodyRecord
        self._label_record_ids: dict[int, str] = {}

        # escape rate tracking
        self._last_escape_count: int = 0
        self._last_vocab_size: int = 0
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

        # Cached numpy copy of the vocab embedding matrix.
        # Populated on first sample() call; avoids a ~154MB GPU→CPU copy every step.
        self._cached_token_embs_np: np.ndarray | None = None
        self._cached_token_embs_id: int = -1

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
        embs_np = context_embeddings.cpu().numpy()
        initial_pos = embs_np[0]

        if self.domain_classifier is not None:
            self.domain = self.domain_classifier.seed(embs_np)

        self.orbital_state = OrbitalState.initialize(initial_pos)

        self.clustering = IncrementalDBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            dim=embedding_dim,
        )

        for i in range(context_embeddings.shape[0]):
            self.clustering.update(token_id=-i, embedding=embs_np[i])

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
        """Derive token mass from model weight norms: m = ||W[token_id]|| / G."""
        weight_norm = torch.norm(weight_matrix[token_id]).item()
        return weight_norm / self.G

    def _gravitational_field_strength(
        self,
        token_embedding: np.ndarray,
        token_mass: float,
        body_centroid: np.ndarray,
        body_mass: float,
    ) -> float:
        """
        Scalar gravitational field strength at a token's location.

            Φ = G * (m_token * m_body) / r^2

        Used for logit biasing in sample(). Correctly returns a large value
        when the token is very close to or coincident with the body centroid.
        """
        c_tok = token_embedding / (np.linalg.norm(token_embedding) + 1e-8)
        c_body = body_centroid / (np.linalg.norm(body_centroid) + 1e-8)
        r = float(1.0 - np.dot(c_tok, c_body))
        r = max(r, 1e-6)
        return self.G * token_mass * body_mass / (r ** 2)

    def _gravitational_force(
        self,
        token_embedding: np.ndarray,
        token_mass: float,
        body_centroid: np.ndarray,
        body_mass: float,
    ) -> np.ndarray:
        """
        Gravitational force vector on a token from a body.

            F = G * (m_token * m_body) / r^2  *  r_hat

        Used for orbital state updates and resonance forces. Returns zero
        vector when token is coincident with the body (direction undefined).
        For logit biasing use _gravitational_field_strength() instead.
        """
        direction = body_centroid - token_embedding
        r = float(1.0 - np.dot(
            token_embedding / (np.linalg.norm(token_embedding) + 1e-8),
            body_centroid / (np.linalg.norm(body_centroid) + 1e-8),
        ))
        r = max(r, 1e-6)  # avoid division by zero
        magnitude = self.G * token_mass * body_mass / (r ** 2)
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-8:
            return np.zeros_like(token_embedding)
        return magnitude * direction / dir_norm

    def _recency_factor(self, body: _GravitySource) -> float:
        """
        Recency decay factor for a gravity source.

        ContextBody (in-session): always 1.0.
        ContextBodyRecord: exp(-lambda * elapsed_seconds).
        lambda=0.0 disables decay (always 1.0).
        """
        if isinstance(body, ContextBody):
            return 1.0
        if self.recency_decay_lambda == 0.0:
            return 1.0
        elapsed = (datetime.utcnow() - body.last_seen).total_seconds()
        return float(np.exp(-self.recency_decay_lambda * elapsed))

    def _group_active_bodies(
        self,
    ) -> list[tuple[np.ndarray, float]]:
        """
        Group nearby active bodies into virtual bodies for amplified force.

        Bodies within amplification_threshold cosine distance of each other
        are merged into a single virtual body with summed mass (weighted by
        recency factor). Returns list of (centroid, effective_mass) pairs.
        """
        if not self.active_bodies:
            return []

        sources = [
            (body, self._recency_factor(body))
            for _, body, _ in self.active_bodies
        ]

        used = [False] * len(sources)
        groups: list[tuple[np.ndarray, float]] = []

        for i, (body_i, factor_i) in enumerate(sources):
            if used[i]:
                continue
            used[i] = True
            group_centroid = body_i.centroid.copy()
            group_mass = body_i.mass * factor_i

            for j, (body_j, factor_j) in enumerate(sources):
                if used[j] or i == j:
                    continue
                ci = body_i.centroid / (np.linalg.norm(body_i.centroid) + 1e-8)
                cj = body_j.centroid / (np.linalg.norm(body_j.centroid) + 1e-8)
                dist = float(1.0 - np.dot(ci, cj))
                if dist < self.amplification_threshold:
                    used[j] = True
                    group_mass += body_j.mass * factor_j
                    # mass-weighted centroid update
                    total = group_mass
                    group_centroid = (
                        group_centroid * (total - body_j.mass * factor_j)
                        + body_j.centroid * body_j.mass * factor_j
                    ) / (total + 1e-8)

            groups.append((group_centroid, group_mass))

        return groups

    def _compute_resonance_forces(
        self,
        token_embedding: np.ndarray,
        token_mass: float,
    ) -> np.ndarray:
        """
        Compute additional force from resonance pairs.

        For each pair of ContextBodyRecord objects in active_bodies with
        resonance score >= resonance_threshold, compute a force toward the
        Lagrange midpoint of the pair scaled by joint_mass = sqrt(m_A * m_B) * score.
        """
        force = np.zeros_like(token_embedding)
        records = [
            body for _, body, _ in self.active_bodies
            if isinstance(body, ContextBodyRecord)
        ]
        if len(records) < 2:
            return force

        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                ra, rb = records[i], records[j]
                score = ra.resonance_partners.get(str(rb.id), 0.0)
                if score < self.resonance_threshold:
                    continue
                midpoint = (ra.centroid + rb.centroid) / 2.0
                joint_mass = float(np.sqrt(ra.mass * rb.mass)) * score
                force += self._gravitational_force(
                    token_embedding, token_mass, midpoint, joint_mass
                )

        return force

    def _check_collisions(self) -> None:
        """
        Inelastic collision: merge any two ContextBody objects in active_bodies
        whose centroids are within collision_distance.

        Only ContextBody (label >= 0) objects participate.
        Merged body has summed mass and mass-weighted centroid.
        """
        changed = True
        while changed:
            changed = False
            body_entries = [
                (i, label, body)
                for i, (label, body, dist) in enumerate(self.active_bodies)
                if isinstance(body, ContextBody) and label >= 0
            ]
            for ai in range(len(body_entries)):
                for bi in range(ai + 1, len(body_entries)):
                    idx_a, label_a, body_a = body_entries[ai]
                    idx_b, label_b, body_b = body_entries[bi]
                    ca = body_a.centroid / (np.linalg.norm(body_a.centroid) + 1e-8)
                    cb = body_b.centroid / (np.linalg.norm(body_b.centroid) + 1e-8)
                    dist = float(1.0 - np.dot(ca, cb))
                    if dist < self.collision_distance:
                        # Merge b into a
                        total_mass = body_a.mass + body_b.mass
                        merged_centroid = (
                            body_a.centroid * body_a.mass
                            + body_b.centroid * body_b.mass
                        ) / (total_mass + 1e-8)
                        merged = ContextBody()
                        merged.centroid = merged_centroid
                        merged.mass = total_mass
                        merged.member_tokens = body_a.member_tokens | body_b.member_tokens
                        merged.stability = (body_a.stability + body_b.stability) / 2.0

                        # Remove both, add merged
                        remove_indices = sorted([idx_a, idx_b], reverse=True)
                        for ri in remove_indices:
                            self.active_bodies.pop(ri)
                        avg_dist = (
                            self.active_bodies[0][2] if self.active_bodies else 0.0
                        )
                        self.active_bodies.append((label_a, merged, avg_dist))
                        changed = True
                        break
                if changed:
                    break

    def sample(
        self,
        logits: torch.Tensor,         # [vocab_size]
        token_embeddings: torch.Tensor,  # [vocab_size, D]
        token_mass: float = 1.0,
    ) -> int:
        """
        Sample one token index from gravity-biased logits.

        Returns integer token index in [0, vocab_size).
        """
        vocab_size = logits.shape[0]
        # Cache the numpy embedding matrix — it doesn't change across steps,
        # but copying 50k × 768 floats from GPU every call costs ~150ms/step.
        emb_id = id(token_embeddings)
        if self._cached_token_embs_id != emb_id or self._cached_token_embs_np is None:
            self._cached_token_embs_np = token_embeddings.detach().cpu().numpy()
            self._cached_token_embs_id = emb_id
        embs_np = self._cached_token_embs_np  # [vocab_size, D]

        # Get virtual body groups
        groups = self._group_active_bodies()

        # Compute gravitational field strength for each candidate token.
        # Uses scalar field strength (G*m*M/r²) rather than force vector norm
        # so tokens coincident with a body centroid correctly receive maximum
        # bias rather than zero (force vector direction is undefined at r=0).
        force_magnitudes = np.zeros(vocab_size)
        for i in range(vocab_size):
            tok_emb = embs_np[i]
            # Scalar field strength from all virtual body groups
            field = 0.0
            for body_centroid, body_mass in groups:
                field += self._gravitational_field_strength(
                    tok_emb, token_mass, body_centroid, body_mass
                )
            # Vector resonance forces (midpoint attraction; direction meaningful)
            resonance_force = self._compute_resonance_forces(tok_emb, token_mass)
            force_magnitudes[i] = field + float(np.linalg.norm(resonance_force))

        # Escape rate tracking
        escape_count = int(np.sum(force_magnitudes < self.escape_threshold))
        self._last_escape_count = escape_count
        self._last_vocab_size = vocab_size

        # Bias logits
        bias = torch.tensor(force_magnitudes, dtype=logits.dtype, device=logits.device)
        adjusted_logits = logits + bias

        probs = torch.softmax(adjusted_logits, dim=-1)
        token_idx = int(torch.multinomial(probs, num_samples=1).item())
        return token_idx

    def post_step(
        self,
        token_id: int,
        token_embedding: np.ndarray,
        token_mass: float = 1.0,
    ) -> None:
        """
        Update clustering with the token that was just sampled.

        Must be called after every sample() call with the embedding of the
        returned token. This is what drives body formation — without it,
        active_bodies remains empty and the gravitational field is never built.

        Responsibilities:
            1. Feed the token into IncrementalDBSCAN.
            2. Rebuild active_bodies from current cluster state.
            3. Persist newly stable bodies to the store.
            4. Boost resonance for collision-event pairs.
            5. Run inelastic collision check on active_bodies.
        """
        if self.clustering is None:
            return

        _, merged_events, _, collision_events = self.clustering.update(
            token_id=token_id,
            embedding=token_embedding,
            token_mass=token_mass,
        )

        # Remove merged-away labels from the record mapping
        for old_label, _ in merged_events:
            self._label_record_ids.pop(old_label, None)

        # Update domain classifier
        if self.domain_classifier is not None:
            self.domain = self.domain_classifier.update(token_embedding)

        # Rebuild cluster portion of active_bodies from current DBSCAN state.
        # Keep store records (label=-1) in place; replace all label>=0 entries.
        store_entries = [
            entry for entry in self.active_bodies
            if not isinstance(entry[1], ContextBody)
        ]
        cluster_entries: list[tuple[int, ContextBody, float]] = []
        for label in self.clustering.clusters:
            body = self.clustering._make_body(label)
            cluster_entries.append((label, body, 0.0))

            # Persist body if stable and not yet persisted
            if (
                body.stability >= self.stability_threshold
                and label not in self._label_record_ids
                and body.mass > 0
            ):
                record_id = self.body_store.record(
                    centroid=body.centroid,
                    mass=body.mass,
                    stability=body.stability,
                    domain=self.domain,
                )
                self._label_record_ids[label] = str(record_id)

        self.active_bodies = store_entries + cluster_entries

        # Collision events → resonance boost for persisted pairs
        for label_a, label_b, _dist in collision_events:
            id_a = self._label_record_ids.get(label_a)
            id_b = self._label_record_ids.get(label_b)
            if id_a and id_b:
                self.body_store.record_resonance(id_a, id_b, delta=0.2)

        # Inelastic collision check within active bodies
        self._check_collisions()

        # Update adaptive G if configured
        if self.adaptive_g is not None:
            escape_rate = (
                self._last_escape_count / (self._last_vocab_size + 1e-8)
                if self._last_vocab_size > 0 else 0.0
            )
            self.G = self.adaptive_g.update(
                active_bodies=[body for _, body, _ in self.active_bodies],
                escape_rate=escape_rate,
                domain=self.domain,
            )

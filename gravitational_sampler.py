from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_dbscan import AdaptiveDBSCAN
from adaptive_g import AdaptiveG
from universe_builder import Universe
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
        G_universe: float = 1.0,
        escape_threshold: float = 0.01,
        stability_threshold: float = 0.8,
        resonance_threshold: float = 0.3,
        body_merge_distance: float = 0.2,
        collision_distance: float = 0.1,
        recency_decay_lambda: float = 0.0,
        adaptive_g: AdaptiveG | None = None,
        adaptive_dbscan: AdaptiveDBSCAN | None = None,
        universe: Universe | None = None,
        use_context_bodies: bool = True,
        deterministic: bool = False,
        domain: str = "",
        domain_classifier: DomainClassifier | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cluster_radius: float = 0.3,
        cluster_min_tokens: int = 3,
    ):
        self.body_store = body_store
        self.G = G
        self.G_universe = G_universe
        self.escape_threshold = escape_threshold
        self.stability_threshold = stability_threshold
        self.resonance_threshold = resonance_threshold
        self.body_merge_distance = body_merge_distance
        self.collision_distance = collision_distance
        self.recency_decay_lambda = recency_decay_lambda
        self.adaptive_g = adaptive_g
        self.adaptive_dbscan = adaptive_dbscan
        self.universe = universe
        self.use_context_bodies = use_context_bodies
        self.deterministic = deterministic
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
        self.cluster_radius = cluster_radius
        self.cluster_min_tokens = cluster_min_tokens

        # Cached embedding matrices populated on first sample() call.
        # numpy copies avoid repeated GPU→CPU transfers for the context body field.
        # The torch copy keeps the universe field computation on GPU.
        self._cached_token_embs_np: np.ndarray | None = None
        self._cached_token_embs_norm: np.ndarray | None = None       # row-normalized [vocab, D], CPU numpy
        self._cached_token_embs_norm_torch: torch.Tensor | None = None  # row-normalized [vocab, D], on device
        self._cached_token_embs_id: int = -1

        # IDF-style mass weights [vocab_size], computed from unconditional token
        # probabilities via precompute_idf_weights(). Common tokens (punctuation,
        # articles) get low weight; rare specific tokens get weight near 1.0.
        # None until precompute_idf_weights() is called.
        self._idf_weights: np.ndarray | None = None

        # Diagnostic state — populated each sample() call when log_diagnostics=True.
        # Access via get_last_diagnostic_info() after each step.
        self._last_diagnostic_info: dict | None = None

        # Tokenizer for readable token strings in diagnostics.
        # Set via set_tokenizer(); optional.
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def precompute_idf_weights(self, model: torch.nn.Module) -> None:
        """
        Compute IDF-style mass weights from the model's unconditional distribution.

        Runs one forward pass with the BOS token to get p(token | <BOS>) across
        the full vocabulary. Common tokens (punctuation, articles, EOS) receive
        low weight because the model assigns them high unconditional probability.
        Rare, semantically specific tokens receive weight near 1.0.

        Weights are stored in self._idf_weights[token_id] ∈ [0, 1] and applied
        in post_step() when accreting tokens onto DBSCAN clusters.

        Call this once before initialize(), or after if the model is loaded later.
        """
        model.eval()
        device = next(model.parameters()).device
        bos_id = getattr(model.config, "bos_token_id", None) or 0
        input_ids = torch.tensor([[bos_id]], device=device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]          # [vocab_size]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # IDF: common tokens have high p → low weight; rare tokens have low p → high weight
        idf = -np.log(probs + 1e-8)
        idf_min, idf_max = idf.min(), idf.max()
        self._idf_weights = (idf - idf_min) / (idf_max - idf_min + 1e-8)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def set_tokenizer(self, tokenizer) -> None:
        """Attach a tokenizer for human-readable token strings in diagnostic output."""
        self._tokenizer = tokenizer

    def get_last_diagnostic_info(self) -> "dict | None":
        """
        Return diagnostic data from the most recent sample() call.

        Returns None if sample() has not been called yet.

        Keys in returned dict:
            top_tokens  : list of dicts — top-k tokens by gravitational force:
                          {token_id, token_str, category, force, idf_weight,
                           base_prob, boosted_prob}
            force_mean  : float — mean force magnitude across vocabulary
            force_max   : float — maximum force magnitude
            n_influenced: int   — tokens with force > escape_threshold
            escape_rate : float — fraction of tokens below escape_threshold
            active_bodies: int  — number of active local bodies
        """
        return self._last_diagnostic_info

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

        # --- Context body layer (DBSCAN) — optional ---
        if self.use_context_bodies:
            if self.adaptive_dbscan is not None:
                eps = self.adaptive_dbscan.initialize_eps(embs_np)
                min_samples = self.adaptive_dbscan.min_samples
            else:
                eps = self.cluster_radius
                min_samples = self.cluster_min_tokens

            self.clustering = IncrementalDBSCAN(
                eps=eps,
                min_samples=min_samples,
                dim=embedding_dim,
            )
            for i in range(context_embeddings.shape[0]):
                self.clustering.update(token_id=-i, embedding=embs_np[i])
        else:
            self.clustering = None

        # --- Active bodies: universe (permanent) + store records (cross-session) ---
        # Universe bodies are not stored in active_bodies list — they are handled
        # separately in sample() via Universe.compute_field() for efficiency.
        # Store records are still loaded for cross-session resonance.
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
        elapsed = (datetime.now(timezone.utc) - body.last_seen).total_seconds()
        return float(np.exp(-self.recency_decay_lambda * elapsed))

    def _group_active_bodies(
        self,
    ) -> list[tuple[np.ndarray, float]]:
        """
        Group nearby active bodies into virtual bodies for amplified force.

        Bodies within body_merge_distance cosine distance of each other
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
                if dist < self.body_merge_distance:
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

    def _check_cross_session_collisions(self) -> None:
        """
        Inelastic collision between newly persisted local bodies and store
        records loaded from previous sessions.

        After a local body (ContextBody, label>=0) is persisted to the store,
        we check whether it overlaps with any ContextBodyRecord already loaded
        into active_bodies. If the cosine distance is below collision_distance,
        the two records are merged in the store: masses sum, centroid becomes
        the mass-weighted average, and the local body's label is redirected to
        the surviving store record.

        This ensures concepts that recur across sessions accumulate mass and
        become progressively stronger attractors, rather than spawning duplicate
        records on every re-encounter.
        """
        # Persisted local bodies: those with a label in _label_record_ids
        persisted = {
            label: (body, self._label_record_ids[label])
            for label, body, _ in self.active_bodies
            if isinstance(body, ContextBody) and label in self._label_record_ids
        }
        if not persisted:
            return

        # Store records from previous sessions currently in active_bodies
        store_entries = [
            body
            for _, body, _ in self.active_bodies
            if isinstance(body, ContextBodyRecord)
        ]
        if not store_entries:
            return

        for label, (local_body, local_id) in persisted.items():
            ca = local_body.centroid / (np.linalg.norm(local_body.centroid) + 1e-8)
            for store_rec in store_entries:
                if str(store_rec.id) == local_id:
                    continue  # same record — shouldn't occur but be safe
                cb = store_rec.centroid / (np.linalg.norm(store_rec.centroid) + 1e-8)
                dist = float(1.0 - np.dot(ca, cb))
                if dist < self.collision_distance:
                    # Cross-session inelastic collision: absorb the new local
                    # record into the older store record.
                    self.body_store.merge_records(
                        incoming_id=local_id,
                        incoming_centroid=local_body.centroid,
                        existing_record=store_rec,
                    )
                    # Redirect this label to the surviving store record.
                    self._label_record_ids[label] = str(store_rec.id)
                    break  # each local body collides at most once

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
        Sample one token index using multiplicative gravitational reweighting.

        Gravity amplifies the model's own probability distribution rather than
        overriding it with an additive logit bias. Tokens the model assigns
        near-zero probability stay near-zero regardless of gravitational pull.
        This preserves diversity while still steering sampling toward body regions.

        Pipeline:
            1. Compute raw softmax probabilities from the model's logits.
            2. Compute gravitational field strength per token (vectorized).
            3. Apply IDF weights to the field (suppresses common tokens).
            4. Multiply probabilities by (1 + field_strength) and renormalize.
            5. Sample from the reweighted distribution.

        Returns integer token index in [0, vocab_size).
        """
        vocab_size = logits.shape[0]

        # Cache the raw and L2-normalized embedding matrix.
        # The raw copy avoids ~150MB GPU→CPU transfer every step.
        # The normalized form is precomputed here so cosine similarities reduce
        # to a single matrix-vector multiply (embs_norm @ body_norm) per body.
        emb_id = id(token_embeddings)
        if self._cached_token_embs_id != emb_id or self._cached_token_embs_np is None:
            self._cached_token_embs_np = token_embeddings.detach().cpu().numpy()
            norms = np.linalg.norm(self._cached_token_embs_np, axis=1, keepdims=True)
            self._cached_token_embs_norm = self._cached_token_embs_np / (norms + 1e-8)
            # Torch version stays on device for the universe field matmul.
            with torch.no_grad():
                norms_t = torch.norm(token_embeddings, dim=1, keepdim=True)
                self._cached_token_embs_norm_torch = token_embeddings / (norms_t + 1e-8)
            self._cached_token_embs_id = emb_id
        embs_norm = self._cached_token_embs_norm  # [vocab_size, D], CPU numpy

        # Get virtual body groups
        groups = self._group_active_bodies()

        # Gravitational field from local context bodies — GPU torch path.
        #
        # All body centroids and masses are stacked into tensors so the entire
        # field can be computed as a single [vocab, D] @ [D, n_bodies] matmul,
        # matching the approach used in Universe.compute_field_torch().
        #
        # r_min=0.1 prevents singularities for tokens at their home centroid.
        # Forces are normalized by n_local_sources so the total local field
        # magnitude is comparable to the universe background regardless of how
        # many bodies are active.
        force_magnitudes_t: torch.Tensor | None = None

        # Build resonance sources (midpoints + joint masses) before the matmul.
        records = [
            body for _, body, _ in self.active_bodies
            if isinstance(body, ContextBodyRecord)
        ]
        resonance_centroids: list[np.ndarray] = []
        resonance_masses: list[float] = []
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                ra, rb = records[i], records[j]
                score = ra.resonance_partners.get(str(rb.id), 0.0)
                if score < self.resonance_threshold:
                    continue
                midpoint = (ra.centroid + rb.centroid) / 2.0
                resonance_centroids.append(midpoint)
                resonance_masses.append(float(np.sqrt(ra.mass * rb.mass)) * score)

        n_local_bodies = len(groups)
        n_resonance_pairs = len(resonance_centroids)
        n_local_sources = n_local_bodies + n_resonance_pairs

        if n_local_sources > 0 and self._cached_token_embs_norm_torch is not None:
            device = self._cached_token_embs_norm_torch.device
            dtype = self._cached_token_embs_norm_torch.dtype

            # Stack all centroid vectors: [n_local_sources, D]
            all_centroids_np = np.stack(
                [c for c, _ in groups] + resonance_centroids, axis=0
            )
            all_masses_np = np.array(
                [m for _, m in groups] + resonance_masses, dtype=np.float32
            )

            with torch.no_grad():
                centroids_t = torch.tensor(all_centroids_np, dtype=dtype, device=device)
                # L2-normalize each centroid row
                centroids_norm_t = centroids_t / (
                    torch.norm(centroids_t, dim=1, keepdim=True) + 1e-8
                )                                                     # [n_src, D]
                masses_t = torch.tensor(all_masses_np, dtype=dtype, device=device)

                # [vocab, n_src] cosine similarities via batched matmul
                cos_sims_t = self._cached_token_embs_norm_torch @ centroids_norm_t.T
                r_t = torch.clamp(1.0 - cos_sims_t, min=0.1)         # [vocab, n_src]
                forces_t = (
                    self.G * token_mass * masses_t / (r_t ** 2)
                )                                                     # [vocab, n_src]
                force_magnitudes_t = forces_t.sum(dim=1) / n_local_sources  # [vocab]

        # Convert to numpy for the escape-rate check and downstream ops.
        if force_magnitudes_t is not None:
            force_magnitudes = force_magnitudes_t.cpu().numpy()
        else:
            force_magnitudes = np.zeros(vocab_size)

        # Escape rate: fraction of tokens with negligible context body influence.
        # Measured HERE, from context body field only, before the universe field
        # is added. The universe is permanent background — every token is always
        # inside it. AdaptiveG should steer context gravity, not fight the
        # universe baseline, so universe forces must not count against escape rate.
        escape_count = int(np.sum(force_magnitudes < self.escape_threshold))
        self._last_escape_count = escape_count
        self._last_vocab_size = vocab_size

        # Universe field: batched matrix multiply over all universe bodies.
        # Context-affinity modulation: each body is weighted by its cosine
        # similarity to the current orbital position, so the universe field
        # reflects what's semantically nearby right now, not a fixed rare-token
        # boost. Bodies far from the current context contribute almost nothing.
        if self.universe is not None:
            context_pos = (
                self.orbital_state.position
                if self.orbital_state is not None
                else None
            )
            if self._cached_token_embs_norm_torch is not None:
                force_magnitudes += self.universe.compute_field_torch(
                    self._cached_token_embs_norm_torch, self.G_universe, token_mass,
                    context_pos=context_pos,
                )
            else:
                force_magnitudes += self.universe.compute_field(
                    embs_norm, self.G_universe, token_mass,
                    context_pos=context_pos,
                )

        # Apply IDF to the output field so common tokens (EOS, articles,
        # punctuation) receive reduced gravitational amplification even when
        # they are geometrically close to a body centroid.  This is independent
        # of the IDF weighting already applied at body formation in post_step();
        # together they suppress common tokens at both ingestion and sampling.
        if self._idf_weights is not None and len(self._idf_weights) == vocab_size:
            force_magnitudes *= self._idf_weights

        # Multiplicative reweighting: start from the model's own distribution
        # and amplify tokens near active bodies, rather than adding a flat bias
        # to raw logits.  Tokens with near-zero model probability stay near-zero
        # regardless of gravitational pull — the model's diversity is preserved.
        probs = torch.softmax(logits, dim=-1)
        base_probs_np: np.ndarray | None = None
        if np.any(force_magnitudes > 0):
            gravity_weights = torch.tensor(
                1.0 + force_magnitudes,
                dtype=logits.dtype,
                device=logits.device,
            )
            base_probs_np = probs.detach().cpu().numpy()
            probs = probs * gravity_weights
            probs = probs / probs.sum()

        # Build diagnostic info for this step (always populated so callers
        # can inspect the field even when log_diagnostics is False).
        self._last_diagnostic_info = self._build_diagnostic_info(
            force_magnitudes=force_magnitudes,
            base_probs_np=base_probs_np if base_probs_np is not None
                          else probs.detach().cpu().numpy(),
            boosted_probs_np=probs.detach().cpu().numpy(),
            vocab_size=vocab_size,
        )

        # Deterministic mode: argmax on the gravity-weighted distribution.
        # Diversity comes entirely from universe geometry rather than stochastic
        # sampling — same prompt always produces the same output.
        if self.deterministic:
            token_idx = int(probs.argmax().item())
        else:
            token_idx = int(torch.multinomial(probs, num_samples=1).item())
        return token_idx

    def _build_diagnostic_info(
        self,
        force_magnitudes: np.ndarray,
        base_probs_np: np.ndarray,
        boosted_probs_np: np.ndarray,
        vocab_size: int,
        top_k: int = 10,
    ) -> dict:
        """Build the diagnostic dict for the last sample() call."""
        top_indices = np.argsort(force_magnitudes)[-top_k:][::-1]
        idf_arr = self._idf_weights if self._idf_weights is not None else np.ones(vocab_size)

        _FUNCTION_WORDS = frozenset({
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "of", "in", "on", "at", "to", "for",
            "with", "from", "by", "as", "it", "its", "and", "or", "but",
            "not", "no", "i", "we", "you", "he", "she", "they",
        })
        _PUNCT = frozenset(".,;:!?()[]{}'\"-_")

        def _category(s: str) -> str:
            if not s or s.isspace():
                return "whitespace"
            if s.startswith("<") and s.endswith(">"):
                return "eos/special"
            inner = s.strip()
            if all(c in _PUNCT for c in inner):
                return "punctuation"
            if inner.lower() in _FUNCTION_WORDS:
                return "function_word"
            if s.startswith(" "):
                return "content_word"
            return "subword"

        top_tokens = []
        for idx in top_indices:
            idx_int = int(idx)
            if self._tokenizer is not None:
                try:
                    tok_str = self._tokenizer.decode([idx_int])
                except Exception:
                    tok_str = f"<id={idx_int}>"
            else:
                tok_str = f"<id={idx_int}>"

            top_tokens.append({
                "token_id":   idx_int,
                "token_str":  tok_str,
                "category":   _category(tok_str),
                "force":      float(force_magnitudes[idx_int]),
                "idf_weight": float(idf_arr[idx_int]) if idx_int < len(idf_arr) else 1.0,
                "base_prob":  float(base_probs_np[idx_int]) if idx_int < len(base_probs_np) else 0.0,
                "boosted_prob": float(boosted_probs_np[idx_int]) if idx_int < len(boosted_probs_np) else 0.0,
            })

        n_influenced = int(np.sum(force_magnitudes >= self.escape_threshold))
        escape_rate = 1.0 - (n_influenced / (vocab_size + 1e-8))

        return {
            "top_tokens":    top_tokens,
            "force_mean":    float(force_magnitudes.mean()),
            "force_max":     float(force_magnitudes.max()),
            "n_influenced":  n_influenced,
            "escape_rate":   round(escape_rate, 4),
            "active_bodies": len(self.active_bodies),
        }

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
        """
        if self.domain_classifier is not None:
            self.domain = self.domain_classifier.update(token_embedding)

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

        if self.clustering is None:
            return

        effective_mass = token_mass
        if (
            self._idf_weights is not None
            and token_id >= 0
            and token_id < len(self._idf_weights)
        ):
            effective_mass = token_mass * float(self._idf_weights[token_id])

        _, merged_events, _, collision_events = self.clustering.update(
            token_id=token_id,
            embedding=token_embedding,
            token_mass=effective_mass,
        )

        for old_label, _ in merged_events:
            self._label_record_ids.pop(old_label, None)

        store_entries = [
            entry for entry in self.active_bodies
            if not isinstance(entry[1], ContextBody)
        ]
        cluster_entries: list[tuple[int, ContextBody, float]] = []
        for label in self.clustering.clusters:
            body = self.clustering._make_body(label)
            cluster_entries.append((label, body, 0.0))

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

        for label_a, label_b, _dist in collision_events:
            id_a = self._label_record_ids.get(label_a)
            id_b = self._label_record_ids.get(label_b)
            if id_a and id_b:
                self.body_store.record_resonance(id_a, id_b, delta=0.2)

        self._check_collisions()
        self._check_cross_session_collisions()

        if self.adaptive_dbscan is not None:
            new_eps = self.adaptive_dbscan.update(len(self.active_bodies))
            self.clustering.eps = new_eps

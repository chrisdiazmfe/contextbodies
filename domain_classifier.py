from __future__ import annotations

import numpy as np


class DomainClassifier:
    """
    Infers the current conversation domain from the token embedding stream.

    Maintains an exponential moving average (EMA) of token embeddings — the
    "context direction" — and classifies it against a set of known domain anchor
    embeddings by cosine distance. The nearest anchor within `match_threshold`
    wins; if nothing is close enough the `fallback_domain` is used.

    The EMA smooths over individual token noise while still tracking topic drift.
    A low `ema_alpha` (e.g. 0.05) makes the context direction sticky — slow to
    change domain mid-conversation. A high value (e.g. 0.3) makes it reactive,
    switching domain as soon as the token stream shifts.

    Domain anchors are L2-normalized embeddings of representative text for each
    domain. They can be:

        Pre-defined:
            anchors = {"code": embed("python function class"),
                       "medical": embed("diagnosis treatment patient")}
            clf = DomainClassifier(anchors)

        Added incrementally:
            clf = DomainClassifier({})
            clf.add_anchor("ml", embed("neural network gradient descent"))

        Derived from a ContextBodyStore (see from_body_centroids()):
            clf = DomainClassifier.from_body_centroids(bodies_by_domain)

    Integration with GravitationalSampler:
        sampler = GravitationalSampler(store, domain_classifier=clf)

    When a domain_classifier is provided, the sampler's active domain is
    updated automatically on every generated token, so body queries and
    persistence always use the inferred domain rather than a hard-coded string.

    Note: domain changes mid-conversation update the domain used for recording
    new bodies and future queries but do not retroactively reload active bodies.
    Bodies from the previous domain remain gravitationally active for the session.
    """

    def __init__(
        self,
        domain_anchors: dict[str, np.ndarray],
        ema_alpha: float = 0.1,
        match_threshold: float = 0.3,
        fallback_domain: str = "general",
    ):
        """
        Parameters
        ----------
        domain_anchors   : {domain_name: anchor_embedding} — representative
                           embeddings for each known domain. Will be L2-normalized.
        ema_alpha        : EMA weight for incoming tokens. Higher = more reactive.
        match_threshold  : Max cosine distance to accept a domain match.
                           If the nearest anchor is farther than this, returns
                           fallback_domain.
        fallback_domain  : Domain string returned when no anchor matches.
        """
        self.ema_alpha = ema_alpha
        self.match_threshold = match_threshold
        self.fallback_domain = fallback_domain

        # store L2-normalized anchors for cheap cosine similarity
        self.domain_anchors: dict[str, np.ndarray] = {
            name: self._normalize(emb)
            for name, emb in domain_anchors.items()
        }

        self._context_direction: np.ndarray | None = None
        self._current_domain: str = fallback_domain

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def seed(self, embeddings: np.ndarray) -> str:
        """
        Initialize the context direction from prompt token embeddings.

        Takes the mean of all prompt embeddings as the starting context
        direction, then classifies immediately.

        Parameters
        ----------
        embeddings : np.ndarray of shape [n_tokens, dim]

        Returns the initial domain classification.
        """
        mean_emb = embeddings.mean(axis=0)
        self._context_direction = self._normalize(mean_emb)
        self._current_domain = self._classify()
        return self._current_domain

    def update(self, token_embedding: np.ndarray) -> str:
        """
        Update context direction with a new token embedding and reclassify.

        Uses an EMA so the direction shifts smoothly rather than jumping on
        individual tokens:

            direction = (1 - α) * direction + α * normalize(token_embedding)

        Returns the current domain (may be the same as before).
        """
        norm_emb = self._normalize(token_embedding)

        if self._context_direction is None:
            self._context_direction = norm_emb
        else:
            raw = (1.0 - self.ema_alpha) * self._context_direction + self.ema_alpha * norm_emb
            self._context_direction = self._normalize(raw)

        self._current_domain = self._classify()
        return self._current_domain

    def add_anchor(self, domain: str, embedding: np.ndarray) -> None:
        """
        Add or update a domain anchor embedding.

        Useful for registering new domains at runtime or overriding existing ones.
        The embedding is L2-normalized before storage.
        """
        self.domain_anchors[domain] = self._normalize(embedding)

    def remove_anchor(self, domain: str) -> None:
        """Remove a domain anchor. If it was the current domain, reclassifies."""
        self.domain_anchors.pop(domain, None)
        if self._current_domain == domain:
            self._current_domain = self._classify()

    @property
    def current_domain(self) -> str:
        """The most recently classified domain."""
        return self._current_domain

    @property
    def context_direction(self) -> np.ndarray | None:
        """Current EMA context direction vector (L2-normalized), or None before seeding."""
        return self._context_direction

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_body_centroids(
        cls,
        bodies_by_domain: dict[str, list[np.ndarray]],
        ema_alpha: float = 0.1,
        match_threshold: float = 0.3,
        fallback_domain: str = "general",
    ) -> DomainClassifier:
        """
        Build domain anchors from lists of body centroids grouped by domain.

        The anchor for each domain is the mean of its body centroids — the
        geometric center of the domain's known concept space.

        Typical usage: pull centroids from a ContextBodyStore query per domain
        and pass them here. This makes domain anchors emerge from the store's
        accumulated knowledge rather than requiring hand-crafted embeddings.

        Example:
            bodies_by_domain = {
                "code":    [body.centroid for body in code_bodies],
                "medical": [body.centroid for body in medical_bodies],
            }
            clf = DomainClassifier.from_body_centroids(bodies_by_domain)
        """
        anchors = {}
        for domain, centroids in bodies_by_domain.items():
            if centroids:
                anchors[domain] = np.mean(centroids, axis=0)
        return cls(
            anchors,
            ema_alpha=ema_alpha,
            match_threshold=match_threshold,
            fallback_domain=fallback_domain,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-8)

    def _classify(self) -> str:
        """
        Find the nearest domain anchor to the current context direction.
        Returns fallback_domain if no anchor is within match_threshold.
        """
        if self._context_direction is None or not self.domain_anchors:
            return self.fallback_domain

        best_domain = self.fallback_domain
        best_dist = float("inf")

        for domain, anchor in self.domain_anchors.items():
            dist = float(1.0 - np.dot(self._context_direction, anchor))
            if dist < best_dist:
                best_dist = dist
                best_domain = domain

        return best_domain if best_dist <= self.match_threshold else self.fallback_domain

from __future__ import annotations

from collections import deque

import numpy as np

from context_body import ContextBody
from context_body_record import ContextBodyRecord

_GravitySource = ContextBody | ContextBodyRecord


class AdaptiveG:
    """
    Adaptive gravitational constant that adjusts G at each sampling step.

    Three mechanisms compose multiplicatively:

        G_eff = G_base
                * mass_norm(t)        # keeps force magnitude stable as bodies grow
                * escape_feedback(t)  # PID controller toward target escape fraction
                * domain_scale        # per-domain tuning knob

    --- Mass normalization ---

    Body mass grows continuously as tokens accrete. Without compensation, a
    mature body in a long conversation exerts orders of magnitude more force than
    the same body 50 tokens earlier, eventually overwhelming the base distribution.

    The fix: track an exponential moving average of active body mass and scale G
    inversely. If mean body mass doubles, G halves, keeping F = G*m_token*m_body/r²
    roughly constant in expectation.

        mass_norm(t) = mass_ref / (ema_body_mass(t) + ε)

    mass_ref is set on the first call from the initial mean body mass, so
    mass_norm starts at 1.0 and deviates only as bodies grow or shrink.

    --- Escape rate feedback ---

    At each step, some fraction of candidate tokens have force_magnitude below
    escape_threshold (they're gravitationally unbound). This "escape rate" is a
    signal of how much freedom the sampler has.

    Target escape rate is a hyperparameter: ~0.7 means 70% of tokens are
    uninfluenced each step — the model retains most of its base distribution
    while gravity steers it. Lower values = tighter constraint.

    The controller integrates error over a window and applies a proportional
    correction to G:

        error(t)        = escape_rate_target - escape_rate_observed(t)
        integral_error  = mean(error over window)
        G_feedback(t)   = clamp(1 + Kp*error + Ki*integral_error, G_floor, G_ceil)

    Kp controls how aggressively G responds to a single step's escape rate.
    Ki controls the slow drift correction for persistent bias.

    --- Domain multiplier ---

    Different output goals need different constraint levels:
        code / legal / medical  → high G (stay precise, penalize semantic drift)
        creative / brainstorm   → low G (let tokens escape into novel territory)
        general                 → 1.0 (no adjustment)

    Multipliers are applied after mass_norm and escape_feedback, so they scale
    the final effective G rather than interacting with the feedback loop.

    --- Model scale ---

    Body mass is derived from model weight norms (||W[token_id]|| / G), so larger
    models have naturally heavier tokens and bodies. mass_ref is initialized from
    the first batch of active bodies, anchoring the normalization to the actual
    scale of the model in use. This makes G_base transferable across model families.

    Parameters
    ----------
    G_base              : Starting gravitational constant. Primary tuning knob.
    escape_rate_target  : Fraction of tokens that should escape gravity each step.
                          0.7 is a reasonable default (loose constraint).
                          Lower → tighter; higher → looser.
    Kp                  : Proportional gain for escape rate feedback.
                          Higher = more reactive to single-step fluctuations.
    Ki                  : Integral gain. Corrects persistent bias over the window.
    feedback_window     : Number of steps over which integral error is averaged.
    mass_ema_alpha      : EMA smoothing for body mass tracking.
                          Lower = smoother but slower to react to mass changes.
    G_min               : Hard floor on G_eff. Prevents gravity from collapsing to 0.
    G_max               : Hard ceiling on G_eff. Prevents runaway amplification.
    domain_multipliers  : {domain_name: scalar} — applied after other adjustments.
    """

    def __init__(
        self,
        G_base: float = 1.0,
        escape_rate_target: float = 0.7,
        Kp: float = 0.1,
        Ki: float = 0.01,
        feedback_window: int = 20,
        mass_ema_alpha: float = 0.05,
        G_min: float = 0.01,
        G_max: float = 10.0,
        domain_multipliers: dict[str, float] | None = None,
    ):
        self.G_base = G_base
        self.escape_rate_target = escape_rate_target
        self.Kp = Kp
        self.Ki = Ki
        self.feedback_window = feedback_window
        self.mass_ema_alpha = mass_ema_alpha
        self.G_min = G_min
        self.G_max = G_max
        self.domain_multipliers: dict[str, float] = domain_multipliers or {}

        # state
        self._ema_body_mass: float | None = None   # lazily initialized
        self._mass_ref: float | None = None        # anchored on first update
        self._error_history: deque[float] = deque(maxlen=feedback_window)
        self._G_feedback: float = 1.0              # current feedback multiplier
        self._G_eff: float = G_base                # last computed effective G

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        active_bodies: list[tuple[int, _GravitySource, float]],
        escape_rate: float,
        domain: str = "",
    ) -> float:
        """
        Compute the effective G for the current sampling step.

        Call once per token, after force magnitudes have been computed but before
        the next token is sampled. The returned G_eff should be used in
        GravitationalSampler for the *next* step (look-ahead correction).

        Parameters
        ----------
        active_bodies : sampler's active_bodies list — used to extract body masses
        escape_rate   : fraction of candidate tokens whose force_magnitude was
                        below escape_threshold this step (computed by the sampler)
        domain        : current domain string (from sampler.domain)

        Returns
        -------
        G_eff : the gravitational constant to use next step
        """
        mass_norm = self._update_mass_norm(active_bodies)
        feedback = self._update_escape_feedback(escape_rate)
        domain_scale = self.domain_multipliers.get(domain, 1.0)

        G_eff = self.G_base * mass_norm * feedback * domain_scale
        self._G_eff = float(np.clip(G_eff, self.G_min, self.G_max))
        return self._G_eff

    @property
    def G(self) -> float:
        """Current effective G. Use this in the sampler instead of a static G."""
        return self._G_eff

    @property
    def mass_norm(self) -> float:
        """Current mass normalization factor (G_base multiplier from body mass)."""
        if self._ema_body_mass is None or self._mass_ref is None:
            return 1.0
        return self._mass_ref / (self._ema_body_mass + 1e-8)

    @property
    def feedback_multiplier(self) -> float:
        """Current escape rate feedback multiplier."""
        return self._G_feedback

    def set_domain_multiplier(self, domain: str, scale: float) -> None:
        """Register or update a per-domain G multiplier."""
        self.domain_multipliers[domain] = scale

    def reset(self) -> None:
        """Reset all state — call between independent conversations."""
        self._ema_body_mass = None
        self._mass_ref = None
        self._error_history.clear()
        self._G_feedback = 1.0
        self._G_eff = self.G_base

    # ------------------------------------------------------------------
    # Internal: mass normalization
    # ------------------------------------------------------------------

    def _update_mass_norm(
        self, active_bodies: list[tuple[int, _GravitySource, float]]
    ) -> float:
        """
        Update EMA of mean body mass and return the normalization factor.

        Mass is summed across all active bodies (both ContextBody and
        ContextBodyRecord) and averaged. The EMA smooths over the noisy
        per-step mass estimates that result from DBSCAN cluster churn.

        On the first call, _mass_ref is anchored to the initial mean mass
        so mass_norm starts at exactly 1.0 and deviates only as mass changes.
        """
        if not active_bodies:
            return 1.0

        masses = [body.mass for body in active_bodies if body.mass > 0]
        if not masses:
            return 1.0

        current_mean = float(np.mean(masses))

        if self._ema_body_mass is None:
            # first step: anchor reference and EMA at the same value
            self._ema_body_mass = current_mean
            self._mass_ref = current_mean
            return 1.0

        # EMA update
        self._ema_body_mass = (
            (1.0 - self.mass_ema_alpha) * self._ema_body_mass
            + self.mass_ema_alpha * current_mean
        )

        return self._mass_ref / (self._ema_body_mass + 1e-8)

    # ------------------------------------------------------------------
    # Internal: escape rate feedback
    # ------------------------------------------------------------------

    def _update_escape_feedback(self, escape_rate: float) -> float:
        """
        PI controller on escape rate → feedback multiplier for G.

        error > 0: too many tokens escaping → gravity too weak → raise G
        error < 0: too few tokens escaping  → gravity too strong → lower G

        The proportional term reacts immediately; the integral term corrects
        slow drift (e.g. G consistently too high across many steps).

        G_feedback is clamped relative to 1.0 so it only scales G_base,
        not replace it. The hard G_min/G_max on G_eff provide the outer guard.
        """
        error = self.escape_rate_target - escape_rate
        self._error_history.append(error)

        integral_error = float(np.mean(self._error_history))

        # positive error → raise G (more gravity needed)
        # negative error → lower G (less gravity needed)
        correction = self.Kp * error + self.Ki * integral_error
        self._G_feedback = float(np.clip(1.0 + correction, 0.1, 10.0))

        return self._G_feedback

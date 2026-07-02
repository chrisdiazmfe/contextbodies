"""
semantic_force_processor.py — HuggingFace-compatible gravitational sampling.

Drop-in LogitsProcessor that applies the contextbodies gravitational field
to next-token scores at each generation step.  Works with model.generate()
alongside top_p, top_k, temperature, or any other standard sampling controls.

Quick start
-----------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from universe_builder import Universe
    from semantic_force_processor import SemanticForceProcessor

    model     = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    universe  = Universe.load("universe.npz")

    processor = SemanticForceProcessor.from_model(model, universe, G=1.0)
    processor.set_tokenizer(tokenizer)          # optional: enables readable diagnostics

    inputs = tokenizer("The history of computing began with", return_tensors="pt")
    outputs = model.generate(
        **inputs,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        logits_processor=[processor],
        renormalize_logits=True,       # recommended: keeps normalization clean
        max_new_tokens=100,
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

Design note: logit vs. multiplicative reweighting
--------------------------------------------------
GravitationalSampler uses multiplicative reweighting:

    p'(x) ∝ p(x) * (1 + force_x)

LogitsProcessor must operate on scores (logits) before softmax.  For the
softmax numerator:

    log(p(x) * (1 + F_x))  =  log p(x)  +  log(1 + F_x)

So the logit-side equivalent is:

    scores'_x  =  scores_x  +  log1p(force_x)

This is mathematically equivalent up to the partition function (which
softmax normalisation absorbs), and integrates cleanly with downstream
top_p / top_k truncation that HuggingFace applies after logits_processor.

Set renormalize_logits=True in model.generate() so the pipeline handles
normalisation explicitly — prevents accumulation of floating-point drift
over long sequences.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from transformers import LogitsProcessor
except ImportError:  # pragma: no cover
    class LogitsProcessor:  # type: ignore
        """Minimal fallback when transformers is not installed."""
        def __call__(self, input_ids, scores):
            return scores

from universe_builder import Universe


# ---------------------------------------------------------------------------
# Token category classifier (for diagnostics)
# ---------------------------------------------------------------------------

_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "must", "can", "of", "in", "on",
    "at", "to", "for", "with", "from", "by", "as", "it", "its", "this",
    "that", "these", "those", "i", "we", "you", "he", "she", "they", "them",
    "their", "our", "your", "my", "his", "her", "and", "or", "but", "not",
    "no", "nor", "so", "yet", "both", "either", "neither", "whether",
    "if", "then", "than", "when", "while", "although", "because", "since",
    "though", "unless", "until", "after", "before", "about", "between",
})

_PUNCTUATION_CHARS = frozenset(".,;:!?()[]{}'\"-_—…/\\|@#$%^&*+=<>~`")


def _token_category(tok_str: str) -> str:
    """
    Classify a decoded token string for diagnostic display.

    Categories:
        whitespace   — empty or all whitespace
        punctuation  — all punctuation characters
        function_word — closed-class grammatical words
        eos/special  — special tokens (<|endoftext|> etc.)
        content_word — space-prefixed content word (typical GPT-2 tokenisation)
        subword      — non-space-prefixed fragment
    """
    s = tok_str.strip()
    if not s:
        return "whitespace"
    if s.startswith("<") and s.endswith(">"):
        return "eos/special"
    if all(c in _PUNCTUATION_CHARS for c in s.replace(" ", "")):
        return "punctuation"
    if s.lower() in _FUNCTION_WORDS:
        return "function_word"
    if tok_str.startswith(" "):
        return "content_word"
    return "subword"


# ---------------------------------------------------------------------------
# SemanticForceProcessor
# ---------------------------------------------------------------------------

class SemanticForceProcessor(LogitsProcessor):
    """
    HuggingFace LogitsProcessor applying the contextbodies gravitational field.

    Parameters
    ----------
    universe : Universe
        Pre-computed vocabulary clustering (UniverseBuilder.build or Universe.load).
    token_embeddings : torch.Tensor
        The model's input embedding matrix [vocab_size, D].  Pass
        model.get_input_embeddings().weight.detach().
    G : float
        Gravitational constant.  Higher values = stronger topical steering.
        Start with G=1.0; sweep to find the perplexity-diversity tradeoff
        that suits the task.
    idf_weights : np.ndarray | None
        Per-token IDF weights [vocab_size] in [0, 1].  Common tokens receive
        low weight; rare, specific tokens receive weight near 1.0.  Pass None
        to disable common-token suppression (useful for the no-IDF ablation).
    token_mass : float
        Scalar token mass used in F = G * m_token * m_body / r².
        Typically 1.0; adjusting changes the overall force scale.
    context_window : int
        Number of recent tokens used to compute the current context position
        for universe affinity modulation.  Larger windows track slower drift;
        smaller windows react to recent topic changes.
    log_diagnostics : bool
        If True, print a table of top boosted tokens at each generation step.
        Useful for verifying the field is steering toward meaningful content.
    top_k_diag : int
        Number of top boosted tokens to show per step when log_diagnostics=True.
    diag_every_n_steps : int
        Print diagnostics every N steps (default 1 = every step).
        Set higher to reduce verbosity during long generations.
    """

    def __init__(
        self,
        universe: Universe,
        token_embeddings: torch.Tensor,   # [vocab_size, D]
        G: float = 1.0,
        idf_weights: np.ndarray | None = None,
        token_mass: float = 1.0,
        context_window: int = 32,
        log_diagnostics: bool = False,
        top_k_diag: int = 10,
        diag_every_n_steps: int = 1,
    ):
        self.universe = universe
        self.G = G
        self.idf_weights = idf_weights
        self.token_mass = token_mass
        self.context_window = context_window
        self.log_diagnostics = log_diagnostics
        self.top_k_diag = top_k_diag
        self.diag_every_n_steps = diag_every_n_steps
        self._step = 0
        self._tokenizer = None

        # Cache L2-normalised embeddings.  Keep a GPU copy for fast context-position
        # computation and a CPU numpy copy for the universe field path.
        with torch.no_grad():
            norms = torch.norm(token_embeddings, dim=1, keepdim=True)
            embs_norm = token_embeddings / (norms + 1e-8)
        # Store on CPU to avoid issues if model moves between devices later.
        self._embs_norm_cpu: torch.Tensor = embs_norm.detach().cpu()
        self._embs_norm_np: np.ndarray = self._embs_norm_cpu.numpy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_tokenizer(self, tokenizer) -> None:
        """Attach tokenizer for human-readable token strings in diagnostics."""
        self._tokenizer = tokenizer

    def reset(self) -> None:
        """Reset step counter between generation calls."""
        self._step = 0

    # ------------------------------------------------------------------
    # LogitsProcessor protocol
    # ------------------------------------------------------------------

    def __call__(
        self,
        input_ids: torch.LongTensor,    # [batch, seq_len]
        scores: torch.FloatTensor,      # [batch, vocab_size]
    ) -> torch.FloatTensor:
        """
        Apply gravitational field update to generation scores.

        Called once per generation step by model.generate().
        Computes the universe field modulated by the current context position,
        then updates scores via: scores' = scores + log1p(force * idf_weight)
        """
        self._step += 1

        # Context position: mean of recent token embeddings (L2-normalised)
        context_pos = self._context_position(input_ids)

        # Universe gravitational field [vocab_size], numpy
        # Uses cached GPU tensors inside universe if available.
        device = scores.device
        embs_on_device = self._embs_norm_cpu.to(device=device, dtype=scores.dtype)
        force_np = self.universe.compute_field_torch(
            embs_on_device, self.G, self.token_mass, context_pos=context_pos,
        )  # [vocab_size], numpy

        # Apply IDF: common tokens get near-zero boost even when geometrically close.
        if self.idf_weights is not None:
            vocab_size = scores.shape[-1]
            if len(self.idf_weights) == vocab_size:
                force_np = force_np * self.idf_weights

        # Diagnostics: print top boosted tokens
        if self.log_diagnostics and (self._step % self.diag_every_n_steps == 0):
            self._print_diagnostics(force_np, scores)

        # Logit-side equivalent of multiplicative reweighting:
        #   scores' = scores + log1p(force)
        # Equivalent to: p' ∝ p * (1 + force)  up to the normalisation constant.
        force_t = torch.tensor(
            force_np, dtype=scores.dtype, device=scores.device,
        )                                             # [vocab]
        log_boost = torch.log1p(force_t).unsqueeze(0)  # [1, vocab] — broadcast over batch
        return scores + log_boost

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: "torch.nn.Module",
        universe: Universe,
        G: float = 1.0,
        use_idf: bool = True,
        context_window: int = 32,
        log_diagnostics: bool = False,
        top_k_diag: int = 10,
        diag_every_n_steps: int = 1,
    ) -> "SemanticForceProcessor":
        """
        Build a SemanticForceProcessor directly from a loaded model.

        Computes IDF weights from the model's unconditional distribution when
        use_idf=True (recommended).  Call set_tokenizer() on the result if
        you want readable token strings in diagnostic output.

        Example
        -------
            model     = AutoModelForCausalLM.from_pretrained("gpt2")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            universe  = Universe.load("universe.npz")

            processor = SemanticForceProcessor.from_model(model, universe, G=1.0)
            processor.set_tokenizer(tokenizer)

            inputs = tokenizer("The history of computing began with", return_tensors="pt")
            outputs = model.generate(
                **inputs,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                logits_processor=[processor],
                renormalize_logits=True,
                max_new_tokens=100,
            )
        """
        model.eval()
        device = next(model.parameters()).device
        token_embeddings = model.get_input_embeddings().weight.detach().cpu()

        idf_weights: np.ndarray | None = None
        if use_idf:
            bos_id = getattr(model.config, "bos_token_id", None) or 0
            input_ids = torch.tensor([[bos_id]], device=device)
            with torch.no_grad():
                outputs = model(input_ids)
                probs = (
                    torch.softmax(outputs.logits[0, -1, :], dim=-1)
                    .cpu()
                    .numpy()
                )
            idf = -np.log(probs + 1e-8)
            idf_min, idf_max = idf.min(), idf.max()
            idf_weights = (idf - idf_min) / (idf_max - idf_min + 1e-8)

        return cls(
            universe=universe,
            token_embeddings=token_embeddings,
            G=G,
            idf_weights=idf_weights,
            context_window=context_window,
            log_diagnostics=log_diagnostics,
            top_k_diag=top_k_diag,
            diag_every_n_steps=diag_every_n_steps,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _context_position(self, input_ids: torch.LongTensor) -> np.ndarray:
        """
        Compute the current context position as mean of recent token embeddings.

        Uses the last context_window tokens so the position tracks recent
        topic rather than the entire sequence history.
        """
        ids = input_ids[0]                            # [seq_len]
        window_ids = ids[-self.context_window:]       # last N tokens
        embs = self._embs_norm_cpu[window_ids.cpu()]  # [window, D]
        pos = embs.mean(dim=0).numpy()                # [D]
        norm = np.linalg.norm(pos)
        return pos / (norm + 1e-8)

    def _print_diagnostics(
        self,
        force_np: np.ndarray,
        scores: torch.FloatTensor,
    ) -> None:
        """Print a table of top-k boosted tokens for this step."""
        k = min(self.top_k_diag, len(force_np))
        top_indices = np.argsort(force_np)[-k:][::-1]

        # Base probabilities (before any boost)
        base_probs = torch.softmax(scores[0], dim=-1).detach().cpu().numpy()

        # Boosted probabilities: p' ∝ p * (1 + force)
        boosted_probs_raw = base_probs * (1.0 + force_np)
        boosted_probs = boosted_probs_raw / (boosted_probs_raw.sum() + 1e-10)

        print(f"\n[SemanticForceProcessor] step={self._step}  "
              f"G={self.G}  idf={'on' if self.idf_weights is not None else 'off'}")
        print(f"  force stats: mean={force_np.mean():.5f}  "
              f"max={force_np.max():.4f}  "
              f"n_nonzero={int((force_np > 0.001).sum())}")
        print()
        hdr = f"  {'rank':>4}  {'token_id':>8}  {'token':>22}  "
        hdr += f"{'force':>9}  {'idf_w':>7}  {'base_p':>9}  {'boost_p':>9}  {'category':>12}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        idf_arr = self.idf_weights if self.idf_weights is not None else np.ones(len(force_np))

        for rank, idx in enumerate(top_indices, 1):
            idx_int = int(idx)
            force_val = float(force_np[idx_int])
            idf_val = float(idf_arr[idx_int]) if len(idf_arr) > idx_int else 1.0
            base_p = float(base_probs[idx_int])
            boost_p = float(boosted_probs[idx_int])

            if self._tokenizer is not None:
                try:
                    tok_str = self._tokenizer.decode([idx_int])
                except Exception:
                    tok_str = f"<id={idx_int}>"
            else:
                tok_str = f"<id={idx_int}>"

            category = _token_category(tok_str)
            # Truncate for display
            tok_display = (tok_str[:20] + "…") if len(tok_str) > 21 else tok_str

            print(
                f"  {rank:>4}  {idx_int:>8}  {tok_display!r:>22}  "
                f"{force_val:>9.4f}  {idf_val:>7.4f}  "
                f"{base_p:>9.6f}  {boost_p:>9.6f}  {category:>12}"
            )
        print()

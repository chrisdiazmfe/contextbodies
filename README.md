# contextbodies

A physics-inspired token sampling system for LLMs that adds a semantic-field reweighting term to the next-token distribution. Semantic bodies derived from vocabulary clustering exert gravitational influence on sampling based on cosine distance in embedding space — steering generation toward topically relevant tokens without overriding the model's own probability structure.

---

## Results

Full ablation on GPT-2, 100 tokens, 8 prompts, temperature matched at T=0.8:

| Condition | Perplexity | Distinct-1 | Rep-2 | Notes |
|---|---|---|---|---|
| Temperature T=0.8 | 9.84 | 0.411 | 0.187 | baseline |
| Top-p (p=0.95) | 6.67 | 0.350 | — | low diversity |
| Typical sampling | 11.64 | 0.427 | 0.141 | +4% d1, −24% rep |
| Local bodies only | **13.42** | 0.457 | 0.174 | +11% d1, 43ms/tok |
| Universe (real centroids) | 14.74 | 0.493 | **0.108** | +20% d1, −42% rep, 93ms/tok |
| Universe (shuffled centroids) | 14.76 | 0.494 | — | ≈ real; geometry not mass distribution |
| Universe (random centroids) | 15.83 | **0.505** | 0.116 | +23% d1; pushes diversity over coherence |

Key findings:

- **All gravitational conditions outperform temperature on lexical diversity and repetition reduction.** The mechanism works regardless of centroid geometry.
- **The semantic universe enforces topical coherence, not maximum diversity.** Real centroids achieve lower perplexity than random (14.74 vs 15.83) but also lower distinct-1 (0.493 vs 0.505). Semantic geometry nudges toward contextually plausible rare tokens; random geometry pushes toward arbitrary embedding regions.
- **Real ≈ shuffled.** Same centroid positions, permuted mass assignments produce nearly identical results. The geometry of centroid locations is what matters, not which cluster gets which mass.
- **Top-k body selection is required for geometry to matter.** With all 256 bodies active simultaneously, the force field is nearly uniform across the vocabulary (force max/mean ≈ 3.1) and real, shuffled, and random conditions are indistinguishable. Restricting force to the top-k=4 most contextually aligned bodies concentrates force selectively and makes semantic geometry meaningful.
- **Local bodies mode is Pareto-optimal** on perplexity (13.42) at 2× lower latency (43ms vs 93ms/tok).
- **Temperature matching is mandatory.** Mismatched temperature (baseline T=0.8 vs gravitational T=1.0) inflates perplexity by ~4× and makes the comparison meaningless.

---

## Installation

```bash
git clone https://github.com/chrisdiazmfe/contextbodies
cd contextbodies
pip install torch numpy faiss-cpu scikit-learn pytest
```

For persistent cross-session body storage, also install Qdrant:

```bash
pip install qdrant-client
```

`scikit-learn` is only used when building a universe (`--build-universe`). All other features work without it.

---

## Quickstart

```python
from contextbodies import GravitationalSampler, ContextBodyStore
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

store = ContextBodyStore(embedding_dim=768)

sampler = GravitationalSampler(
    body_store=store,
    G=1.0,                  # local context body gravitational constant
    G_universe=1.0,         # universe field gravitational constant (always static)
    escape_threshold=0.01,  # minimum force to influence sampling
)

# Optional: build a universe from the model's vocabulary
from contextbodies import UniverseBuilder
universe = UniverseBuilder().build(model, n_clusters=256, mass_scheme="idf")
sampler.universe = universe

# Generate
from contextbodies.generate import generate
text = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="Tell me about transformers",
    sampler=sampler,
    max_tokens=200,
)
print(text)
```

---

## HuggingFace Integration (SemanticForceProcessor)

`SemanticForceProcessor` wraps the gravitational field as a standard `LogitsProcessor` for use with `model.generate()`. No custom generation loop required.

```python
from contextbodies import SemanticForceProcessor, UniverseBuilder
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Build universe (one-time; save/load with universe.save() / Universe.load())
universe = UniverseBuilder().build(model, n_clusters=256, mass_scheme="idf")

processor = SemanticForceProcessor(
    universe=universe,
    G_universe=1.0,
    universe_top_k=4,   # restrict force to top-k contextually aligned bodies
)

inputs = tokenizer("Tell me about transformers", return_tensors="pt")
outputs = model.generate(
    **inputs,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    logits_processor=[processor],
    renormalize_logits=True,
    max_new_tokens=200,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

The logit-space update is `scores += log1p(force_magnitudes)`, which approximates multiplicative reweighting in probability space. Setting `renormalize_logits=True` in `generate()` is recommended.

To compare field-on vs field-off with identical seeds:

```python
# Baseline
torch.manual_seed(42)
out_base = model.generate(**inputs, do_sample=True, temperature=0.8, top_p=0.9, max_new_tokens=100)

# With semantic field
torch.manual_seed(42)
out_grav = model.generate(**inputs, do_sample=True, temperature=0.8, top_p=0.9,
                           logits_processor=[processor], renormalize_logits=True, max_new_tokens=100)
```

---

## Benchmarking

`benchmark.py` compares gravitational sampling against a temperature baseline across a set of prompts.

### Temperature baseline only

```bash
python benchmark.py --temperature-only --temperature 0.8 --output results/temp-baseline
```

### Gravitational with local context bodies (DBSCAN)

Fixed G:

```bash
python benchmark.py --local-bodies --G-local 1.0 --output results/local-bodies
```

With AdaptiveG (`--G-local` and `--adaptive-g` are mutually exclusive — use `--G-base` as the AdaptiveG anchor):

```bash
python benchmark.py --local-bodies --G-base 1.0 --adaptive-g --output results/local-bodies-adaptive
```

### Universe field only (no DBSCAN)

Build a universe once and reuse it:

```bash
# Build and run
python benchmark.py --build-universe --universe-clusters 256 --universe-mass idf \
    --G-universe 1.0 --output results/universe-baseline

# Subsequent runs — load the cached universe
python benchmark.py --universe-path results/universe-baseline/universe.npz \
    --G-universe 1.0 --output results/universe-run2
```

### Universe + local context bodies

Fixed G:

```bash
python benchmark.py --universe-path results/universe-baseline/universe.npz \
    --local-bodies --G-local 1.0 --G-universe 1.0 --output results/universe-plus-local
```

With AdaptiveG on local bodies (universe G stays static):

```bash
python benchmark.py --universe-path results/universe-baseline/universe.npz \
    --local-bodies --G-base 1.0 --adaptive-g --G-universe 1.0 --output results/universe-plus-local-adaptive
```

### Deterministic sampling

When using a universe, sampling can be made fully deterministic (argmax instead of multinomial). Diversity comes from universe geometry — the same prompt always produces the same output.

```bash
python benchmark.py --universe-path results/universe-baseline/universe.npz \
    --deterministic --output results/deterministic
```

### Ablation mode

Run the full ablation battery (real, shuffled, random centroids, no-IDF, uniform mass, local-only, combined, and baselines) in one pass:

```bash
python benchmark.py --build-universe --ablation --universe-top-k 4 \
    --temperature 0.8 --output results/ablation
```

### All benchmark flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `gpt2` | HuggingFace model name |
| `--max-tokens` | `100` | Max tokens per prompt |
| `--runs` | `2` | Runs per prompt (averaged) |
| `--temperature` | `0.8` | Temperature for all samplers (baseline and gravitational — must match for fair comparison) |
| `--temperature-only` | off | Run only the temperature baseline |
| `--ablation` | off | Run full ablation battery across all geometry/mass/baseline conditions |
| `--universe-top-k` | `16` | Top-k contextually aligned universe bodies to apply force from. k=4 recommended — with all 256 bodies active the force field is nearly uniform and conditions are indistinguishable |
| `--G-local` | `1.0` | Fixed gravitational constant for local context bodies. Mutually exclusive with `--adaptive-g` |
| `--G-universe` | `1.0` | Gravitational constant for the universe background field. Always static — never adjusted by AdaptiveG |
| `--G-base` | `1.0` | Starting G for AdaptiveG. Only relevant when `--adaptive-g` is set; ignored otherwise |
| `--escape-threshold` | `0.01` | Minimum force magnitude to count as bound |
| `--adaptive-g` | off | Enable AdaptiveG controller for local bodies. Mutually exclusive with `--G-local` |
| `--escape-rate-target` | `0.7` | AdaptiveG target fraction of unbound tokens |
| `--mass-damping` | `1.0` | How aggressively body mass growth reduces G. 1.0=full, 0.5=square-root, 0.0=off |
| `--build-universe` | off | Build universe from model vocabulary before running |
| `--universe-path` | — | Load a pre-built universe `.npz` file |
| `--universe-clusters` | `256` | Number of k-means clusters for universe building |
| `--universe-mass` | `idf` | Universe body mass scheme: `uniform`, `size`, or `idf` |
| `--local-bodies` | off | Enable local context body layer (prompt-specific gravity via clustering) |
| `--deterministic` | off | Use argmax instead of multinomial sampling |
| `--cluster-radius` | `0.3` | Cosine distance radius for context body clustering (ignored with `--adaptive-clustering`) |
| `--cluster-min-tokens` | `3` | Minimum tokens required to form a context body cluster core |
| `--no-idf` | off | Disable IDF weighting |
| `--recency-decay` | `0.0` | Rate at which persisted body force fades over time (0 = off) |
| `--output` | `benchmark_results` | Output directory |

Outputs saved to `<output>/`: `results.json`, `steps.json`, `summary.txt`, and (if built) `universe.npz`.

---

## Concept

Standard LLM sampling uses **temperature** to flatten or sharpen a probability distribution uniformly across all tokens. `contextbodies` replaces this with **context gravity**: semantic bodies that exert gravitational influence on sampling based on cosine distance in embedding space.

### The gravitational force formula

$$F = G \frac{m_{token} \cdot m_{body}}{r^2}$$

- **G** — gravitational constant (replaces temperature as the primary tuning knob)
- **m_token** — token mass, derived from model weight norms: `m = ‖W[token_id]‖ / G`
- **m_body** — body mass, derived from cluster density × weight norms
- **r** — cosine distance between the token embedding and the body centroid

### Multiplicative reweighting

Gravity does not add a flat bias to logits. Instead it multiplies the model's own probability distribution:

```
probs = softmax(logits)
probs = probs × (1 + force_magnitudes)   # amplify tokens near bodies
probs = probs / probs.sum()              # renormalize
```

Tokens the model assigns near-zero probability stay near-zero regardless of gravitational pull. The model's inherent diversity structure is preserved — gravity steers rather than overrides.

### IDF weighting

Common tokens (punctuation, articles, EOS) receive low IDF weight. This suppression is applied at two points:

1. **Body formation** (`post_step`) — common tokens accrete less mass onto DBSCAN clusters
2. **Output field** (`sample`) — common tokens receive reduced gravitational amplification even when geometrically close to a body centroid

### The Universe

The **universe** is a pre-computed set of semantic bodies derived from clustering the model's full vocabulary (50k+ tokens) with k-means. Unlike context bodies, which form dynamically from generated tokens, universe bodies are permanent and cover every semantic region expressible by the model.

This solves the diversity collapse problem: context-only bodies form a feedback loop (generated tokens pull toward already-generated semantic regions). Universe bodies break this loop — the gravitational field exists before any tokens are generated and does not change during generation.

Any idea expressible in the model's vocabulary has a location in its universe. Novel combinations are new trajectories through the universe, not new locations.

#### Context-affinity modulation

The universe field is not omnidirectional. Each body's effective mass is weighted by its cosine similarity to the current **orbital position** (the recent context embedding):

```
effective_mass = body_mass × max(0, cosine_similarity(body_centroid, context_position))
```

Bodies semantically close to what is currently being generated exert full force; bodies far away contribute near zero. This makes the universe field context-sensitive — it boosts rare tokens that are relevant to the current topic, not all rare tokens everywhere.

The total field is a sum of both layers:

```
total_field = universe_field + context_field   # context_field is 0 when --local-bodies is off
```

### Context bodies (local layer)

When `--local-bodies` is enabled, an incremental DBSCAN clustering runs over prompt and generated tokens, forming **local context bodies** — prompt-specific gravitational perturbations on top of the universe field. These are useful for tightly domain-specific generation but can cause diversity collapse in long outputs. Disabled by default when using a universe.

### Orbital mechanics

The current context vector has **position**, **velocity**, and **acceleration** in embedding space:

- **Velocity** — direction the context is trending semantically
- **Acceleration** — rate of topic shift
- **Momentum** — resistance to gravitational deflection from new bodies

Tokens can orbit context bodies, transfer between them as topics shift, or achieve escape velocity to produce novel output.

### Context body types

Body type is not assigned manually — it emerges from cluster density and model weight norms over time.

| Body Type | Mass | Behavior |
|---|---|---|
| Black hole | Very high | Dominant, inescapable theme (e.g. system prompt constraints) |
| Neutron star | High | Rare but highly specific, dense context |
| Planet | Moderate | Stable topic with consistent influence |
| Moon | Low | Sub-topic orbiting a parent body |
| Asteroid | Very low | Passing mention, minimal pull |

---

## Key Parameters

| Parameter | Description | Default |
|---|---|---|
| `G` | Gravitational constant for local context bodies. Higher = stronger context pull, less diversity | `1.0` |
| `G_universe` | Gravitational constant for the universe background field. Always static — not adjusted by AdaptiveG | `1.0` |
| `escape_threshold` | Minimum force magnitude to count as gravitationally bound. With IDF weighting, tracks tokens where gravity has negligible effective influence | `0.01` |
| `stability_threshold` | Minimum stability score for an emergent body to be persisted to the store | `0.8` |
| `resonance_threshold` | Minimum resonance score for a body pair to produce a Lagrange midpoint force | `0.3` |
| `body_merge_distance` | Cosine distance below which nearby bodies merge into a virtual body before force computation | `0.2` |
| `collision_distance` | Cosine distance below which two in-session bodies undergo an inelastic collision | `0.1` |
| `recency_decay_lambda` | Rate at which persisted body force fades over time: `exp(-rate × elapsed_seconds)`. `0.0` = disabled. `1e-4` ≈ 2-hour half-life | `0.0` |
| `universe` | Optional `Universe` instance. Provides background gravitational field from vocabulary clustering | `None` |
| `use_context_bodies` | Whether to run DBSCAN clustering on the prompt and generated tokens | `True` |
| `deterministic` | Use `argmax` instead of `multinomial`. Requires a universe | `False` |
| `adaptive_g` | Optional `AdaptiveG` instance. Adjusts G each step via mass normalization + escape rate feedback | `None` |
| `domain` | Static domain label. Ignored when `domain_classifier` is provided | `""` |
| `domain_classifier` | Optional `DomainClassifier` — infers domain from token stream automatically | `None` |

---

## Adaptive G

`AdaptiveG` adjusts the gravitational constant at each step via three multiplicative terms:

```
G_eff = G_base × mass_norm(t) × escape_feedback(t) × domain_scale
```

**Mass normalization** — body mass grows as tokens accrete. Without compensation, older bodies exert disproportionate force. `mass_norm` tracks an EMA of active body mass and scales G inversely, keeping force magnitude stable throughout generation.

**Escape rate feedback** — a PI controller targets a configurable fraction of tokens remaining "unbound" (force below `escape_threshold`) each step. Too few escaping → G is reduced; too many → G is raised.

**Domain multiplier** — static per-domain scaling applied after the feedback terms.

```python
from contextbodies import AdaptiveG

adaptive_g = AdaptiveG(
    G_base=1.0,
    escape_rate_target=0.7,   # 70% of tokens unbound per step
    mass_norm_strength=1.0,   # full inverse normalization
    Kp=0.1,
    Ki=0.01,
)

sampler = GravitationalSampler(body_store=store, G=1.0, adaptive_g=adaptive_g)
```

---

## Body Persistence

Stable emergent bodies are recorded to the `ContextBodyStore` and reused across conversations. This builds a **living knowledge graph** of domain-specific gravitational structure over time:

- Bodies gain mass as related tokens accrete onto them
- Bodies decay in mass when not reinforced by recent context
- Bodies can merge (topic convergence) or fragment (topic divergence)
- Historical bodies seed the gravitational field at the start of new conversations

### Store insertion: dedup and re-emergence

Before inserting a new body, the store runs a three-stage check:

1. **Exact dedup** — if a body already exists within `dedup_distance`, skip insertion and update its mass
2. **Re-emergence** — if a dormant (low-mass) body exists within `reemergence_distance`, re-energize it with a mass-weighted boost rather than creating a duplicate
3. **New record** — otherwise insert

This prevents the same concept from accumulating multiple low-mass ghosts that together double the gravitational influence of a theme.

### Cross-session collision detection

After a local body is persisted to the store, it is checked against all records from prior sessions. If the new body's centroid falls within `collision_distance` of an existing record, they undergo an inelastic collision:

- Centroids are merged by mass-weighted average
- Masses are summed
- Resonance links from the incoming record are transferred to the surviving record
- The incoming record is deleted

This prevents related concepts from accumulating as separate low-mass records across sessions. Over time, repeated themes converge into single, increasingly massive bodies.

---

## Domain Classifier

`DomainClassifier` infers the active domain from the token stream via EMA context direction compared against known domain anchors by cosine distance.

```python
from contextbodies import DomainClassifier, GravitationalSampler, ContextBodyStore

clf = DomainClassifier(
    domain_anchors={
        "code":    embed("python function class method variable"),
        "medical": embed("diagnosis treatment patient clinical"),
        "legal":   embed("contract statute liability jurisdiction"),
    },
    fallback_domain="general",
)

sampler = GravitationalSampler(
    body_store=store,
    G=1.0,
    domain_classifier=clf,   # domain updated automatically each token
)
```

When bodies have already accumulated in the store, build anchors from them automatically:

```python
clf = DomainClassifier.from_body_centroids(
    {d: [r.centroid for r, _ in pairs] for d, pairs in bodies_by_domain.items()},
    match_threshold=0.3,
    ema_alpha=0.1,
)
```

---

## Persistent Store (Qdrant)

By default the store is in-memory (`FAISSBackend`). For cross-session persistence, swap in `QdrantBackend`.

Start Qdrant via Docker:

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

Then connect:

```python
from qdrant_client import QdrantClient
from contextbodies import ContextBodyStore, QdrantBackend

client = QdrantClient(host="localhost", port=6333)
# client = QdrantClient(url="https://your-cluster.qdrant.io", api_key="your-key")
# client = QdrantClient(":memory:")   # in-memory, no Docker

store = ContextBodyStore(
    embedding_dim=768,
    backend=QdrantBackend(client, collection_name="context_bodies"),
)
```

The collection is created automatically on first use.

---

## Architecture

```
contextbodies/
├── context_body.py          # ContextBody — mass, centroid, orbital membership
├── context_body_record.py   # ContextBodyRecord — persistent form stored in vector DB
├── context_body_store.py    # Persistent store — thin wrapper around VectorBackend
├── vector_backend.py        # VectorBackend protocol + FAISSBackend + QdrantBackend
├── domain_classifier.py     # DomainClassifier — infers domain from token stream
├── orbital_state.py         # Position/velocity/acceleration of the context vector
├── incremental_dbscan.py    # Online clustering — discovers emergent bodies token by token
├── adaptive_dbscan.py       # AdaptiveDBSCAN — auto-calibrates cluster radius from embedding geometry
├── adaptive_g.py            # AdaptiveG — mass-normalized PI controller for G
├── universe_builder.py      # Universe + UniverseBuilder — vocabulary-wide semantic field
├── semantic_force_processor.py # SemanticForceProcessor — LogitsProcessor for model.generate()
├── gravitational_sampler.py # Core sampler — custom generation loop with full diagnostics
├── benchmark.py             # Evaluation harness — ablation battery + baseline comparison
├── generate.py              # Drop-in generation loop
└── tests/
    ├── conftest.py
    ├── test_context_body.py
    ├── test_context_body_record.py
    ├── test_context_body_store.py
    ├── test_vector_backend.py
    ├── test_orbital_state.py
    ├── test_domain_classifier.py
    ├── test_adaptive_g.py
    ├── test_dbscan_new_features.py
    ├── test_fragmentation.py
    └── test_gravitational_sampler.py
```

---

## Testing

```bash
cd contextbodies
pytest tests/ -v
```

Run a specific file:

```bash
pytest tests/test_gravitational_sampler.py -v
```

Run with output for debugging:

```bash
pytest tests/ -v -s
```

190 tests pass. `test_gravitational_sampler.py` requires `torch` and auto-skips when it is not installed.

| File | Coverage |
|---|---|
| `test_context_body.py` | Mass accumulation, centroid running mean, orbital radii, `classify()` tiers |
| `test_context_body_record.py` | `to_metadata`/`from_metadata` round-trip, resonance serialization, malformed-field handling |
| `test_context_body_store.py` | Three-stage `record()` (dedup, re-emergence, new), gravitational ranking, decay, resonance |
| `test_vector_backend.py` | `FAISSBackend` upsert/search/delete/update_metadata, domain filter, 4-tuple return |
| `test_orbital_state.py` | Initialize, velocity/acceleration, `momentum`, `speed_of_change` |
| `test_domain_classifier.py` | Seed, EMA update, threshold fallback, add/remove anchor, `from_body_centroids()` |
| `test_adaptive_g.py` | Mass normalization anchoring, PI escape-rate feedback, G_min/G_max clamping, domain multipliers |
| `test_dbscan_new_features.py` | Token mass accumulation, 4-tuple return, collision events, border-point bridge preservation |
| `test_fragmentation.py` | Power iteration, bimodality detection, split correctness, connected components, full pipeline |
| `test_gravitational_sampler.py` | Force formula, recency factor, body grouping, resonance, inelastic collisions, `sample()` *(requires torch)* |

---

## Status

Early research implementation.

### Completed

- **Multiplicative reweighting** — gravity amplifies the model's own distribution (`probs × (1 + force)`) rather than adding a flat logit bias. Tokens the model assigns near-zero probability stay near-zero regardless of gravitational pull. Logit-space equivalent: `scores += log1p(force)`.
- **SemanticForceProcessor** — `LogitsProcessor` implementation for `model.generate()`. Wraps the universe field with no custom generation loop required. Supports `universe_top_k` for force concentration.
- **Top-k body selection** — restricts force computation to the k bodies most aligned with the current context position. Required for semantic geometry to produce meaningful force differentials; without it, all 256 bodies contribute nearly uniformly and real/shuffled/random conditions are indistinguishable.
- **Full ablation battery** — `--ablation` flag runs all geometry/mass/IDF/baseline conditions in one pass: real, shuffled, random centroids, no-IDF, uniform mass, local-only, universe+local, temperature, top-p, typical sampling.
- **Diagnostic logging** — per-step force mean/max, active body count, top boosted tokens, escape rate, and before/after token rank logged during generation.
- **Universe** — pre-computed k-means clustering of all vocabulary embeddings provides a permanent background gravitational field, breaking the diversity feedback loop of context-only bodies. Three mass schemes: `uniform`, `size` (cluster density), `idf` (mean IDF weight of member tokens). Saves/loads as `.npz`. Reports `semantic_coverage` (fraction of universe bodies visited) as a new diversity metric.
- **AdaptiveG** — PI controller that adjusts G each step via mass normalization + escape rate feedback + domain multipliers.
- **Dual IDF** — IDF weighting applied at both body formation (`post_step`) and the output field (`sample`). Common tokens suppressed at ingestion and at sampling.
- **Deterministic sampling** — `argmax` mode when diversity is provided by universe geometry rather than stochastic draws.
- **Body mass from weight norms** — `m = ‖W[token_id]‖ / G`.
- **Orbital resonance** — Lagrange midpoint force for co-active resonant body pairs.
- **Domain classifier** — EMA-based inference of active domain from the token stream.
- **Gravitational amplification** — nearby bodies merge into virtual bodies before force computation.
- **Inelastic collisions** — bodies within `collision_distance` merge with momentum conservation.
- **Context-affinity modulation** — universe body forces weighted by cosine similarity to the current orbital position, making the universe field context-sensitive rather than omnidirectional.
- **G split** — separate `G_local` (for context bodies, adaptive or fixed) and `G_universe` (always static) with CLI mutual exclusion between `--G-local` and `--adaptive-g`.
- **Cross-session collision detection** — newly persisted local bodies checked against prior-session store records; overlapping bodies merge via inelastic collision (mass-weighted centroid, summed mass, transferred resonance links).
- **GPU force computation** — local body force loop ported to a single batched torch matmul; 5× speedup on GPT-2 Medium (637ms → 124ms/token).
- **Recency weighting** — force from persisted bodies decays with `exp(-λ × elapsed)`.
- **QdrantBackend** — production-grade persistent store.
- **Vector-native architecture** — `ContextBodyStore` is a thin wrapper around a `VectorBackend` protocol; swap FAISS for Qdrant, Pinecone, or pgvector at construction.

### Planned

- `PineconeBackend`, `PgvectorBackend`
- MAUVE and KL-divergence metrics in the benchmark harness

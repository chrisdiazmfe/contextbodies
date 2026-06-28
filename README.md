# contextbodies

A physics-inspired token sampling system for LLMs that replaces temperature-based sampling with a gravitational field model.

## Concept

Standard LLM sampling uses **temperature** to flatten or sharpen a probability distribution uniformly across all tokens. `contextbodies` replaces this with **context gravity** — semantic clusters that emerge from the token stream and exert gravitational influence on sampling, analogous to celestial bodies in a gravitational field.

### Context Bodies

As tokens are generated, they cluster in embedding space. Dense clusters become **context bodies** — gravitational objects that pull candidate tokens toward semantically related output. The mass of a body determines its influence:

| Body Type | Mass | Behavior |
|---|---|---|
| Black hole | Very high | Dominant, inescapable theme (e.g. system prompt constraints) |
| Neutron star | High | Rare but highly specific, dense context |
| Planet | Moderate | Stable topic with consistent influence |
| Moon | Low | Sub-topic orbiting a parent body |
| Asteroid | Very low | Passing mention, minimal pull |

Body type is not assigned manually — it **emerges** from cluster density and model weight norms over time.

### Gravitational Force

The force a context body exerts on a candidate token follows Newton's law of gravitation:

$$F = G \frac{m_{token} \cdot m_{body}}{r^2}$$

Where:
- **G** — gravitational constant (tunable, replaces temperature)
- **m_token** — token mass, derived from model weight norms: `m = W / G`
- **m_body** — body mass, derived from cluster density × weight norms
- **r** — cosine distance between the token embedding and the body centroid

### Orbital Mechanics

The current context vector has **position**, **velocity**, and **acceleration** in embedding space:

- **Velocity** — direction the context is trending semantically
- **Acceleration** — rate of topic shift
- **Momentum** — resistance to gravitational deflection from new bodies

Tokens can orbit context bodies, transfer between them as topics shift, or achieve escape velocity to produce novel output.

## Usage

```python
from contextbodies import GravitationalSampler, ContextBodyStore
from contextbodies.generate import generate
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

# In-memory (development) — no external dependencies
store = ContextBodyStore(embedding_dim=768)

# Persistent (production) — swap in Qdrant with one line
# from contextbodies import QdrantBackend
# from qdrant_client import QdrantClient
# store = ContextBodyStore(embedding_dim=768, backend=QdrantBackend(
#     QdrantClient(host="localhost", port=6333)
# ))

sampler = GravitationalSampler(
    body_store=store,
    G=1.0,                  # gravitational constant — primary tuning knob
    escape_threshold=0.01,  # minimum force to influence sampling
    domain="ml",            # domain for body persistence and retrieval
)

text = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="Tell me about transformers",
    sampler=sampler,
    max_tokens=200,
)
print(text)
```

### With automatic domain inference

When bodies have already accumulated in the store across sessions, use `DomainClassifier.from_body_centroids()` to build anchors from them. The sampler then infers and tracks domain automatically from the token stream — no hard-coded domain string needed.

```python
from contextbodies import DomainClassifier, GravitationalSampler, ContextBodyStore

store = ContextBodyStore(embedding_dim=768, backend=QdrantBackend(
    QdrantClient(host="localhost", port=6333)
))

# build domain anchors from bodies already in the store
bodies_by_domain = {
    "code":    [b.centroid for b in store.query_nearby(code_anchor, domain="code", k=50)],
    "medical": [b.centroid for b in store.query_nearby(medical_anchor, domain="medical", k=50)],
}
clf = DomainClassifier.from_body_centroids(
    {d: [r.centroid for r, _ in pairs] for d, pairs in bodies_by_domain.items()},
    match_threshold=0.3,   # max cosine distance to accept a domain match
    ema_alpha=0.1,         # how quickly domain shifts mid-conversation
)

sampler = GravitationalSampler(
    body_store=store,
    G=1.0,
    domain_classifier=clf,  # domain inferred from token stream; "domain" param ignored
)

text = generate(model=model, tokenizer=tokenizer,
                prompt="Write a Python function to parse JSON",
                sampler=sampler, max_tokens=200)
```

You can also register anchors manually if you have representative embeddings for each domain:

```python
clf = DomainClassifier(
    domain_anchors={
        "code":    embed("python function class method variable"),
        "medical": embed("diagnosis treatment patient clinical"),
        "legal":   embed("contract statute liability jurisdiction"),
    },
    fallback_domain="general",
)
```

## Key Parameters

| Parameter | Description | Default |
|---|---|---|
| `G` | Gravitational constant. Higher = stronger context pull, less diversity | `1.0` |
| `escape_threshold` | Minimum force magnitude to bias sampling. Tokens below this are unaffected | `0.01` |
| `stability_threshold` | Minimum stability score for an emergent body to be persisted | `0.8` |
| `resonance_threshold` | Minimum resonance score for a body pair to produce a Lagrange midpoint force | `0.3` |
| `amplification_threshold` | Cosine distance below which nearby bodies are merged into a virtual body for force computation, preventing gravity well amplification | `0.2` |
| `collision_distance` | Cosine distance below which two in-session bodies undergo an inelastic collision — lighter absorbed by heavier with momentum conservation | `0.1` |
| `domain` | Static domain label. Ignored when `domain_classifier` is provided | `""` |
| `domain_classifier` | Optional `DomainClassifier` — infers domain from the token stream automatically | `None` |

## Body Persistence

Stable emergent bodies are recorded to the `ContextBodyStore` and reused across conversations. This builds a **living knowledge graph** of domain-specific gravitational structure over time:

- Bodies gain mass as related tokens accrete onto them
- Bodies decay in mass when not reinforced by recent context
- Bodies can merge (topic convergence) or fragment (topic divergence)
- Historical bodies seed the gravitational field at the start of new conversations

## Architecture

```
contextbodies/
├── context_body.py          # ContextBody dataclass — mass, centroid, orbital membership
├── context_body_record.py   # ContextBodyRecord — persistent form stored in vector DB
├── context_body_store.py    # Persistent store — thin wrapper around VectorBackend
├── vector_backend.py        # VectorBackend protocol + FAISSBackend + QdrantBackend
├── domain_classifier.py     # DomainClassifier — infers domain from token stream
├── orbital_state.py         # Position/velocity/acceleration of the context vector
├── incremental_dbscan.py    # Online clustering — discovers emergent bodies token by token
├── gravitational_sampler.py # Core sampler — replaces temperature at inference time
├── generate.py              # Drop-in generation loop
└── tests/
    ├── conftest.py
    └── test_fragmentation.py
```

## Dependencies

```
torch
numpy
faiss-cpu      # or faiss-gpu
pytest         # for running tests
qdrant-client  # optional, only needed for QdrantBackend
```

Install core dependencies:

```bash
pip install torch numpy faiss-cpu pytest
```

Install with Qdrant support:

```bash
pip install torch numpy faiss-cpu pytest qdrant-client
```

## Qdrant Setup

`QdrantBackend` requires a running Qdrant instance. The fastest way to get one is via Docker:

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

This starts Qdrant on port `6333` (HTTP/gRPC) with data persisted to `./qdrant_storage`.

Then connect in Python:

```python
from qdrant_client import QdrantClient
from contextbodies import ContextBodyStore, QdrantBackend

# Local server
client = QdrantClient(host="localhost", port=6333)

# Qdrant Cloud (get url and api_key from cloud.qdrant.io)
# client = QdrantClient(url="https://your-cluster.qdrant.io", api_key="your-key")

# In-memory — useful for testing, no Docker required
# client = QdrantClient(":memory:")

store = ContextBodyStore(
    embedding_dim=768,
    backend=QdrantBackend(client, collection_name="context_bodies"),
)
```

The collection is created automatically on first use. Bodies written to Qdrant persist across sessions — the gravitational field accumulates knowledge over time.

## Testing

Tests cover fragmentation and bimodality detection across all internal methods and the full `update()` pipeline.

```bash
cd contextbodies
pytest tests/ -v
```

Run a specific test class:

```bash
pytest tests/test_fragmentation.py::TestCheckBimodality -v
```

Run with output for debugging:

```bash
pytest tests/ -v -s
```

### What's tested

| Class | What it covers |
|---|---|
| `TestFirstPrincipalComponent` | Power iteration correctness, unit norm, determinism |
| `TestHasValley` | Clear bimodal, unimodal, uniform, strict threshold, flat histogram |
| `TestCheckBimodality` | Bimodal detected, unimodal rejected, size gate, elongation gate |
| `TestSplitBimodal` | Two clusters produced, original removed, no overlap, lineage, abort guard |
| `TestConnectedComponents` | Disconnected groups, full partition, tight cluster, border-only cluster |
| `TestFragmentCluster` | Cluster count, label removal, point relabeling, centroid history inherited |
| `TestMaybeFragment` | Stable skip, missing label, connectivity priority, bimodality fallback |
| `TestUpdateIntegration` | New body events, merge events, fragmentation events, noise, single cluster |

## Status

Early research implementation. Open issues are grouped below by area.

### Storage & Persistence
- ✅ Vector-native architecture — `ContextBodyStore` refactored to a thin wrapper around a `VectorBackend` protocol. `FAISSBackend` is the default in-memory implementation. Swap to Qdrant, Pinecone, or pgvector by passing a different backend at construction.
- ✅ `ContextBodyRecord` — lightweight persistent record (centroid + scalar metadata only). No relational fields. Serializes to/from flat vector DB payloads via `to_metadata()` / `from_metadata()`.
- ✅ Decay scheduler — per-query trigger in `query_nearby()`. Decay runs automatically when `decay_interval` seconds have elapsed since the last run (default 60s). No background process required.
- ✅ `QdrantBackend` — production backend backed by Qdrant. Accepts a pre-configured `QdrantClient` (local, cloud, or in-memory). Returns stored vectors in search results, which eliminates the `_fetch_centroid` approximation. Install with `pip install qdrant-client`.
- ✅ `_fetch_centroid` workaround resolved for Qdrant — `VectorBackend.search()` now returns a fourth element `np.ndarray | None`; `ContextBodyStore.query_nearby()` uses the actual centroid when available and falls back to the approximation only for `FAISSBackend`.
- `PineconeBackend`, `PgvectorBackend` — not yet implemented

### Physics Model
- ✅ Body mass refinement — body mass is now the sum of constituent token masses (`Σ ||W[token_id]|| / G`). `IncrementalDBSCAN.update()` accepts `token_mass`, stores it per point, and `_build_body()` sums them. `GravitationalSampler.sample()` computes mass from weight norms and passes it through. Prompt tokens default to `token_mass=1.0` as an approximation.
- ✅ Orbital resonance detection — `ContextBodyRecord.resonance_partners` accumulates co-occurrence scores across sessions via `ContextBodyStore.record_resonance()`. When two resonant bodies are co-active, `GravitationalSampler._compute_resonance_forces()` adds a Lagrange midpoint force toward the semantic region between them, scaled by `sqrt(m_A * m_B) * score`. Resonance is recorded automatically when a new body is persisted alongside existing store-loaded bodies. Tunable via `resonance_threshold` (default 0.3).
- ✅ Domain classifier — `DomainClassifier` infers the active domain from the token embedding stream via an EMA context direction compared against known domain anchors by cosine distance. Seeded from the full prompt in `initialize()`, updated on every generated token in `sample()`. Anchors can be pre-defined embeddings, derived from body centroids via `from_body_centroids()`, or added at runtime via `add_anchor()`. Pass as `domain_classifier=` to `GravitationalSampler`; `domain` is then updated automatically each step.

### Collision Mechanics
- ✅ Gravitational amplification — `GravitationalSampler._group_active_bodies()` merges bodies within `amplification_threshold` (default 0.2 cosine distance) into virtual bodies with mass-weighted centroids and summed masses before force computation. Three bodies near "machine learning" produce the same force as one body of their combined mass, not 3×. Tunable via `amplification_threshold`.
- ✅ Inelastic collision rule — `GravitationalSampler._check_collisions()` scans active `ContextBody` pairs after each DBSCAN update. When two bodies are within `collision_distance` (default 0.1 cosine), the lighter is absorbed by the heavier with momentum conservation: merged centroid and velocity are mass-weighted averages, merged mass is the sum. The merged body replaces both originals in `active_bodies` and is persisted if stable. Tunable via `collision_distance`.
- Cross-session collision — a decayed body and a new emergent body representing the same concept can coexist as separate records if their distance exceeds `dedup_distance`, doubling the gravitational influence of that concept; needs a broader re-emergence detection pass
- Recency weighting — `last_seen` currently only drives extinction; gravitational force should be scaled by `exp(-λ * time_since_last_seen)` so temporally distant bodies exert proportionally less pull regardless of stored mass
- Collision events — `IncrementalDBSCAN.update()` returns `new_bodies`, `merged_events`, and `fragmented_events` but no `collision_events`; converging bodies that don't fully merge are a distinct physical event worth surfacing to `GravitationalSampler`

### Clustering
- Border-point bridge case — connectivity fragmentation BFS only walks core points; a cluster bridged solely through border points will not fragment via the structural check (bimodality may still catch it)

### Evaluation
- Benchmarking — no evaluation harness against temperature / top-p baselines; suggested metrics: perplexity, MAUVE, distinct-n, KL divergence from baseline distribution

### Testing
- `GravitationalSampler` has no tests — force computation, escape threshold, orbital state updates, and active body sync all need coverage
- `ContextBodyStore` has no tests — `record()` deduplication, `query_nearby()` ranking, and `decay()` extinction logic are untested
- `VectorBackend` / `FAISSBackend` have no tests — upsert, search, delete, and update_metadata need coverage
- `OrbitalState` has no tests — position, velocity, and acceleration update logic is untested

### Documentation
- White paper — formal writeup of the methodology, theoretical grounding, and thesis; should cover the physics analogy, the gravitational sampling formula, emergent body detection, orbital mechanics, and comparison to existing sampling strategies

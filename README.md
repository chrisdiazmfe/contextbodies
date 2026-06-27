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

store = ContextBodyStore(embedding_dim=768)
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

## Key Parameters

| Parameter | Description | Default |
|---|---|---|
| `G` | Gravitational constant. Higher = stronger context pull, less diversity | `1.0` |
| `escape_threshold` | Minimum force magnitude to bias sampling. Tokens below this are unaffected | `0.01` |
| `stability_threshold` | Minimum stability score for an emergent body to be persisted | `0.8` |
| `domain` | Domain label for body storage and retrieval | `""` |

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
├── context_body_store.py    # Persistent store — FAISS hot layer + cold vector DB
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
faiss-cpu  # or faiss-gpu
pytest     # for running tests
```

Install all at once:

```bash
pip install torch numpy faiss-cpu pytest
```

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
- Production backend implementations — `QdrantBackend`, `PineconeBackend`, `PgvectorBackend` conforming to the `VectorBackend` protocol
- `_fetch_centroid` workaround — `FAISSBackend` does not expose stored vectors in search results; production backends (Qdrant, Pinecone) return vectors directly and should eliminate this approximation

### Physics Model
- Body mass refinement — mass currently equals cluster density only; model weight norms are not yet factored in despite being defined in the formula (`m = W / G`)
- Orbital resonance detection — co-present bodies that periodically reinforce each other's influence are not detected or exploited
- Domain classifier — domain is passed manually at construction time; no mechanism exists to infer it from the token stream

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

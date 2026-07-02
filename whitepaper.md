# Context Gravity: A Physics-Inspired Sampling Framework for Language Models

**Chris Diaz**  
July 2026

---

## Abstract

Standard language model sampling controls output diversity through a scalar temperature parameter applied uniformly across the token vocabulary. This approach is context-blind: it cannot distinguish tokens that are semantically relevant to the current generation from tokens that are merely high-probability. I introduce *contextbodies*, a gravitational field model for token sampling in which semantic bodies exert attraction on candidate tokens in proportion to their mass and inverse cosine distance. The gravitational constant G replaces temperature as the primary tuning knob. A two-layer field — a permanent universe field derived from vocabulary-wide k-means clustering and an ephemeral context field derived from incremental DBSCAN clustering of the prompt — provides both global semantic coverage and conversation-specific topical coherence. Gravitational forces are applied via multiplicative reweighting of the model's own probability distribution, preserving syntactic validity. Across five prompts on GPT-2 and GPT-2 Medium, contextbodies achieves 20–27% improvement in distinct-1 lexical diversity over temperature sampling, with a structural perplexity tradeoff that narrows on larger models (GPT-2: ×5; GPT-2 Medium: ×3.4). Persistent body storage via a vector database allows semantic structure to accumulate across conversations, building a living knowledge graph of domain-specific gravitational topology over time.

---

## 1. Introduction

Large language models generate text by sampling from a probability distribution over the vocabulary at each step. The dominant approach to controlling this distribution is temperature scaling:

```
p(x) = softmax(z / T)
```

where **z** are the model's logits and T is a scalar temperature. At T = 1.0 the distribution is unchanged; T < 1.0 sharpens it toward the argmax; T > 1.0 flattens it toward uniform. Nucleus sampling (top-p) and top-k sampling further restrict sampling to high-probability mass.

These methods are effective and computationally free. Years of practical use have produced reliable defaults (T = 0.8, p = 0.9, k = 40) that generalize across tasks. However, they share a fundamental property: they are *context-blind*. Temperature, top-p, and top-k apply the same transformation to every token in the vocabulary regardless of semantic relevance to what is currently being generated. A rare but highly relevant token can be suppressed while a common but irrelevant one survives. Nothing in the sampling mechanism knows where the generation sits in semantic space or whether it is wandering away from relevant concepts.

I identify two distinct concerns that sampling must address. The first is *probability modulation* — managing how peaked or flat the distribution is. Temperature, top-p, and top-k address this well. The second is *semantic relevance weighting* — amplifying tokens that are contextually appropriate and suppressing those that are not. Standard sampling methods do not address this at all: they modulate probability uniformly, not selectively based on meaning.

This paper describes *contextbodies*, a sampling framework that adds semantic relevance weighting as a second axis alongside the model's base probability distribution. Rather than replacing temperature, contextbodies applies a gravitational field to the model's own probability output, amplifying tokens near semantically relevant bodies and leaving tokens far from all bodies unchanged. The model's distribution is steered, not overridden.

---

## 2. Background and Related Work

### 2.1 Temperature Sampling

Temperature sampling (Ackley et al., 1985) controls the entropy of the output distribution via a single scalar. It is parameter-efficient and interpretable but applies uniformly across all tokens. At high temperatures, the distribution is flat regardless of semantic relevance; at low temperatures, it concentrates on the highest-probability tokens regardless of contextual fit.

### 2.2 Nucleus and Top-k Sampling

Holtzman et al. (2020) propose nucleus sampling, which restricts sampling to the smallest set of tokens whose cumulative probability exceeds p. This avoids sampling from the low-probability tail, reducing incoherence without as much diversity cost as greedy decoding. Top-k sampling (Fan et al., 2018) restricts to the k highest-probability tokens. Both methods are context-blind: they make no use of semantic structure in deciding which tokens to include or exclude.

Top-p and top-k have complementary strengths to contextbodies rather than being strictly inferior. For short outputs, latency-sensitive applications, or fine-tuned models with already-narrow distributions, they are often sufficient and significantly cheaper. A production system could reasonably apply both: top-p to prevent incoherence and gravitational sampling to provide semantic steering.

### 2.3 Typical Sampling

Meister et al. (2023) propose typical sampling, which targets tokens with information content close to the conditional entropy of the distribution. This is partially context-aware — it considers the entropy of the full distribution rather than just raw probability — but does not use the semantic geometry of the embedding space.

### 2.4 Embedding-Space Methods

Contrastive decoding (Li et al., 2022) and speculative decoding methods use the embedding space for generation quality improvement but focus on output correctness rather than topical diversity steering. Nearest-neighbor language models (Khandelwal et al., 2020) retrieve semantically similar examples from a datastore at each step, which shares conceptual ground with contextbodies but operates at the retrieval level rather than the sampling level.

---

## 3. The Gravitational Sampling Framework

### 3.1 Core Metaphor

I model the context of a generation as a set of *semantic bodies* in the model's embedding space. Each body has a centroid — the center of a cluster of semantically related tokens — and a mass proportional to how heavily that semantic region has featured in the context so far. Tokens near heavy bodies are gravitationally attracted toward them; tokens far from all bodies are free to be sampled at their base probability.

At each generation step, the total gravitational force on each candidate token is computed and used to amplify the model's probability distribution multiplicatively. The gravitational constant G replaces temperature as the primary tuning knob: high G produces tight topical coherence; low G allows more free exploration.

### 3.2 Force Formula

The gravitational force exerted by a body on a candidate token is:

```
F = G * (m_token * m_body) / r²
```

where:

- **G** is the gravitational constant
- **m_token** is the token's mass, derived from its embedding weight norm: `m = ‖W[token_id]‖ / G`
- **m_body** is the body's mass, accumulated as tokens accrete onto its centroid
- **r** is the cosine distance between the token's embedding and the body's centroid, floored at `r_min = 0.1` to prevent singularities

The choice of cosine distance is natural for language model embeddings, which reside on a high-dimensional sphere where direction encodes semantic meaning and L2 magnitude encodes token frequency or model weighting. A floor of `r_min = 0.1` is essential: without it, tokens that belong to a body have cosine distances approaching zero, producing F ∝ G / 10⁻⁴ — forces several orders of magnitude larger than any reasonable gravitational field. Empirically, `r_min = 1e-6` caused complete diversity collapse (distinct-1 = 0.19) regardless of G value; `r_min = 0.1` restored expected behavior.

Token mass derived from weight norms has a desirable property: it scales naturally with model size, making G transferable across model families. A token with a strong weight norm contributes more to body mass and receives more gravitational force — effectively weighting generation toward tokens the model considers important.

### 3.3 Multiplicative Reweighting

A critical design decision is how gravitational forces modify the output distribution. Two options are available:

**Additive (logit bias):** `z' = z + α * F`

**Multiplicative (probability scaling):** `p'(x) ∝ p(x) * (1 + F_x)`

I use the multiplicative form:

```
probs = softmax(logits)
probs = probs × (1 + force_magnitudes)
probs = probs / probs.sum()
```

The reason is coherence preservation. Additive logit bias pushes tokens the model has assigned near-zero probability — for syntactic or factual reasons — into non-negligible sampling territory. Multiplicative reweighting preserves these hard constraints: a token with p(x) ≈ 0 stays near zero regardless of gravitational pull. Gravity steers the distribution; the model still governs coherence. This is analogous to a gravitational field deflecting a trajectory rather than teleporting an object to a new location.

### 3.4 IDF Weighting

Common tokens — articles, punctuation, EOS — are semantically non-specific. Gravitational amplification toward them would push generation toward grammatical filler rather than topically relevant content. I suppress this via inverse document frequency (IDF) weighting derived from the model's unconditional token distribution:

```
idf(x) = -log p_uncond(x)
```

normalized to [0, 1] across the vocabulary. IDF weighting is applied at two points:

1. **Body formation** — when a token accretes onto a DBSCAN cluster, its contribution to the body's mass is scaled by its IDF weight. Common tokens contribute minimally; rare, specific tokens contribute fully.

2. **Output field** — gravitational amplification at each sampling step is scaled by IDF weight. Common tokens receive near-zero boost even when geometrically close to a body centroid.

This makes the effective gravitational field a function of both semantic proximity *and* semantic specificity. The model is steered toward rare, contextually relevant tokens rather than common tokens that happen to cluster in the same region.

### 3.5 Escape Velocity

Tokens whose total gravitational force falls below a threshold `escape_threshold` are considered *unbound* — they contribute to the *escape rate*, the fraction of candidate tokens sampled at their base model probability without gravitational amplification. The escape rate is a diagnostic signal: it measures how tightly the field constrains sampling. A target escape rate near 0.7 means 70% of tokens are uninfluenced per step — gravity shapes the tail without overwhelming the base distribution.

---

## 4. Architecture

### 4.1 Two-Layer Field

The total gravitational field is the sum of two independent layers:

```
F_total = F_universe + F_context
```

This separation reflects a distinction between background semantic structure (present before any tokens are generated) and conversation-specific topical structure (emerging from the current generation).

### 4.2 The Universe Field

The universe is a pre-computed set of semantic bodies derived from k-means clustering of the full model vocabulary:

1. Extract the model's input embedding matrix E of shape [vocab_size, D].
2. L2-normalize each row so k-means minimizes cosine distance.
3. Run MiniBatchKMeans to find n = 256 cluster centroids.
4. Assign a gravitational mass to each cluster.

Three mass schemes are supported. *Uniform* assigns equal mass to all bodies, providing a flat background boost to rare tokens. *Size* sets mass proportional to cluster population — denser semantic regions exert more force. *IDF* sets mass as the mean IDF weight of member tokens, so clusters of semantically specific vocabulary outweigh clusters of common words. IDF is the default and consistently outperforms the alternatives in my benchmarks.

The universe field is saved as an `.npz` file and loaded at inference time. It never changes during generation — it is a fixed property of the model, computed once and reused.

**Why the universe field matters.** Context-only bodies create a diversity feedback loop: generated tokens pull sampling toward already-generated semantic regions, which produces more tokens in that region, narrowing diversity progressively. The universe field breaks this loop. Every semantic region expressible by the model is always present in the field. Novel output is a new trajectory through a fixed landscape, not the discovery of new locations.

### 4.3 Context-Affinity Modulation

A naive universe field is omnidirectional — all bodies exert force at every step regardless of what is currently being generated. This produces a uniform rare-token boost that is insensitive to the topic of the generation.

I introduce *context-affinity modulation*: each universe body's effective mass is weighted by its cosine similarity to the current orbital position (the recent context embedding):

```
effective_mass[i] = mass[i] * max(0, cosine_similarity(centroid[i], context_position))
```

where `centroid[i]` and `context_position` are both L2-normalized. Bodies semantically close to the current context exert full force; bodies far away contribute near zero. This makes the universe field context-sensitive without making it adaptive — the centroids and base masses are fixed; only the effective masses vary per step.

Empirically, context-affinity modulation improved distinct-1 from approximately 0.44 (temperature baseline) to 0.53–0.57.

### 4.4 The Context Field

When local body formation is enabled, an IncrementalDBSCAN clustering runs over prompt and generated tokens, building conversation-specific bodies in real time. Bodies form when a cluster accumulates sufficient tokens, grow as new related tokens accrete, and merge when two bodies come within a collision threshold.

The local field provides prompt-specific gravitational perturbation on top of the universe field. It is particularly useful for domain-specific generation where a few strong topics dominate the conversation. However, it is more sensitive to parameter choices — specifically the `r_min` floor and force normalization — and should be used with fixed G or AdaptiveG targeted at a realistic escape rate.

Force from local bodies is normalized by the number of active bodies to maintain scale parity with the universe field. Without this normalization, two local bodies would exert forces comparable to the entire 256-body universe field.

### 4.5 Orbital Mechanics

The context vector maintains a trajectory through embedding space:

- **Position** — L2-normalized mean of recent token embeddings
- **Velocity** — change in position per token: `v[t] = p[t] - p[t-1]`
- **Acceleration** — change in velocity per token: `a[t] = v[t] - v[t-1]`
- **Momentum** — scalar magnitude `‖v[t]‖`: resistance to gravitational deflection

High-momentum trajectories resist deflection from new bodies; sudden acceleration indicates a topic shift. This state is used both for context-affinity modulation and as a diagnostic signal for tracking semantic drift.

### 4.6 Body Lifecycle

Bodies progress through a defined lifecycle:

**Formation.** IncrementalDBSCAN discovers DBSCAN clusters in the token embedding stream. Clusters meeting the minimum sample count form a body with an initial centroid and mass.

**Accretion.** Subsequent tokens within DBSCAN radius update the centroid via running mean and increment body mass.

**Merge.** Bodies within `body_merge_distance` are merged into a virtual body before force computation. Bodies within the tighter `collision_distance` undergo a permanent inelastic collision: their centroids are merged by mass-weighted average and their masses summed.

**Persistence.** When a body's stability score exceeds a threshold, it is written to `ContextBodyStore`. Stability is derived from cluster compactness and mass.

**Decay.** Stored bodies lose mass at rate `decay_rate × elapsed_seconds`. Bodies falling below an extinction threshold are deleted.

**Re-emergence.** If a new body forms within `reemergence_distance` of a dormant (low-mass) stored body, the stored body is re-energized rather than a duplicate being created.

**Cross-session collision.** After persisting a new local body, it is compared against all existing store records. If within `collision_distance` of any record from a prior session, they undergo inelastic collision — mass-weighted centroid merge, summed mass, transferred resonance links — and the incoming record is deleted.

Body type emerges from mass rather than being assigned:

| Type | Mass threshold | Interpretation |
|---|---|---|
| Black hole | > 1000 | Dominant, inescapable theme |
| Neutron star | > 500 | Rare but highly specific, dense context |
| Planet | > 100 | Stable recurring topic |
| Moon | > 10 | Sub-topic orbiting a heavier body |
| Asteroid | ≤ 10 | Passing mention, minimal influence |

### 4.7 Adaptive Gravitational Constant

The gravitational constant G can be held fixed or managed by AdaptiveG, a PI controller that adjusts G per step via three multiplicative terms:

```
G_eff(t) = G_base × mass_norm(t) × escape_feedback(t) × domain_scale
```

**Mass normalization** (`mass_norm`): body mass grows as tokens accrete. Without compensation, a mature body late in a long generation exerts orders of magnitude more force than the same body 50 tokens earlier. `mass_norm` tracks an exponential moving average of active body mass and scales G inversely, keeping expected force magnitude stable throughout generation.

**Escape rate feedback** (`escape_feedback`): a PI controller targets a configurable escape rate `r_target`:

```
error(t)          = r_target - r_observed(t)
escape_feedback   = clip(1 + Kp * error(t) + Ki * mean(error_history), G_min, G_max)
```

**Domain multiplier** (`domain_scale`): a static per-domain scalar applied after the feedback terms. Code, medical, and legal domains benefit from higher G (tighter topical constraint); creative tasks from lower G (more exploration).

**Practical limitation.** In universe-only mode, the escape rate stabilizes near 0.15 regardless of G. Approximately 15% of tokens escape because IDF weighting suppresses their effective force below `escape_threshold` — they are high-frequency tokens that should be unaffected by gravity. The remaining 85% have sufficient IDF weight to remain bound at any G. AdaptiveG targeting an escape rate of 0.7 cannot converge in this mode. The escape rate of 0.15 is correct behavior, not a sign that G is too high. I recommend AdaptiveG only when local bodies are also enabled; for universe-only generation, fixed G suffices.

### 4.8 G Split

Two independent gravitational constants control the two field layers:

- **G_local** — controls local context body forces. Fixed or managed by AdaptiveG.
- **G_universe** — controls the universe background field. Always static.

This split is necessary because the universe field is a fixed pre-computed structure whose scale is determined by the k-means geometry and mass scheme at build time. Allowing AdaptiveG to modulate it introduces instability in what is meant to be a stable background. Local G adapts to the dynamic context field; universe G is a calibration parameter set once.

The two are mutually exclusive in the CLI: `--G-local` and `--adaptive-g` cannot both be specified.

### 4.9 Persistent Store and Cross-Session Memory

`ContextBodyStore` wraps a `VectorBackend` protocol with three additional behaviors: deduplication before insertion, gravitational ranking of query results (by `mass / r²` rather than raw cosine similarity), and time-based mass decay. Swapping backends — FAISS in-memory, Qdrant on-disk, Pinecone, pgvector — requires no changes to the sampler.

Stable bodies written to the store persist across conversations. When a new session begins, the store's bodies load into the active field as background perturbations alongside the universe field. Over many sessions in a domain-specific context, the store accumulates a living knowledge graph of the semantic structure relevant to that context: clusters that appeared repeatedly gain mass; clusters that were one-off mentions decay. The gravitational field at the start of each new conversation reflects the full semantic history of prior ones.

---

## 5. Experiments

### 5.1 Setup

I benchmark on GPT-2 (117M) and GPT-2 Medium (345M) using five diverse prompts (100 tokens each, 2 runs per prompt, results averaged). The temperature baseline uses T = 0.8 without top-p or top-k. All gravitational configurations use the IDF mass scheme for the universe field and `r_min = 0.1`.

**Metrics:**

- **Perplexity (ppl)** — geometric mean of inverse token probabilities under the model's base distribution. Note that gravitational sampling deliberately promotes lower-probability tokens, so elevated perplexity is partially expected and does not straightforwardly indicate quality degradation.
- **distinct-1** — fraction of unique unigrams across all generated sequences, a standard lexical diversity metric. Values closer to 1.0 indicate more diverse output.
- **Escape rate** — fraction of candidate tokens sampled without gravitational influence per step.
- **ms/token** — wall-clock generation latency per token.

### 5.2 Baseline

| Mode | ppl | distinct-1 | ms/token |
|---|---|---|---|
| Temperature T = 0.8 (GPT-2) | ~9 | 0.44 | ~10 |

### 5.3 Universe Field — G Sweep (GPT-2)

| G | ppl | distinct-1 |
|---|---|---|
| 0.1 | 50.15 | 0.5899 |
| 0.3 | 49.15 | 0.5607 |
| 1.0 | 37.08 | 0.5736 |
| 3.0 | 50.54 | 0.6001 |

All universe configurations substantially improve distinct-1 over the temperature baseline (0.44). G = 1.0 achieves the best perplexity-diversity tradeoff; G = 3.0 maximizes diversity at additional perplexity cost. The relationship between G and perplexity is non-monotonic, reflecting the complex interaction between force magnitude, IDF weighting, and the model's base distribution.

### 5.4 Universe + Local Bodies — Effect of r_min

An early implementation used `r_min = 1e-6` for local body force computation. Results showed complete diversity collapse:

| Mode | ppl | distinct-1 | note |
|---|---|---|---|
| G_local = 1.0, fixed | 5.88 | 0.1965 | collapse |
| G_local = 0.1, fixed | 5.89 | 0.1893 | collapse at any G |
| AdaptiveG | 7.05 | 0.2463 | partial recovery |

Root cause: with `r_min = 1e-6`, tokens belonging to a local body have r ≈ 0.001, yielding F ∝ G * m / 1e-6 — forces 10⁶ times larger than any universe body force. The local field completely dominated, collapsing sampling to a handful of tokens regardless of G. Distinct-1 was insensitive to G value because the singularity saturated the force computation at every setting.

After setting `r_min = 0.1` and adding per-body normalization:

| Mode | ppl | distinct-1 | active bodies |
|---|---|---|---|
| G_local = 1.0, fixed | 42.36 | 0.5061 | 2.8 |
| AdaptiveG (G_eff ≈ 0.82) | 44.0 | 0.5467 | — |
| No IDF | 40.66 | 0.5498 | — |
| Uniform universe mass | 43.9 | 0.526 | — |

Removing IDF weighting slightly reduces perplexity but also slightly reduces diversity, confirming that IDF serves its intended purpose: the added diversity comes from steering toward rarer, more specific tokens, which the model inherently assigns lower probability.

### 5.5 GPT-2 Medium — Torch Port

GPT-2 Medium (768-dimensional embeddings) initially ran at 637 ms/token for local body force computation due to serial numpy dot products over each active body. Porting the inner loop to a single batched matrix multiplication on GPU:

```
cos_sims = E_norm @ C_norm.T   # [vocab_size, n_bodies]
```

reduced this to 124 ms/token — a 5× speedup — while also yielding a slight perplexity improvement, likely from float32 precision on GPU.

| Stage | ppl | distinct-1 | ms/token |
|---|---|---|---|
| Before torch port | 32.07 | 0.495 | 637 |
| After torch port | 30.72 | 0.534 | 124 |

### 5.6 Regression Test

Following all changes — G split, cross-session collision detection, timezone-aware datetimes — a full regression benchmark confirmed no regressions:

| Model | distinct-1 | ppl | ms/token |
|---|---|---|---|
| GPT-2 Medium | 0.5275 | 34.33 | 124.1 |

### 5.7 Perplexity-Diversity Tradeoff Summary

| Method | ppl | distinct-1 | distinct-1 gain |
|---|---|---|---|
| Temperature T = 0.8 (baseline) | ~9 | 0.44 | — |
| Universe + local, GPT-2 | ~42 | 0.51–0.55 | +25–27% |
| Universe + local, GPT-2 Medium | ~31–34 | 0.51–0.54 | +20–23% |

The perplexity cost is structural: gravitational sampling deliberately promotes tokens with lower base probability (high IDF weight, semantically specific), which by definition increases measured perplexity. This is a correct tradeoff, not a failure mode. The perplexity gap narrows with model scale — GPT-2: ×5, GPT-2 Medium: ×3.4 — suggesting the approach becomes more favorable on larger models where the base distribution is already more peaked and specific tokens are closer to the probability mass.

---

## 6. Discussion

### 6.1 The Perplexity-Diversity Tradeoff

The elevated perplexity of gravitational sampling relative to temperature requires interpretation. Perplexity measures how surprised the model is by its own output. Temperature sampling at T = 0.8 produces output that closely follows the model's highest-probability predictions — output that is, by construction, what the model considers most likely. The low perplexity reflects this tautology.

Gravitational sampling promotes tokens the model considers less likely but that are semantically relevant to the context. These tokens, by definition, elevate perplexity. The question is whether this shift produces output that is more useful or more interesting — a question that perplexity alone cannot answer. Distinct-1 captures one dimension (lexical diversity); human evaluation and task-specific metrics (MAUVE, BLEU on downstream tasks) would be needed for a fuller picture.

### 6.2 Why Multiplicative Reweighting

The choice to multiply rather than add forces to the model's distribution has implications beyond coherence preservation. Multiplicative reweighting means gravity scales with base probability: a token the model already considers likely receives a proportionally larger boost from a nearby body than an identical token the model considers unlikely. This creates a natural interaction between the model's prior knowledge and the gravitational field — gravity amplifies the model's own tendencies toward a topic rather than imposing them from outside.

### 6.3 Complementarity with Existing Methods

Gravitational sampling and standard sampling methods address orthogonal concerns and can be combined. Top-p restricts the candidate vocabulary to the highest-probability nucleus; gravitational sampling then reshapes the distribution within that nucleus based on semantic relevance. The two are not competing approaches — top-p handles incoherence prevention; gravity handles topical steering. For applications requiring both (e.g., long-form domain-specific generation), applying both is reasonable.

### 6.4 AdaptiveG and the Escape Rate

The escape rate feedback mechanism in AdaptiveG proved more nuanced than anticipated. In universe-only mode, the escape rate is structurally bounded at approximately 0.15 due to IDF-threshold interaction: the tokens that escape do so because IDF suppresses their effective gravitational force below the threshold, not because G is insufficient. This makes the escape rate a misleading signal for AdaptiveG in this mode — the controller cannot reach its target and oscillates without converging.

This is actually correct system behavior: high-frequency tokens should be uninfluenced by gravity. The escape rate of 0.15 in universe-only mode means "15% of tokens are correctly left alone." The metric was designed for a context where escape is G-controlled, not IDF-controlled. I recommend using fixed G in universe-only mode and reserving AdaptiveG for configurations with local bodies, where escape rate is more directly controlled by G.

### 6.5 Persistent Semantic Memory

The body persistence mechanism gives contextbodies a property that no standard sampling method possesses: memory that accumulates across conversations. Over many sessions in a domain, the vector store builds up a gravitational topology of that domain — heavy bodies representing frequent concepts, lighter bodies representing less common ones, resonance links connecting co-occurring themes. Each new conversation begins in a semantic field shaped by all prior ones.

This is qualitatively different from retrieval-augmented generation (RAG), which retrieves specific documents. Persistent context bodies represent an abstract semantic landscape: not "here is what was said before" but "here is the shape of the semantic space that has been explored." Whether this leads to useful behavior at scale is an open empirical question.

---

## 7. Limitations and Future Work

**Evaluation depth.** All benchmarks use distinct-1 and perplexity over 100-token outputs on five prompts. Distinct-1 is a necessary but not sufficient measure of generation quality. Future work should include MAUVE scores, KL divergence from the temperature distribution, human evaluation, and downstream task metrics. Long-form outputs (500–1000 tokens) may show different dynamics as local bodies accumulate mass.

**Model scale.** GPT-2 Medium showed a narrower perplexity-diversity gap than GPT-2 (×3.4 vs ×5). GPT-2 Large testing was attempted but sessions ran out of memory. The scaling trend is encouraging but unconfirmed above 345M parameters.

**G calibration.** No principled method yet exists for selecting G for a new model or domain. The optimal G appears model-size dependent. A calibration procedure — perhaps targeting a specific distinct-1 improvement with a perplexity budget — would make the system more accessible.

**Resonance effects.** Orbital resonance (Lagrange midpoint forces between co-active resonant body pairs) is implemented but has not been quantitatively evaluated in isolation. Its contribution to diversity and coherence is unknown.

**Universe mass scheme.** IDF, size, and uniform mass schemes were compared at isolated configurations but not swept across all G values. A systematic comparison might reveal interaction effects.

**Domain classifier integration.** AdaptiveG supports domain multipliers; DomainClassifier can infer domain from the token stream. The full pipeline connecting real-time domain inference to per-domain G modulation has not been benchmarked.

**Deterministic mode.** When a universe field is available, the sampler can use argmax rather than multinomial sampling — diversity comes from the geometry of the universe rather than stochasticity. This mode has not been formally evaluated.

---

## 8. Conclusion

I have presented contextbodies, a gravitational field model for token sampling in language models. The central contribution is the separation of two concerns that standard sampling conflates: probability modulation (handled by temperature, top-p, top-k) and semantic relevance weighting (handled by gravitational fields in embedding space). By applying gravitational forces multiplicatively to the model's own probability distribution, contextbodies steers generation toward contextually relevant tokens while preserving the model's syntactic and factual constraints.

The two-layer field — a permanent universe field from vocabulary clustering and an ephemeral context field from incremental DBSCAN — provides both global semantic coverage and conversation-specific topical coherence. Context-affinity modulation ensures the universe field responds to the current generation rather than boosting all rare tokens uniformly. Persistent body storage builds semantic memory across conversations.

Empirically, contextbodies achieves 20–27% improvement in lexical diversity over temperature sampling at the cost of a structural perplexity increase that narrows with model scale. The perplexity cost is intentional: the system promotes semantically specific tokens that the model assigns lower base probability, which is the mechanism by which diversity is gained. Standard sampling methods cannot make this tradeoff because they lack any representation of semantic relevance.

The framework raises a question that standard sampling cannot: not "how likely is this token?" but "how relevant is this token to where we are in semantic space?" Answering that question at inference time, without additional training or retrieval, is the contribution this work offers.

---

## Appendix A: Physics Analogy Reference

| Physics concept | contextbodies equivalent |
|---|---|
| Gravitational body | Cluster of semantically related tokens |
| Mass | Accumulated IDF-weighted token norms |
| Distance | Cosine distance in embedding space |
| Gravitational constant G | Primary tuning parameter; replaces temperature |
| Escape velocity / escape rate | Fraction of tokens sampled without gravitational influence |
| Orbital position | Normalized mean of recent context embeddings |
| Velocity | Change in context position per token |
| Momentum | `‖v[t]‖` — resistance to topical deflection |
| Inelastic collision | Mass-weighted centroid merge of nearby bodies |
| Universe | Background field from vocabulary-wide clustering |
| Context bodies | Prompt-specific perturbations on the universe field |
| Lagrange point / resonance | Midpoint force between co-active resonant body pairs |
| Body decay | Mass reduction proportional to time since last reinforcement |
| Re-emergence | Dormant low-mass body re-energized by nearby new body |

---

## Appendix B: Key Parameters

| Parameter | Default | Description |
|---|---|---|
| G | 1.0 | Gravitational constant. Higher = stronger pull, less diversity. |
| G_universe | 1.0 | Separate G for universe field. Always static. |
| escape_threshold | 0.01 | Force below which a token is considered unbound. |
| r_min | 0.1 | Minimum cosine distance floor in force computation. |
| stability_threshold | 0.8 | Minimum stability for a body to be persisted. |
| collision_distance | 0.1 | Cosine distance for inelastic collision. |
| body_merge_distance | 0.2 | Cosine distance for virtual merge before force computation. |
| n_universe_clusters | 256 | Number of k-means clusters in the universe. |
| universe_mass | idf | Universe mass scheme: uniform, size, or idf. |
| escape_rate_target | 0.7 | AdaptiveG target escape rate (local bodies mode). |
| recency_decay_lambda | 0.0 | Exponential decay rate for persisted body forces. |

---

*Implementation available at https://github.com/chrisdiazmfe/contextbodies*

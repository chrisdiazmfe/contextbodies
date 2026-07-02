"""
benchmark.py — compare gravitational sampling against temperature baseline.

Usage:
    # Standard comparison (temperature vs gravitational)
    python benchmark.py --universe-path universe.npz --G-universe 1.0

    # Ablation battery (real/shuffled/random/no-idf/uniform-mass/local-only/combined)
    python benchmark.py --ablation --universe-path universe.npz --G-universe 1.0

    # Include top-p and typical sampling baselines
    python benchmark.py --baselines all --universe-path universe.npz

    # Diagnostic mode: print top boosted tokens each step (first prompt only)
    python benchmark.py --log-diagnostics --universe-path universe.npz --runs 1

    # Other options
    python benchmark.py --model gpt2-medium --max-tokens 150 --runs 3
    python benchmark.py --prompts-file my_prompts.txt --G-universe 2.0
    python benchmark.py --output results/run1 --seed 42

Outputs:
    <output-dir>/results.json       — full numeric results
    <output-dir>/summary.txt        — human-readable comparison table
    <output-dir>/steps.json         — per-step gravitational metrics
    <output-dir>/ablation.txt       — ablation comparison table (--ablation mode)
    <output-dir>/diagnostics.json   — per-step top-boosted-token data (--log-diagnostics)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from context_body_store import ContextBodyStore
from gravitational_sampler import GravitationalSampler
from adaptive_g import AdaptiveG
from adaptive_dbscan import AdaptiveDBSCAN
from universe_builder import Universe, UniverseBuilder


# ---------------------------------------------------------------------------
# Default prompt set — diverse topics for broad coverage
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    "The history of computing began with",
    "In the deep ocean, scientists recently discovered",
    "The most effective approach to machine learning is",
    "Once upon a time in a small village",
    "The economic implications of artificial intelligence include",
    "To prepare a perfect risotto, you must first",
    "Quantum entanglement is a phenomenon where",
    "The city at night was transformed into",
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def distinct_n(texts: list[str], n: int) -> float:
    """Fraction of unique n-grams across all texts (higher = more diverse)."""
    all_ngrams: list[tuple] = []
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def repetition_rate(texts: list[str], n: int = 2) -> float:
    """
    Fraction of n-grams that are repeated (i.e. not unique).

    Complement of distinct_n: repetition_rate = 1 - distinct_n.
    Reported separately because it maps directly to the failure mode name
    (\"repetition\") that reviewers check for.
    """
    return 1.0 - distinct_n(texts, n)


def perplexity(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    device: str,
    max_length: int = 512,
) -> float:
    """
    Average per-token perplexity of the model on generated texts.
    Lower = model assigns higher probability to its own output.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)
            input_ids = enc["input_ids"]
            if input_ids.shape[1] < 2:
                continue
            outputs = model(input_ids, labels=input_ids)
            # outputs.loss is mean NLL per token
            n_tokens = input_ids.shape[1] - 1
            total_nll += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")
    return float(np.exp(total_nll / total_tokens))


def avg_length(texts: list[str]) -> float:
    """Average word count."""
    if not texts:
        return 0.0
    return sum(len(t.split()) for t in texts) / len(texts)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def generate_temperature(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    device: str,
) -> tuple[str, float]:
    """Standard temperature sampling. Returns (text, elapsed_seconds)."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text, elapsed


def generate_topp(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> tuple[str, float]:
    """Nucleus (top-p) sampling baseline. Returns (text, elapsed_seconds)."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0
    return tokenizer.decode(out[0], skip_special_tokens=True), elapsed


def generate_typical(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int,
    typical_p: float,
    device: str,
) -> tuple[str, float]:
    """
    Locally typical sampling baseline (Meister et al., 2023).

    Selects tokens whose information content is close to the conditional
    entropy of the distribution, avoiding both the most predictable and the
    most surprising tokens.

    Returns (text, elapsed_seconds).
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            typical_p=typical_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0
    return tokenizer.decode(out[0], skip_special_tokens=True), elapsed


@dataclass
class StepMetrics:
    step: int
    escape_rate: float
    active_bodies: int
    latency_ms: float
    G_eff: Optional[float] = None


def generate_gravitational(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_tokens: int,
    sampler: GravitationalSampler,
    device: str,
    collect_diagnostics: bool = False,
    temperature: float = 1.0,
) -> "tuple[str, float, list[StepMetrics], list[dict]]":
    """
    Gravitational sampling with per-step metric collection.
    Returns (text, elapsed_seconds, step_metrics, diagnostic_data).
    diagnostic_data is a list of per-step dicts (empty if collect_diagnostics=False).
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    token_embeddings = model.get_input_embeddings().weight
    embedding_dim = token_embeddings.shape[1]

    with torch.no_grad():
        context_embeddings = token_embeddings[input_ids[0]]

    sampler.initialize(context_embeddings, embedding_dim)

    generated = input_ids[0].tolist()
    step_metrics: list[StepMetrics] = []
    diagnostic_data: list[dict] = []
    total_start = time.perf_counter()
    past_key_values = None

    for step in range(max_tokens):
        with torch.no_grad():
            t0 = time.perf_counter()
            if past_key_values is None:
                outputs = model(
                    torch.tensor([generated], device=device),
                    use_cache=True,
                )
            else:
                outputs = model(
                    torch.tensor([[generated[-1]]], device=device),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[0, -1, :]

            next_token = sampler.sample(
                logits=logits / temperature,
                token_embeddings=token_embeddings,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

        # Update clustering with the selected token
        tok_emb = token_embeddings[next_token].detach().cpu().numpy()
        sampler.post_step(token_id=next_token, token_embedding=tok_emb)

        escape_rate = (
            sampler._last_escape_count / (sampler._last_vocab_size + 1e-8)
            if sampler._last_vocab_size > 0 else 0.0
        )
        active_bodies = len(sampler.active_bodies)
        G_eff = sampler.adaptive_g._G_eff if sampler.adaptive_g else None

        step_metrics.append(StepMetrics(
            step=step,
            escape_rate=escape_rate,
            active_bodies=active_bodies,
            latency_ms=latency_ms,
            G_eff=G_eff,
        ))

        # Collect per-step diagnostic info (top-boosted tokens)
        if collect_diagnostics:
            info = sampler.get_last_diagnostic_info()
            if info is not None:
                info = dict(info)
                info["step"] = step
                diagnostic_data.append(info)

        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break

    elapsed = time.perf_counter() - total_start
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, elapsed, step_metrics, diagnostic_data


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_summary(results: dict) -> None:
    cfg = results["config"]
    temp = results["temperature"]
    grav = results.get("gravitational")

    print("\n" + "=" * 60)
    print("  contextbodies benchmark")
    print("=" * 60)
    print(f"  model          : {cfg['model']}")
    print(f"  prompts        : {cfg['num_prompts']}")
    print(f"  max tokens     : {cfg['max_tokens']}")
    print(f"  runs per prompt: {cfg['runs']}")
    print(f"  temperature    : {cfg['temperature']}")
    if grav is not None:
        if cfg.get("adaptive_g"):
            print(f"  G_local        : adaptive")
            print(f"  G_base         : {cfg.get('G_base', 1.0)}")
        else:
            print(f"  G_local        : {cfg['G_local']}")
        print(f"  G_universe     : {cfg['G_universe']}")
        print(f"  adaptive_g     : {cfg['adaptive_g']}")
        if cfg.get("escape_rate_target") is not None:
            print(f"  escape_rate_tgt: {cfg['escape_rate_target']}")
        if cfg.get("mass_damping") is not None:
            print(f"  mass_damping   : {cfg['mass_damping']}")
        if cfg.get("adaptive_clustering"):
            print(f"  target_bodies  : {cfg['target_bodies']}")
            print(f"  eps_percentile : {cfg['eps_percentile']}")
            print(f"  eps_adj_rate   : {cfg['eps_adjustment_rate']}")
        if cfg.get("universe_path"):
            print(f"  universe       : {cfg['universe_path']}")
            print(f"  universe_mass  : {cfg['universe_mass']}")
        print(f"  local_bodies   : {cfg['local_bodies']}")
        print(f"  deterministic  : {cfg['deterministic']}")
    print()

    if grav is None:
        # Temperature-only mode
        print(f"  {'metric':<28} {'temperature':>14}")
        print(f"  {'-'*28} {'-'*14}")
        print(f"  {'perplexity':<28} {temp['perplexity']:>14.4f}")
        print(f"  {'distinct-1':<28} {temp['distinct_1']:>14.4f}")
        print(f"  {'distinct-2':<28} {temp['distinct_2']:>14.4f}")
        print(f"  {'avg length (words)':<28} {temp['avg_length']:>14.1f}")
        print(f"  {'total time (s)':<28} {temp['total_time_s']:>14.2f}")
        print(f"  {'ms / token (mean)':<28} {temp['ms_per_token']:>14.1f}")
    else:
        print(f"  {'metric':<28} {'temperature':>14} {'gravitational':>14}")
        print(f"  {'-'*28} {'-'*14} {'-'*14}")

        def row(label, t_val, g_val, fmt=".4f"):
            print(f"  {label:<28} {t_val:>14{fmt}} {g_val:>14{fmt}}")

        row("perplexity",         temp["perplexity"],   grav["perplexity"])
        row("distinct-1",         temp["distinct_1"],   grav["distinct_1"])
        row("distinct-2",         temp["distinct_2"],   grav["distinct_2"])
        row("avg length (words)", temp["avg_length"],   grav["avg_length"],  ".1f")
        row("total time (s)",     temp["total_time_s"], grav["total_time_s"], ".2f")
        row("ms / token (mean)",  temp["ms_per_token"], grav["ms_per_token"], ".1f")

        if grav.get("semantic_coverage") is not None:
            label = "semantic coverage"
            g_val = grav["semantic_coverage"]
            print(f"  {label:<28} {'—':>14} {g_val:>14.4f}")

        if grav.get("step_metrics_summary"):
            sm = grav["step_metrics_summary"]
            print()
            print("  gravitational system metrics (mean across all steps):")
            print(f"  {'escape rate':<28} {sm['mean_escape_rate']:>14.4f}")
            print(f"  {'active bodies':<28} {sm['mean_active_bodies']:>14.1f}")
            if sm.get("mean_G_eff") is not None:
                print(f"  {'G_eff':<28} {sm['mean_G_eff']:>14.4f}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="contextbodies benchmark")
    p.add_argument("--model", default="gpt2",
                   help="HuggingFace model name (default: gpt2)")
    p.add_argument("--prompts-file", default=None,
                   help="Path to a text file with one prompt per line")
    p.add_argument("--max-tokens", type=int, default=100,
                   help="Max tokens to generate per prompt (default: 100)")
    p.add_argument("--runs", type=int, default=2,
                   help="Runs per prompt per sampler (results averaged)")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Temperature for baseline sampler (default: 0.8)")
    p.add_argument("--temperature-only", action="store_true",
                   help="Run only the temperature baseline, skip gravitational sampling. "
                        "Use to sweep temperature values for a matched-perplexity comparison.")
    # G_local and --adaptive-g are mutually exclusive: you either fix local
    # body G with --G-local, or let AdaptiveG manage it automatically.
    g_group = p.add_mutually_exclusive_group()
    g_group.add_argument("--G-local", type=float, default=None, dest="G_local",
                   help="Fixed gravitational constant for local context bodies (default: 1.0). "
                        "Mutually exclusive with --adaptive-g.")
    g_group.add_argument("--adaptive-g", action="store_true",
                   help="Let AdaptiveG automatically tune local body G to hit the escape-rate "
                        "target. Mutually exclusive with --G-local.")
    p.add_argument("--G-base", type=float, default=1.0, dest="G_base",
                   help="Starting G for AdaptiveG (default: 1.0). Only used with --adaptive-g. "
                        "Ignored when --G-local is set.")
    p.add_argument("--G-universe", type=float, default=1.0, dest="G_universe",
                   help="Gravitational constant for the universe background field (default: 1.0). "
                        "Independent of --G-local / --adaptive-g. Tune this to control the "
                        "background diversity boost without interfering with AdaptiveG.")
    p.add_argument("--escape-threshold", type=float, default=0.01,
                   help="Escape threshold (default: 0.01). A token escapes gravity if "
                        "its IDF-weighted force magnitude is below this value. Because "
                        "IDF suppresses common tokens toward zero, this correctly "
                        "captures tokens where gravity has negligible effective influence.")
    p.add_argument("--escape-rate-target", type=float, default=0.7,
                   help="AdaptiveG target escape rate (default: 0.7). Lower values "
                        "allow gravity to influence more tokens; 0.15-0.20 works "
                        "well with multiplicative reweighting.")
    p.add_argument("--mass-damping", type=float, default=1.0, dest="mass_damping",
                   help="How aggressively body mass growth reduces G (default: 1.0). "
                        "1.0=full inverse scaling, 0.5=square-root dampening, "
                        "0.0=disabled. Keeps force magnitudes stable as bodies accumulate "
                        "mass over a long generation.")
    p.add_argument("--recency-decay", type=float, default=0.0, dest="recency_decay",
                   help="Rate at which persisted body force fades over time (default: 0.0 = off). "
                        "1e-4 ≈ 2-hour half-life; 1e-5 ≈ 19-hour half-life.")
    p.add_argument("--no-idf", action="store_true",
                   help="Disable IDF mass normalization (use raw token_mass for all tokens)")
    p.add_argument("--cluster-radius", type=float, default=0.3, dest="cluster_radius",
                   help="Cosine distance radius for local context body clustering (default: 0.3). "
                        "Smaller = tighter clusters. Ignored when --adaptive-clustering is set.")
    p.add_argument("--cluster-min-tokens", type=int, default=3, dest="cluster_min_tokens",
                   help="Minimum tokens required to form a context body cluster core (default: 3)")
    p.add_argument("--adaptive-clustering", action="store_true", dest="adaptive_clustering",
                   help="Auto-calibrate cluster radius from prompt embedding geometry "
                        "and adjust toward --target-bodies during generation.")
    p.add_argument("--target-bodies", type=int, default=3,
                   help="AdaptiveDBSCAN target number of simultaneously active bodies (default: 3).")
    p.add_argument("--eps-percentile", type=float, default=20.0,
                   help="Percentile of pairwise context distances used to seed eps (default: 20.0).")
    p.add_argument("--eps-adjustment-rate", type=float, default=0.01,
                   help="How much eps changes per step when body count is off-target (default: 0.01). "
                        "Lower values = smoother but slower adaptation.")
    # Universe
    p.add_argument("--universe-path", default=None,
                   help="Path to a pre-built universe .npz file. "
                        "If omitted and --build-universe is not set, no universe is used.")
    p.add_argument("--build-universe", action="store_true",
                   help="Build a new universe from the model's vocabulary embeddings "
                        "before running benchmarks. Saved to <output>/universe.npz.")
    p.add_argument("--universe-clusters", type=int, default=256,
                   help="Number of k-means clusters (universe bodies) to build (default: 256).")
    p.add_argument("--universe-mass", default="idf",
                   choices=["uniform", "size", "idf"],
                   help="Mass scheme for universe bodies: uniform, size (cluster size), "
                        "or idf (mean IDF weight of member tokens). Default: idf.")
    p.add_argument("--universe-top-k", type=int, default=16, dest="universe_top_k",
                   help="Number of top contextually-aligned universe bodies to apply force "
                        "from (default: 16). Lower values concentrate force on fewer, more "
                        "relevant bodies, making semantic geometry matter more. "
                        "Set to 256 (n_clusters) to use all bodies.")
    # Context body layer
    p.add_argument("--local-bodies", action="store_true",
                   help="Enable the local context body layer (DBSCAN clustering of "
                        "prompt and generated tokens). Provides prompt-specific "
                        "gravitational perturbations on top of the universe field. "
                        "When omitted, only the universe field is used.")
    p.add_argument("--deterministic", action="store_true",
                   help="Use argmax instead of multinomial sampling. "
                        "Diversity comes entirely from universe geometry — "
                        "same prompt always produces the same output.")
    p.add_argument("--output", default="benchmark_results",
                   help="Output directory (default: benchmark_results)")
    p.add_argument("--device", default=None,
                   help="Device: cuda / cpu (auto-detected if omitted)")
    p.add_argument("--seed", type=int, default=None,
                   help="Global random seed for reproducibility (default: None = non-deterministic). "
                        "Sets numpy + torch seeds before each run.")

    # ── Additional baselines ────────────────────────────────────────────────
    p.add_argument("--baselines", default="temperature",
                   choices=["temperature", "topp", "typical", "all"],
                   help="Which baselines to run. 'temperature' = default single baseline. "
                        "'topp' adds nucleus sampling. 'typical' adds typical sampling. "
                        "'all' runs temperature + top-p + typical (default: temperature).")
    p.add_argument("--top-p", type=float, default=0.9, dest="top_p",
                   help="Nucleus sampling p parameter (default: 0.9). Used when --baselines "
                        "includes 'topp' or 'all'.")
    p.add_argument("--typical-p", type=float, default=0.95, dest="typical_p",
                   help="Typical sampling p parameter (default: 0.95). Used when --baselines "
                        "includes 'typical' or 'all'.")

    # ── Ablation mode ───────────────────────────────────────────────────────
    p.add_argument("--ablation", action="store_true",
                   help="Run full ablation battery: real universe, shuffled centroids, "
                        "random centroids, no-IDF, uniform mass, local-only, and combined. "
                        "Requires --universe-path or --build-universe. "
                        "Outputs ablation.txt comparison table.")
    p.add_argument("--ablation-conditions", default=None,
                   help="Comma-separated subset of ablation conditions to run. "
                        "Available: real,shuffled,random,no_idf,uniform_mass,local_only,combined. "
                        "Default (when --ablation is set): all conditions.")

    # ── Diagnostics ─────────────────────────────────────────────────────────
    p.add_argument("--log-diagnostics", action="store_true",
                   help="Print top boosted tokens at each generation step. "
                        "Runs only on the first prompt to limit output. "
                        "Saves full data to diagnostics.json.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

def _run_one_condition(
    label: str,
    model,
    tokenizer,
    prompts: list[str],
    args,
    device: str,
    G_local: float,
    universe,           # Universe | None — the condition-specific universe variant
    use_idf: bool,
    local_bodies: bool,
    idf_weights,
    seed: int | None,
    collect_diagnostics: bool = False,
) -> dict:
    """Run gravitational sampling for one ablation condition. Returns metrics dict."""
    embedding_dim = model.get_input_embeddings().weight.shape[1]

    texts: list[str] = []
    times: list[float] = []
    token_counts: list[int] = []
    all_step_metrics: list[dict] = []
    all_diagnostic_data: list[dict] = []

    for i, prompt in enumerate(prompts):
        for run in range(args.runs):
            if seed is not None:
                np.random.seed(seed + i * 100 + run)
                torch.manual_seed(seed + i * 100 + run)

            store = ContextBodyStore(embedding_dim=embedding_dim, decay_interval=9999)
            sampler = GravitationalSampler(
                body_store=store,
                G=G_local,
                G_universe=args.G_universe,
                escape_threshold=args.escape_threshold,
                recency_decay_lambda=args.recency_decay,
                universe=universe,
                use_context_bodies=local_bodies,
                deterministic=args.deterministic,
                device=device,
                cluster_radius=args.cluster_radius,
                cluster_min_tokens=args.cluster_min_tokens,
            )
            sampler.universe_top_k_bodies = args.universe_top_k
            sampler._idf_weights = idf_weights if use_idf else None
            if collect_diagnostics and i == 0 and run == 0:
                sampler.set_tokenizer(tokenizer)

            do_diag = collect_diagnostics and i == 0 and run == 0
            text, elapsed, step_mets, diag_data = generate_gravitational(
                model, tokenizer, prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
                device=device,
                collect_diagnostics=do_diag,
                temperature=args.temperature,
            )

            gen_only = text[len(prompt):]
            n_tokens = len(tokenizer.encode(gen_only))
            texts.append(text)
            times.append(elapsed)
            token_counts.append(n_tokens)

            for sm in step_mets:
                d = asdict(sm)
                d["prompt_idx"] = i
                d["run"] = run
                d["condition"] = label
                all_step_metrics.append(d)

            if do_diag:
                all_diagnostic_data.extend(diag_data)

    ppl = perplexity(model, tokenizer, texts, device)
    total_tokens = sum(token_counts)
    total_time = sum(times)

    escape_rates = [m["escape_rate"] for m in all_step_metrics]
    active_bodies_list = [m["active_bodies"] for m in all_step_metrics]

    return {
        "label":          label,
        "perplexity":     round(ppl, 4),
        "distinct_1":     round(distinct_n(texts, 1), 4),
        "distinct_2":     round(distinct_n(texts, 2), 4),
        "rep_rate_2":     round(repetition_rate(texts, 2), 4),
        "avg_length":     round(avg_length(texts), 2),
        "total_time_s":   round(total_time, 3),
        "ms_per_token":   round(total_time * 1000 / (total_tokens + 1e-8), 2),
        "mean_escape_rate":   round(float(np.mean(escape_rates)), 4) if escape_rates else None,
        "mean_active_bodies": round(float(np.mean(active_bodies_list)), 2) if active_bodies_list else None,
        "texts":          texts,
        "step_metrics":   all_step_metrics,
        "diagnostic_data": all_diagnostic_data,
    }


def _run_baseline(
    label: str,
    model,
    tokenizer,
    prompts: list[str],
    args,
    device: str,
    mode: str,   # "temperature" | "topp" | "typical"
    seed: int | None,
) -> dict:
    """Run one sampling baseline (temperature/top-p/typical). Returns metrics dict."""
    texts: list[str] = []
    times: list[float] = []
    token_counts: list[int] = []

    for i, prompt in enumerate(prompts):
        for run in range(args.runs):
            if seed is not None:
                np.random.seed(seed + i * 100 + run)
                torch.manual_seed(seed + i * 100 + run)

            if mode == "temperature":
                text, elapsed = generate_temperature(
                    model, tokenizer, prompt, args.max_tokens, args.temperature, device,
                )
            elif mode == "topp":
                text, elapsed = generate_topp(
                    model, tokenizer, prompt, args.max_tokens,
                    args.temperature, args.top_p, device,
                )
            elif mode == "typical":
                text, elapsed = generate_typical(
                    model, tokenizer, prompt, args.max_tokens, args.typical_p, device,
                )
            else:
                raise ValueError(f"Unknown baseline mode: {mode!r}")

            gen_only = text[len(prompt):]
            n_tokens = len(tokenizer.encode(gen_only))
            texts.append(text)
            times.append(elapsed)
            token_counts.append(n_tokens)

    total_tokens = sum(token_counts)
    total_time = sum(times)

    return {
        "label":        label,
        "perplexity":   round(perplexity(model, tokenizer, texts, device), 4),
        "distinct_1":   round(distinct_n(texts, 1), 4),
        "distinct_2":   round(distinct_n(texts, 2), 4),
        "rep_rate_2":   round(repetition_rate(texts, 2), 4),
        "avg_length":   round(avg_length(texts), 2),
        "total_time_s": round(total_time, 3),
        "ms_per_token": round(total_time * 1000 / (total_tokens + 1e-8), 2),
        "texts":        texts,
    }


def print_ablation_table(conditions: list[dict]) -> str:
    """Format an ablation comparison table. Returns the table string (also prints it)."""
    cols = [
        ("method",    "label",               "<", 28),
        ("ppl",       "perplexity",          ">",  8),
        ("dist-1",    "distinct_1",          ">",  7),
        ("dist-2",    "distinct_2",          ">",  7),
        ("rep-2",     "rep_rate_2",          ">",  7),
        ("ms/tok",    "ms_per_token",        ">",  8),
        ("esc_rate",  "mean_escape_rate",    ">",  9),
        ("bodies",    "mean_active_bodies",  ">",  7),
    ]
    header = "  " + "  ".join(f"{name:{align}{width}}" for name, _, align, width in cols)
    sep    = "  " + "  ".join("-" * width for _, _, _, width in cols)
    lines  = ["", "=" * (len(header) + 2), "  contextbodies ablation table",
              "=" * (len(header) + 2), header, sep]

    for cond in conditions:
        parts = []
        for name, key, align, width in cols:
            val = cond.get(key)
            if val is None:
                cell = "—"
            elif key == "label":
                cell = str(val)
            elif isinstance(val, float):
                cell = f"{val:.4f}" if key not in ("ms_per_token", "perplexity") else f"{val:.2f}"
            else:
                cell = str(val)
            parts.append(f"{cell:{align}{width}}")
        lines.append("  " + "  ".join(parts))

    lines += ["=" * (len(header) + 2), ""]
    table = "\n".join(lines)
    print(table)
    return table


def main() -> None:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Seed: {args.seed}")

    print(f"Loading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.prompts_file:
        prompts = Path(args.prompts_file).read_text().strip().splitlines()
        prompts = [p.strip() for p in prompts if p.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    G_local = args.G_local if args.G_local is not None else args.G_base
    print(f"Prompts: {len(prompts)}  max_tokens: {args.max_tokens}  runs: {args.runs}")
    print(f"G_local={G_local}  G_universe={args.G_universe}  temperature={args.temperature}\n")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    embedding_dim = model.get_input_embeddings().weight.shape[1]

    config = {
        "model": args.model, "num_prompts": len(prompts), "max_tokens": args.max_tokens,
        "runs": args.runs, "seed": args.seed, "temperature": args.temperature,
        "G_local": G_local, "G_universe": args.G_universe,
        "G_base": args.G_base if args.adaptive_g else None,
        "adaptive_g": args.adaptive_g,
        "escape_rate_target": args.escape_rate_target if args.adaptive_g else None,
        "mass_damping": args.mass_damping if args.adaptive_g else None,
        "adaptive_clustering": args.adaptive_clustering,
        "universe_path": args.universe_path, "universe_clusters": args.universe_clusters,
        "universe_mass": args.universe_mass, "local_bodies": args.local_bodies,
        "deterministic": args.deterministic, "device": device,
        "baselines": args.baselines, "top_p": args.top_p, "typical_p": args.typical_p,
    }

    # ── Universe ──────────────────────────────────────────────────────────
    universe: Universe | None = None
    if args.universe_path:
        print(f"Loading universe from {args.universe_path}...")
        universe = Universe.load(args.universe_path)
        print(f"  Loaded: {universe.n_bodies} bodies, dim={universe.centroids.shape[1]}")
    elif args.build_universe:
        print(f"Building universe ({args.universe_clusters} clusters, mass={args.universe_mass})...")
        builder = UniverseBuilder()
        universe = builder.build(model=model, n_clusters=args.universe_clusters,
                                 mass_scheme=args.universe_mass)
        universe_path = out_dir / "universe.npz"
        universe.save(universe_path)
        config["universe_path"] = str(universe_path)
        print(f"  Universe saved to {universe_path}")

    # ── IDF weights ───────────────────────────────────────────────────────
    idf_weights = None
    if not args.no_idf:
        print("Precomputing IDF weights...")
        _tmp = GravitationalSampler(
            body_store=ContextBodyStore(embedding_dim=embedding_dim, decay_interval=9999), G=1.0)
        _tmp.precompute_idf_weights(model)
        idf_weights = _tmp._idf_weights
        print(f"  IDF: min={idf_weights.min():.3f}  max={idf_weights.max():.3f}  "
              f"mean={idf_weights.mean():.3f}")

    # ── Baselines ─────────────────────────────────────────────────────────
    run_topp    = args.baselines in ("topp",    "all")
    run_typical = args.baselines in ("typical", "all")

    print("Running temperature baseline...")
    temp_result = _run_baseline("temperature", model, tokenizer, prompts,
                                args, device, "temperature", args.seed)
    print("  done.")

    topp_result: dict | None = None
    if run_topp:
        print(f"Running top-p baseline (p={args.top_p})...")
        topp_result = _run_baseline(f"top-p (p={args.top_p})", model, tokenizer,
                                    prompts, args, device, "topp", args.seed)
        print("  done.")

    typical_result: dict | None = None
    if run_typical:
        print(f"Running typical sampling (p={args.typical_p})...")
        typical_result = _run_baseline(f"typical (p={args.typical_p})", model, tokenizer,
                                       prompts, args, device, "typical", args.seed)
        print("  done.")

    # ── Temperature-only early exit ───────────────────────────────────────
    if args.temperature_only:
        print("\nTemperature-only mode: skipping gravitational sampling.")
        print_summary({"config": config, "temperature": temp_result, "gravitational": None})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_summary({"config": config, "temperature": temp_result, "gravitational": None})
        (out_dir / "summary.txt").write_text(buf.getvalue())
        (out_dir / "results.json").write_text(
            json.dumps({"config": config, "temperature": temp_result}, indent=2, default=str))
        print(f"Results saved to {out_dir}/")
        return

    # ── Ablation mode ─────────────────────────────────────────────────────
    if args.ablation:
        if universe is None:
            print("ERROR: --ablation requires --universe-path or --build-universe")
            import sys; sys.exit(1)

        all_conditions_map = {
            "real":         ("universe (real)",        universe,                                    True,  False),
            "shuffled":     ("shuffled centroids",     universe.shuffled(seed=args.seed),           True,  False),
            "random":       ("random centroids",       universe.with_random_centroids(seed=args.seed), True, False),
            "no_idf":       ("no IDF",                 universe,                                    False, False),
            "uniform_mass": ("uniform mass",           universe.with_uniform_mass(),                True,  False),
            "local_only":   ("local bodies only",      None,                                        True,  True),
            "combined":     ("universe + local",       universe,                                    True,  True),
        }
        keys = ([k.strip() for k in args.ablation_conditions.split(",")]
                if args.ablation_conditions else list(all_conditions_map.keys()))
        print(f"\nRunning ablation battery: {keys}\n")

        ablation_conditions: list[dict] = []
        for bl in [temp_result, topp_result, typical_result]:
            if bl:
                ablation_conditions.append({**bl, "mean_escape_rate": None,
                                             "mean_active_bodies": None, "rep_rate_2": bl.get("rep_rate_2")})

        all_step_metrics: list[dict] = []
        all_diag: list[dict] = []

        for key in keys:
            if key not in all_conditions_map:
                print(f"  WARNING: unknown ablation condition {key!r}, skipping"); continue
            label, univ_variant, use_idf, local_bodies = all_conditions_map[key]
            print(f"  [{key}] {label}...")
            cond = _run_one_condition(
                label=label, model=model, tokenizer=tokenizer, prompts=prompts,
                args=args, device=device, G_local=G_local, universe=univ_variant,
                use_idf=use_idf, local_bodies=local_bodies, idf_weights=idf_weights,
                seed=args.seed, collect_diagnostics=args.log_diagnostics,
            )
            ablation_conditions.append(cond)
            all_step_metrics.extend(cond.get("step_metrics", []))
            all_diag.extend(cond.get("diagnostic_data", []))
            print(f"    ppl={cond['perplexity']:.2f}  dist-1={cond['distinct_1']:.4f}  "
                  f"rep-2={cond['rep_rate_2']:.4f}  ms/tok={cond['ms_per_token']:.1f}")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_ablation_table(ablation_conditions)
        ablation_table = buf.getvalue()
        print(ablation_table)

        (out_dir / "ablation.txt").write_text(ablation_table)
        (out_dir / "ablation.json").write_text(json.dumps({
            "config": config,
            "conditions": [
                {k: v for k, v in c.items() if k not in ("texts", "step_metrics", "diagnostic_data")}
                for c in ablation_conditions
            ],
        }, indent=2, default=str))
        (out_dir / "ablation_steps.json").write_text(
            json.dumps(all_step_metrics, indent=2, default=str))
        if all_diag:
            (out_dir / "diagnostics.json").write_text(
                json.dumps(all_diag, indent=2, default=str))
        print(f"Ablation results saved to {out_dir}/")
        return

    # ── Standard single-condition run ─────────────────────────────────────
    print("Running gravitational sampling...")

    adaptive_g = AdaptiveG(
        G_base=args.G_base, escape_rate_target=args.escape_rate_target,
        mass_damping=args.mass_damping,
    ) if args.adaptive_g else None

    adaptive_dbscan = AdaptiveDBSCAN(
        target_bodies=args.target_bodies, eps_percentile=args.eps_percentile,
        min_samples=args.cluster_min_tokens, adjustment_rate=args.eps_adjustment_rate,
    ) if args.adaptive_clustering else None

    all_step_metrics: list[dict] = []
    all_diag: list[dict] = []
    grav_texts: list[str] = []
    grav_times: list[float] = []
    grav_token_counts: list[int] = []

    for i, prompt in enumerate(prompts):
        for run in range(args.runs):
            print(f"  prompt {i+1}/{len(prompts)}, run {run+1}/{args.runs}", end="\r")
            if args.seed is not None:
                np.random.seed(args.seed + i * 100 + run)
                torch.manual_seed(args.seed + i * 100 + run)

            store = ContextBodyStore(embedding_dim=embedding_dim, decay_interval=9999)
            sampler = GravitationalSampler(
                body_store=store, G=G_local, G_universe=args.G_universe,
                escape_threshold=args.escape_threshold,
                recency_decay_lambda=args.recency_decay,
                adaptive_g=adaptive_g, adaptive_dbscan=adaptive_dbscan,
                universe=universe, use_context_bodies=args.local_bodies,
                deterministic=args.deterministic, device=device,
                cluster_radius=args.cluster_radius,
                cluster_min_tokens=args.cluster_min_tokens,
            )
            sampler._idf_weights = idf_weights

            do_diag = args.log_diagnostics and i == 0 and run == 0
            if do_diag:
                sampler.set_tokenizer(tokenizer)

            text, elapsed, step_mets, diag_data = generate_gravitational(
                model, tokenizer, prompt, max_tokens=args.max_tokens,
                sampler=sampler, device=device, collect_diagnostics=do_diag,
                temperature=args.temperature,
            )

            gen_only = text[len(prompt):]
            n_tokens = len(tokenizer.encode(gen_only))
            grav_texts.append(text)
            grav_times.append(elapsed)
            grav_token_counts.append(n_tokens)

            for sm in step_mets:
                d = asdict(sm); d["prompt_idx"] = i; d["run"] = run
                all_step_metrics.append(d)
            all_diag.extend(diag_data)

    print("\nGravitational sampling complete.")
    grav_ppl = perplexity(model, tokenizer, grav_texts, device)
    grav_total_tokens = sum(grav_token_counts)
    grav_total_time = sum(grav_times)

    escape_rates = [m["escape_rate"] for m in all_step_metrics]
    active_bodies_list = [m["active_bodies"] for m in all_step_metrics]
    G_effs = [m["G_eff"] for m in all_step_metrics if m.get("G_eff") is not None]

    step_summary = {
        "mean_escape_rate":   round(float(np.mean(escape_rates)), 4) if escape_rates else None,
        "mean_active_bodies": round(float(np.mean(active_bodies_list)), 2) if active_bodies_list else None,
        "mean_G_eff":         round(float(np.mean(G_effs)), 4) if G_effs else None,
    }

    semantic_coverage: float | None = None
    if universe is not None:
        coverages = [universe.semantic_coverage(tokenizer.encode(t)) for t in grav_texts]
        semantic_coverage = round(float(np.mean(coverages)), 4)

    grav_results = {
        "perplexity":           round(grav_ppl, 4),
        "distinct_1":           round(distinct_n(grav_texts, 1), 4),
        "distinct_2":           round(distinct_n(grav_texts, 2), 4),
        "rep_rate_2":           round(repetition_rate(grav_texts, 2), 4),
        "avg_length":           round(avg_length(grav_texts), 2),
        "total_time_s":         round(grav_total_time, 3),
        "ms_per_token":         round(grav_total_time * 1000 / (grav_total_tokens + 1e-8), 2),
        "semantic_coverage":    semantic_coverage,
        "step_metrics_summary": step_summary,
        "texts":                grav_texts,
    }

    # ── Save ──────────────────────────────────────────────────────────────
    baselines_out = {"temperature": temp_result}
    if topp_result:    baselines_out["top_p"]   = topp_result
    if typical_result: baselines_out["typical"] = typical_result

    full_results = {
        "config": config, "baselines": baselines_out,
        "gravitational": grav_results, "temperature": temp_result,
    }
    (out_dir / "results.json").write_text(
        json.dumps(full_results, indent=2, default=str))
    (out_dir / "steps.json").write_text(
        json.dumps(all_step_metrics, indent=2, default=str))
    if all_diag:
        (out_dir / "diagnostics.json").write_text(
            json.dumps(all_diag, indent=2, default=str))
        print(f"  Diagnostic data saved to {out_dir}/diagnostics.json")

    print_summary({"config": config, "temperature": temp_result, "gravitational": grav_results})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_summary({"config": config, "temperature": temp_result, "gravitational": grav_results})
    (out_dir / "summary.txt").write_text(buf.getvalue())

    if topp_result or typical_result:
        all_conds = [{**temp_result, "mean_escape_rate": None, "mean_active_bodies": None}]
        if topp_result:    all_conds.append({**topp_result,    "mean_escape_rate": None, "mean_active_bodies": None})
        if typical_result: all_conds.append({**typical_result, "mean_escape_rate": None, "mean_active_bodies": None})
        all_conds.append({
            **grav_results, "label": "gravitational",
            "mean_escape_rate":    step_summary.get("mean_escape_rate"),
            "mean_active_bodies":  step_summary.get("mean_active_bodies"),
        })
        print("\nBaseline comparison:")
        print_ablation_table(all_conds)

    print(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    main()

"""
benchmark.py — compare gravitational sampling against temperature baseline.

Usage:
    python benchmark.py
    python benchmark.py --model gpt2-medium --max-tokens 150 --runs 3
    python benchmark.py --prompts-file my_prompts.txt --G 2.0 --temperature 0.8
    python benchmark.py --output results/run1

Outputs:
    <output-dir>/results.json   — full numeric results
    <output-dir>/summary.txt    — human-readable table
    <output-dir>/steps.json     — per-step gravitational metrics
"""

from __future__ import annotations

import argparse
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
) -> tuple[str, float, list[StepMetrics]]:
    """
    Gravitational sampling with per-step metric collection.
    Returns (text, elapsed_seconds, step_metrics).
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    token_embeddings = model.get_input_embeddings().weight
    embedding_dim = token_embeddings.shape[1]

    with torch.no_grad():
        context_embeddings = token_embeddings[input_ids[0]]

    sampler.initialize(context_embeddings, embedding_dim)

    generated = input_ids[0].tolist()
    step_metrics: list[StepMetrics] = []
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
                logits=logits,
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

        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break

    elapsed = time.perf_counter() - total_start
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, elapsed, step_metrics


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_summary(results: dict) -> None:
    cfg = results["config"]
    temp = results["temperature"]
    grav = results["gravitational"]

    print("\n" + "=" * 60)
    print("  contextbodies benchmark")
    print("=" * 60)
    print(f"  model          : {cfg['model']}")
    print(f"  prompts        : {cfg['num_prompts']}")
    print(f"  max tokens     : {cfg['max_tokens']}")
    print(f"  runs per prompt: {cfg['runs']}")
    print(f"  temperature    : {cfg['temperature']}")
    print(f"  G              : {cfg['G']}")
    print(f"  adaptive_g     : {cfg['adaptive_g']}")
    print()
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
    p.add_argument("--G", type=float, default=1.0,
                   help="Gravitational constant (default: 1.0)")
    p.add_argument("--escape-threshold", type=float, default=0.01,
                   help="Escape threshold (default: 0.01)")
    p.add_argument("--adaptive-g", action="store_true",
                   help="Enable AdaptiveG controller")
    p.add_argument("--recency-lambda", type=float, default=0.0,
                   help="Recency decay lambda (default: 0.0 = disabled)")
    p.add_argument("--dbscan-eps", type=float, default=0.3,
                   help="DBSCAN epsilon: cosine distance radius for cluster membership (default: 0.3)")
    p.add_argument("--dbscan-min-samples", type=int, default=3,
                   help="DBSCAN min_samples: tokens needed to form a cluster core (default: 3)")
    p.add_argument("--output", default="benchmark_results",
                   help="Output directory (default: benchmark_results)")
    p.add_argument("--device", default=None,
                   help="Device: cuda / cpu (auto-detected if omitted)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load model
    print(f"Loading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load prompts
    if args.prompts_file:
        prompts = Path(args.prompts_file).read_text().strip().splitlines()
        prompts = [p.strip() for p in prompts if p.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    print(f"Prompts: {len(prompts)}  |  max_tokens: {args.max_tokens}  |  runs: {args.runs}")
    print(f"G={args.G}  temperature={args.temperature}  adaptive_g={args.adaptive_g}")
    print(f"dbscan_eps={args.dbscan_eps}  dbscan_min_samples={args.dbscan_min_samples}\n")

    # Output dir
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # Temperature baseline
    # -------------------------------------------------------------------
    print("Running temperature baseline...")
    temp_texts: list[str] = []
    temp_times: list[float] = []
    temp_token_counts: list[int] = []

    for i, prompt in enumerate(prompts):
        for run in range(args.runs):
            print(f"  prompt {i+1}/{len(prompts)}, run {run+1}/{args.runs}", end="\r")
            text, elapsed = generate_temperature(
                model, tokenizer, prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                device=device,
            )
            generated_only = text[len(prompt):]
            n_tokens = len(tokenizer.encode(generated_only))
            temp_texts.append(text)
            temp_times.append(elapsed)
            temp_token_counts.append(n_tokens)

    print("\nTemperature baseline complete.")
    temp_ppl = perplexity(model, tokenizer, temp_texts, device)
    temp_total_tokens = sum(temp_token_counts)
    temp_total_time = sum(temp_times)

    temp_results = {
        "perplexity":   round(temp_ppl, 4),
        "distinct_1":   round(distinct_n(temp_texts, 1), 4),
        "distinct_2":   round(distinct_n(temp_texts, 2), 4),
        "avg_length":   round(avg_length(temp_texts), 2),
        "total_time_s": round(temp_total_time, 3),
        "ms_per_token": round(temp_total_time * 1000 / (temp_total_tokens + 1e-8), 2),
        "texts":        temp_texts,
    }

    # -------------------------------------------------------------------
    # Gravitational sampling
    # -------------------------------------------------------------------
    print("Running gravitational sampling...")
    grav_texts: list[str] = []
    grav_times: list[float] = []
    grav_token_counts: list[int] = []
    all_step_metrics: list[dict] = []

    embedding_dim = model.get_input_embeddings().weight.shape[1]

    adaptive_g = AdaptiveG(G_base=args.G) if args.adaptive_g else None

    for i, prompt in enumerate(prompts):
        for run in range(args.runs):
            print(f"  prompt {i+1}/{len(prompts)}, run {run+1}/{args.runs}", end="\r")

            store = ContextBodyStore(embedding_dim=embedding_dim, decay_interval=9999)
            sampler = GravitationalSampler(
                body_store=store,
                G=args.G,
                escape_threshold=args.escape_threshold,
                recency_decay_lambda=args.recency_lambda,
                adaptive_g=adaptive_g,
                device=device,
                dbscan_eps=args.dbscan_eps,
                dbscan_min_samples=args.dbscan_min_samples,
            )

            text, elapsed, step_mets = generate_gravitational(
                model, tokenizer, prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
                device=device,
            )

            generated_only = text[len(prompt):]
            n_tokens = len(tokenizer.encode(generated_only))
            grav_texts.append(text)
            grav_times.append(elapsed)
            grav_token_counts.append(n_tokens)

            for sm in step_mets:
                d = asdict(sm)
                d["prompt_idx"] = i
                d["run"] = run
                all_step_metrics.append(d)

    print("\nGravitational sampling complete.")
    grav_ppl = perplexity(model, tokenizer, grav_texts, device)
    grav_total_tokens = sum(grav_token_counts)
    grav_total_time = sum(grav_times)

    # Step metrics summary
    escape_rates = [m["escape_rate"] for m in all_step_metrics]
    active_bodies = [m["active_bodies"] for m in all_step_metrics]
    G_effs = [m["G_eff"] for m in all_step_metrics if m["G_eff"] is not None]

    step_summary = {
        "mean_escape_rate":   round(float(np.mean(escape_rates)), 4) if escape_rates else None,
        "mean_active_bodies": round(float(np.mean(active_bodies)), 2) if active_bodies else None,
        "mean_G_eff":         round(float(np.mean(G_effs)), 4) if G_effs else None,
    }

    grav_results = {
        "perplexity":          round(grav_ppl, 4),
        "distinct_1":          round(distinct_n(grav_texts, 1), 4),
        "distinct_2":          round(distinct_n(grav_texts, 2), 4),
        "avg_length":          round(avg_length(grav_texts), 2),
        "total_time_s":        round(grav_total_time, 3),
        "ms_per_token":        round(grav_total_time * 1000 / (grav_total_tokens + 1e-8), 2),
        "step_metrics_summary": step_summary,
        "texts":               grav_texts,
    }

    # -------------------------------------------------------------------
    # Assemble and save
    # -------------------------------------------------------------------
    config = {
        "model":       args.model,
        "num_prompts": len(prompts),
        "max_tokens":  args.max_tokens,
        "runs":        args.runs,
        "temperature": args.temperature,
        "G":           args.G,
        "adaptive_g":  args.adaptive_g,
        "device":      device,
    }

    full_results = {
        "config":        config,
        "temperature":   temp_results,
        "gravitational": grav_results,
    }

    (out_dir / "results.json").write_text(json.dumps(full_results, indent=2))
    (out_dir / "steps.json").write_text(json.dumps(all_step_metrics, indent=2))

    # Print summary
    print_summary(full_results)

    # Save text summary
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_summary(full_results)
    (out_dir / "summary.txt").write_text(buf.getvalue())

    print(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    main()

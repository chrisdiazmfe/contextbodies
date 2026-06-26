from __future__ import annotations

import torch
import torch.nn as nn

from gravitational_sampler import GravitationalSampler


def generate(
    model: nn.Module,
    tokenizer,
    prompt: str,
    sampler: GravitationalSampler,
    max_tokens: int = 200,
) -> str:
    """
    Generate text using gravitational sampling in place of temperature sampling.

    At each step:
        1. Run the model forward pass to get logits.
        2. Delegate to GravitationalSampler.sample() which applies the
           gravitational field and returns the next token id.
        3. Append the token and continue until EOS or max_tokens.

    Args:
        model       — any HuggingFace-compatible causal LM
        tokenizer   — matching tokenizer
        prompt      — input text
        sampler     — initialized GravitationalSampler
        max_tokens  — maximum tokens to generate

    Returns:
        decoded string of the full sequence (prompt + generated tokens)
    """
    device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # embedding and output weight matrices (retrieved once, reused every step)
    token_embeddings = model.get_input_embeddings().weight   # [vocab, D]
    weight_matrix = model.get_output_embeddings().weight     # [vocab, D]
    embedding_dim = token_embeddings.shape[1]

    # initialize sampler with prompt context
    with torch.no_grad():
        context_embeddings = token_embeddings[input_ids[0]]  # [prompt_len, D]

    sampler.initialize(context_embeddings, embedding_dim)

    generated = input_ids[0].tolist()

    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(torch.tensor([generated], device=device))
            logits = outputs.logits[0, -1, :]   # [vocab_size] — last position only

        next_token = sampler.sample(
            logits=logits,
            token_embeddings=token_embeddings,
            weight_matrix=weight_matrix,
        )

        generated.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)

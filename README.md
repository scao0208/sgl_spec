# Speculative decoding by vllm 0.15.0

This repositary provides guidance to deploy classic speculative decoding experiments following the paper "Accelerating Large Language Model Decoding
with Speculative Sampling"(https://arxiv.org/pdf/2302.01318) and "Training-Free Loosely Speculative Decoding: Accepting Semantically Correct Drafts Beyond Exact Match"(https://arxiv.org/abs/2511.22972).

## Project Overview

This is a benchmarking toolkit for evaluating **vLLM speculative decoding** performance. It measures speedup achieved when using a smaller "draft" model to speculate tokens for a larger "target" model.

## Running Benchmarks

**Baseline (no speculative decoding):**
```bash
python scripts/benchmark_baseline.py \
    --target-model meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 4 \
    --num-prompts 20 \
    --max-tokens 256 \
    --max-num-seqs 2 \
    --gsm8k /path/to/gsm8k
```

**Speculative decoding with draft model:**
```bash
python scripts/benchmark_draft_model.py \
    --target-model /path/to/70B-model \
    --draft-model meta-llama/Llama-3.1-8B-Instruct \
    --num-speculative-tokens 24 \
    --tensor-parallel-size 4 \
    --draft-tensor-parallel-size 4 \
    --use-tree \
    --max-num-seqs 2 \
    --gsm8k /path/to/gsm8k
```

**Training-Free Loosely Speculative Decoding:**
```bash
python scripts/benchmark_draft_model.py \
    --target-model /path/to/70B-model \
    --draft-model meta-llama/Llama-3.1-8B-Instruct \
    --num-speculative-tokens 15 \
    --tensor-parallel-size 4 \
    --draft-tensor-parallel-size 4 \
    --max-num-seqs 2 \
    --max-tokens 512 \
    --gsm8k /path/to/gsm8k \
    --fly-enabled
```

**Key flags:**
- `--enforce-eager`: Disable CUDA graphs (helps with stability)
- `--use-tree`: Enable tree-based speculation with 512-node structure
- `--gsm8k`: Path to GSM8K dataset (expects parquet at `{path}/main/test-00000-of-00001.parquet`)

## FLy Algorithm (Training-Free Loosely Speculative Decoding)

This project also explores the FLy algorithm ([arXiv:2511.22972](https://arxiv.org/abs/2511.22972)), which relaxes standard speculative decoding's strict exact-match verification to accept semantically valid draft tokens.

**Problem:** Standard SPD rejects a draft token the moment it differs from the target model's prediction, even if the draft is semantically correct (e.g., "When" vs "After" to start a sentence). This wastes useful tokens and limits speedup.

**Core idea:** LLMs exhibit self-corrective behavior when conditioned on genuinely wrong tokens, but continue smoothly when conditioned on semantically valid alternatives. FLy exploits this property through a two-tier verification mechanism:

1. **Entropy-Level Gate:** At each mismatch position, compute the normalized entropy of the target model's distribution from its already-available logits. If entropy is low (h < θ, default θ=0.3), the target is confident and the mismatch is likely a real error (e.g., wrong digit in arithmetic) — reject immediately via standard SPD. If entropy is high, multiple tokens are plausible — proceed to tier 2.

2. **Token-Level Deferred Window:** For high-entropy mismatches, monitor the next W tokens (default W=6). If no further mismatches appear in that window, the target model is continuing smoothly from the draft token, indicating semantic validity — accept the mismatch. If another mismatch appears, the target is course-correcting — reject.

**Multi-Level Acceleration (MLA):** Since FLy dramatically increases mean accepted tokens (τ ≈ 12 vs ≈ 4 in standard SPD), the drafter must propose more tokens per round (K=15-25), making draft cost the bottleneck. MLA accelerates the drafter itself using prompt lookup decoding (n-gram retrieval), reducing draft latency by ~20%.

## Architecture

Two benchmark scripts with a shared structure:
1. `benchmark_baseline.py` - Pure target model inference, establishes baseline throughput
2. `benchmark_draft_model.py` - Speculative decoding with draft model, extracts acceptance metrics

**Key metrics extracted:**
- Throughput (tokens/s)
- Time per token (ms) - primary comparison metric
- Acceptance rate by position (speculative only)
- Draft efficiency = accepted_tokens / proposed_tokens (speculative only)

**Tree structures:** The draft benchmark includes hardcoded Monte Carlo simulation trees (`mc_sim_8b_12`, `mc_sim_8b_512`) for tree-based speculative decoding. The 512-node tree spans 24 levels.

**Metrics extraction:** Uses vLLM's internal counters:
- `vllm:spec_decode_num_drafts`
- `vllm:spec_decode_num_draft_tokens`
- `vllm:spec_decode_num_accepted_tokens`
- `vllm:spec_decode_num_accepted_tokens_per_pos`
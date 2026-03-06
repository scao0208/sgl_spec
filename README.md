# Speculative Decoding with SGLang

This repository provides benchmarking tools for speculative decoding experiments with [SGLang](https://github.com/sgl-project/sglang), following the papers "Accelerating Large Language Model Decoding with Speculative Sampling" (https://arxiv.org/pdf/2302.01318) and "Training-Free Loosely Speculative Decoding: Accepting Semantically Correct Drafts Beyond Exact Match" (https://arxiv.org/abs/2511.22972).

## Project Overview

This is a benchmarking toolkit for evaluating **SGLang speculative decoding** performance. It supports two speculative algorithms:
- **EAGLE3**: Dynamic tree drafting with EAGLE3-specific draft models and confidence-based reranking
- **STANDALONE**: Draft-model speculation using any compatible smaller model (chain or tree), with optional FLy verification

## Setup

```bash
conda activate sgl-spec
```

Key dependency: `sglang` (from `thirdparty/sglang` submodule). For STANDALONE mode with a patched sglang:
```bash
export PYTHONPATH="/path/to/thirdparty/sglang/python:${PYTHONPATH:-}"
```

## Running Benchmarks

All scripts must run from `scripts/` due to relative imports.

**Baseline (no speculative decoding):**
```bash
cd scripts
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmark_baseline.py \
    --model /path/to/Llama-3.1-70B-Instruct \
    --tp-size 4 --max-tokens 512 --num-prompts 164 \
    --dataset /path/to/humaneval
```

**EAGLE3 speculative decoding:**
```bash
cd scripts
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmark_sglang_eagle3.py \
    --target-model meta-llama/Llama-3.3-70B-Instruct \
    --draft-model /path/to/EAGLE3-LLaMA3.1-Instruct-70B \
    --tp-size 4 --num-steps 5 --eagle-topk 8 --num-draft-tokens 64 \
    --dataset /path/to/gsm8k --num-prompts 50
```

**STANDALONE chain drafting with FLy:**
```bash
cd scripts
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmark_sglang_eagle3.py \
    --target-model /path/to/70B-model \
    --draft-model /path/to/8B-model \
    --speculative-algorithm STANDALONE \
    --tp-size 4 --num-steps 22 --num-draft-tokens 23 \
    --fly-enabled --fly-entropy-threshold 0.3 --fly-window-size 6 \
    --dataset /path/to/humaneval --num-prompts 164
```

**STANDALONE tree drafting with FLy:**
```bash
cd scripts
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmark_sglang_eagle3.py \
    --target-model /path/to/70B-model \
    --draft-model /path/to/8B-model \
    --speculative-algorithm STANDALONE \
    --tp-size 4 --eagle-topk 5 --num-steps 22 --num-draft-tokens 512 \
    --fly-enabled --dataset /path/to/humaneval
```

**Key parameters:**
- `--num-steps`: Draft depth (default: 5)
- `--eagle-topk`: Branching factor per step (EAGLE3 default: 8, STANDALONE default: 1)
- `--num-draft-tokens`: Max parallel verification tokens (EAGLE3 default: 63, STANDALONE default: num_steps+1)
- `--profile-attention`: Instrument RadixAttention for CUDA event timing

Pre-built shell scripts are available in `scripts/` for specific experiment configurations (e.g., `run_baseline_he.sh`, `run_fly_chain_he.sh`).

## FLy Algorithm (Training-Free Loosely Speculative Decoding)

This project also explores the FLy algorithm ([arXiv:2511.22972](https://arxiv.org/abs/2511.22972)), which relaxes standard speculative decoding's strict exact-match verification to accept semantically valid draft tokens.

**Problem:** Standard SPD rejects a draft token the moment it differs from the target model's prediction, even if the draft is semantically correct (e.g., "When" vs "After" to start a sentence). This wastes useful tokens and limits speedup.

**Core idea:** LLMs exhibit self-corrective behavior when conditioned on genuinely wrong tokens, but continue smoothly when conditioned on semantically valid alternatives. FLy exploits this property through a two-tier verification mechanism:

1. **Entropy-Level Gate:** At each mismatch position, compute the normalized entropy of the target model's distribution from its already-available logits. If entropy is low (h < θ, default θ=0.3), the target is confident and the mismatch is likely a real error — reject immediately. If entropy is high, multiple tokens are plausible — proceed to tier 2.

2. **Token-Level Deferred Window:** For high-entropy mismatches, monitor the next W tokens (default W=6). If no further mismatches appear in that window, the target model is continuing smoothly — accept the mismatch. If another mismatch appears, the target is course-correcting — reject.

## Architecture

Two benchmark scripts:
1. `benchmark_baseline.py` — Pure target model inference via SGLang, establishes baseline throughput
2. `benchmark_sglang_eagle3.py` — Speculative decoding (EAGLE3 or STANDALONE) with optional FLy verification and attention profiling

**Key metrics extracted from SGLang's per-request `meta_info`:**
- Throughput (tokens/s) and time per token (ms)
- `spec_verify_ct`: number of verification rounds
- `spec_accepted_tokens`: total accepted draft tokens
- Avg acceptance length = total_tokens / verify_rounds
- FLy metrics: deferral counts, acceptance rate, entropy gate rejections

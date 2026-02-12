# CLAUDE.md - Scenario B: Fixed mc_sim_8b_512 Tree

## Overview

This worktree implements **Scenario B**: benchmarking fixed mc_sim_8b_512 tree (512 nodes, 24 depth levels) speculative decoding with the ea_attn_exp two-stage tree attention kernel.

The mc_sim_8b_512 tree is a pre-computed Monte Carlo simulation tree optimized for 8B draft models. Unlike EAGLE's dynamic trees, this structure is static and used with a standard draft model proposer.

## Branch

`scenario-b-fixed-tree` (git worktree of vllm_spec)

## Setup

```bash
conda activate vllm-spec
cd scripts
```

## Files

- `choices.py` - Contains mc_sim_8b_512 (512 nodes, 24 depth) and mc_sim_8b_12 (~70 nodes)
- `eagle_tree_attention.py` - ea_attn_exp Triton kernel (copied from ea_attn_exp project)
- `benchmark_fixed_tree.py` - Benchmark script comparing vLLM unified vs ea_attn_exp kernel
- `test_kernel_correctness.py` - Verify ea_attn_exp kernel matches PyTorch reference

## Running Benchmarks

```bash
# vLLM unified attention (baseline)
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmark_fixed_tree.py \
    --target-model meta-llama/Llama-3.3-70B-Instruct \
    --draft-model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 4 \
    --dataset /path/to/gsm8k \
    --enforce-eager

# ea_attn_exp tree kernel
CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmark_fixed_tree.py \
    --target-model meta-llama/Llama-3.3-70B-Instruct \
    --draft-model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 4 \
    --use-eagle-kernel \
    --dataset /path/to/gsm8k \
    --enforce-eager
```

## Correctness Test

```bash
python test_kernel_correctness.py
```

Tests:
1. Tree mask structure (root visibility, ancestor chains, sibling isolation)
2. Small tree subset (64 nodes) vs PyTorch reference
3. Full 512-node tree with past_len = 100, 500, 1000
4. GQA configuration (32 Q heads / 8 KV heads, LLaMA 3 style)

## Metrics Collected

- `avg_acceptance_length` - Mean accepted tokens per speculation round
- `time_per_token_ms` - End-to-end latency per output token
- `draft_efficiency` - Fraction of draft tokens accepted
- `throughput_tokens_per_s` - Output tokens per second

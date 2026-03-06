#!/bin/bash
# Benchmark: SGLang baseline (no speculative decoding)
# Model: Llama-3.1-70B-Instruct | TP=4 | BS=2
# Dataset: HumanEval (164 prompts)

set -euo pipefail
cd "$(dirname "$0")"

eval "$(conda shell.bash hook)"
conda activate vllm-spec

export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL="/home/dataset_model/model/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b"
DATASET="/home/dataset_model/dataset/openai_humaneval/openai_humaneval"

python benchmark_baseline.py \
    --model "$MODEL" \
    --tp-size 4 \
    --max-tokens 512 \
    --num-prompts 164 \
    --max-num-seqs 2 \
    --dtype float16 \
    --dataset "$DATASET" \
    --log-level error \
    2>&1 | tee base_he.log

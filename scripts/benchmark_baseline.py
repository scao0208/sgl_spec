"""
Benchmark baseline (no speculative decoding) with SGLang.

Usage:
    cd scripts
    conda activate vllm-spec

    CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmark_baseline.py \
        --model /path/to/Llama-3.1-70B-Instruct \
        --tp-size 4 --max-tokens 512 --num-prompts 164 \
        --dataset /path/to/humaneval
"""

import argparse
import json
import time

import sglang as sgl
from datasets import detect_and_load


def main():
    parser = argparse.ArgumentParser(description="Baseline benchmark (no spec decoding)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--log-level", type=str, default="error")
    args = parser.parse_args()

    prompts = detect_and_load(args.dataset, args.num_prompts)
    print(f"Running with {len(prompts)} prompts")
    print(f"\nModel: {args.model}")
    print(f"TP size: {args.tp_size}")
    print(f"Max batch size: {args.max_num_seqs}")

    engine_kwargs = dict(
        model_path=args.model,
        tp_size=args.tp_size,
        max_running_requests=args.max_num_seqs,
        dtype=args.dtype,
        log_level=args.log_level,
        watchdog_timeout=1800,
    )
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static

    engine = sgl.Engine(**engine_kwargs)

    sampling_params = {
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    print(f"\nGenerating up to {args.max_tokens} tokens per prompt...")
    start_time = time.perf_counter()
    outputs = engine.generate(prompts, sampling_params)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    total_output_tokens = sum(o["meta_info"]["completion_tokens"] for o in outputs)
    throughput = total_output_tokens / total_time
    time_per_token_ms = (total_time / total_output_tokens) * 1000 if total_output_tokens > 0 else 0

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS (SGLang Baseline - No Speculation)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"TP size: {args.tp_size}")
    print(f"Max batch size: {args.max_num_seqs}")
    print(f"Num prompts: {len(prompts)}")
    print("-" * 70)
    print(f"Total time: {total_time:.2f}s")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print(f"Time per token: {time_per_token_ms:.4f} ms")
    print("=" * 70)

    # Sample outputs
    print("\n--- Sample outputs ---")
    for i, output in enumerate(outputs[:3]):
        prompt_preview = prompts[i][:80] + "..." if len(prompts[i]) > 80 else prompts[i]
        output_preview = output["text"][:200] + "..." if len(output["text"]) > 200 else output["text"]
        print(f"\nPrompt {i}: {prompt_preview}")
        print(f"Output: {output_preview}")

    # JSON metrics
    metrics = {
        "framework": "sglang",
        "mode": "baseline",
        "model": args.model,
        "total_time_s": total_time,
        "total_tokens": total_output_tokens,
        "throughput_tokens_per_s": throughput,
        "num_prompts": len(prompts),
        "time_per_token_ms": time_per_token_ms,
    }
    print(f"\nJSON metrics: {json.dumps(metrics)}")

    engine.shutdown()


if __name__ == "__main__":
    main()

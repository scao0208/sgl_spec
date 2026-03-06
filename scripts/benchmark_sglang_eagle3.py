"""
Benchmark EAGLE3 speculative decoding with SGLang (dynamic tree).

SGLang implements EAGLE3's dynamic tree building at runtime using
confidence-based reranking, unlike vLLM which only supports static trees.

Usage:
    cd scripts
    conda activate vllm-spec

    # Baseline EAGLE3 (FlashInfer attention):
    CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmark_sglang_eagle3.py \
        --target-model meta-llama/Llama-3.3-70B-Instruct \
        --draft-model /path/to/EAGLE3-LLaMA3.1-Instruct-70B \
        --tp-size 4 \
        --num-steps 5 \
        --eagle-topk 8 \
        --num-draft-tokens 64 \
        --dataset /path/to/gsm8k \
        --num-prompts 50

    # Profile attention time during EAGLE3 verification:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmark_sglang_eagle3.py \
        --target-model meta-llama/Llama-3.3-70B-Instruct \
        --draft-model /path/to/EAGLE3-LLaMA3.1-Instruct-70B \
        --tp-size 4 --num-draft-tokens 64 --profile-attention \
        --dataset /path/to/gsm8k --num-prompts 50
"""

import argparse
import json
import os
import sys
import tempfile
import time

import sglang as sgl
from datasets import detect_and_load

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_PROMPTS = [
    "Write a Python function to implement binary search.",
    "Explain the concept of quantum entanglement in simple terms.",
    "What are the main differences between Python and JavaScript?",
    "Write a short story about a robot learning to paint.",
    "Explain how neural networks learn through backpropagation.",
    "What is the time complexity of quicksort and why?",
    "Describe the process of photosynthesis step by step.",
    "Write a bash script to find all files larger than 100MB.",
    "Explain the CAP theorem in distributed systems.",
    "What are the SOLID principles in software engineering?",
]


# ============================================================================
# Attention profiling (runs in scheduler subprocess)
# ============================================================================

def _apply_profiling_patch(profile_file):
    """Monkey-patch RadixAttention.forward to collect CUDA event timing.

    Called inside the scheduler subprocess. Instruments layer 0 only to
    measure per-call attention time without double-counting across layers.
    Uses deferred CUDA event timing (no sync during benchmark).

    Writes JSON to profile_file on process exit:
        verify_extend_times_ms: list of per-call ms during verification
        other_extend_times_ms: list of per-call ms during other modes
        verify_extend_q_tokens: list of Q token counts per verify call
    """
    import atexit
    import torch

    from sglang.srt.layers.radix_attention import RadixAttention

    _events = []  # (start_event, end_event, is_verify, q_tokens)
    _orig_forward = RadixAttention.forward

    def _profiled_forward(self, q, k, v, forward_batch, save_kv_cache=True, **kwargs):
        # Only instrument layer 0 to avoid N_layers x counting
        if self.layer_id != 0:
            return _orig_forward(self, q, k, v, forward_batch, save_kv_cache, **kwargs)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = _orig_forward(self, q, k, v, forward_batch, save_kv_cache, **kwargs)
        end.record()

        is_verify = (hasattr(forward_batch, 'forward_mode') and
                     forward_batch.forward_mode.is_target_verify())
        _events.append((start, end, is_verify, q.shape[0]))

        return result

    RadixAttention.forward = _profiled_forward

    def _write_results():
        try:
            import torch as _torch
            _torch.cuda.synchronize()
        except Exception:
            pass
        verify_times = []
        other_times = []
        verify_q_tokens = []
        for start, end, is_verify, qt in _events:
            try:
                ms = start.elapsed_time(end)
            except Exception:
                continue
            if is_verify:
                verify_times.append(ms)
                verify_q_tokens.append(qt)
            else:
                other_times.append(ms)
        data = {
            "verify_extend_times_ms": verify_times,
            "other_extend_times_ms": other_times,
            "verify_extend_q_tokens": verify_q_tokens,
        }
        try:
            with open(profile_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[PROFILE] Error writing results: {e}", flush=True)

    atexit.register(_write_results)


# ============================================================================
# Scheduler subprocess wrapper (must be module-level for pickling)
# ============================================================================

def _patched_run_scheduler_process(server_args, port_args, gpu_id, tp_rank,
                                   moe_ep_rank, pp_rank, dp_rank, pipe_writer):
    """Scheduler process wrapper that applies monkey-patches before starting.

    Reads configuration from environment variables:
        SGLANG_BENCH_SCRIPTS_DIR: path to scripts/ for imports
        SGLANG_BENCH_FLAGS: comma-separated flags (profile)
        SGLANG_ATTN_PROFILE_FILE: path for profiling JSON output
    """
    scripts_dir = os.environ.get("SGLANG_BENCH_SCRIPTS_DIR", "")
    if scripts_dir and scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    flags = os.environ.get("SGLANG_BENCH_FLAGS", "")
    profile_file = os.environ.get("SGLANG_ATTN_PROFILE_FILE", "")

    if "profile" in flags and profile_file:
        _apply_profiling_patch(profile_file)
        print(f"[PROFILE] Attention profiling enabled in TP rank {tp_rank}", flush=True)

    from sglang.srt.managers.scheduler import run_scheduler_process
    return run_scheduler_process(server_args, port_args, gpu_id, tp_rank,
                                 moe_ep_rank, pp_rank, dp_rank, pipe_writer)


# ============================================================================
# CLI and benchmark logic
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark EAGLE3 speculative decoding with SGLang"
    )
    parser.add_argument("--target-model", type=str, required=True)
    parser.add_argument("--draft-model", type=str, required=True)
    parser.add_argument("--speculative-algorithm", type=str, default="STANDALONE",
                        choices=["STANDALONE", "EAGLE3"],
                        help="Speculative decoding algorithm (default: STANDALONE). "
                             "STANDALONE uses the draft model as a linear chain (topk=1). "
                             "EAGLE3 uses dynamic tree with EAGLE3-specific draft model.")
    parser.add_argument("--tp-size", type=int, default=1)
    # STANDALONE defaults: num_steps controls draft tokens per round, topk=1, num_draft_tokens=num_steps+1
    # EAGLE3 defaults: num_steps=5, topk=8, num_draft_tokens=63
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Draft steps per round (default: auto). "
                             "STANDALONE: number of draft tokens. EAGLE3: tree depth.")
    parser.add_argument("--eagle-topk", type=int, default=None,
                        help="Branching factor per step (default: auto). "
                             "EAGLE3 default: 8, STANDALONE default: 1 (use >1 for tree drafting).")
    parser.add_argument("--num-draft-tokens", type=int, default=None,
                        help="Max parallel verification tokens (default: auto). "
                             "STANDALONE: auto = num_steps+1. EAGLE3: tree width.")
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=48,
                        help="Max running requests / batch size (default: 48)")
    parser.add_argument("--dtype", type=str, default="float16",
                        help="Model dtype (default: float16)")
    parser.add_argument("--disable-cuda-graph", action="store_true",
                        help="Disable CUDA graph capture")
    parser.add_argument("--cuda-graph-max-bs", type=int, default=8)
    parser.add_argument("--mem-fraction-static", type=float, default=None,
                        help="Fraction of GPU memory for static allocation (model + KV cache)")
    parser.add_argument("--log-level", type=str, default="error",
                        help="SGLang log level (default: error). Use 'info' for scheduling details.")
    parser.add_argument("--profile-attention", action="store_true",
                        help="Profile attention kernel time during tree verification. "
                             "Instruments RadixAttention.forward (layer 0) with CUDA event timing.")
    parser.add_argument("--max-context-tokens", type=int, default=None,
                        help="Max prompt tokens for dataset filtering (longbench only)")
    parser.add_argument("--attention-backend", type=str, default=None,
                        choices=["triton", "flashinfer"],
                        help="Force attention backend (default: auto-detect)")
    parser.add_argument("--fly-enabled", action="store_true",
                        help="Enable FLy (Training-Free Loosely Speculative Decoding) "
                             "for greedy verification")
    parser.add_argument("--fly-entropy-threshold", type=float, default=0.3,
                        help="Normalized entropy threshold for FLy deferral (default: 0.3)")
    parser.add_argument("--fly-window-size", type=int, default=6,
                        help="Lookahead window size for FLy verification (default: 6)")
    parser.add_argument("--fly-use-cuda-kernel", action="store_true",
                        help="Use CUDA kernel for FLy verification (requires recompiled sgl-kernel)")
    return parser.parse_args()


def read_profile_results(profile_file, total_time, total_verify_ct):
    """Read and display attention profiling results from subprocess JSON."""
    for _ in range(30):
        if os.path.exists(profile_file):
            time.sleep(0.5)
            break
        time.sleep(0.2)

    if not os.path.exists(profile_file):
        print(f"[PROFILE] Warning: profile file not found at {profile_file}")
        return None

    try:
        with open(profile_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[PROFILE] Warning: could not read profile file: {e}")
        return None

    verify_times = data.get("verify_extend_times_ms", [])
    other_times = data.get("other_extend_times_ms", [])
    verify_q_tokens = data.get("verify_extend_q_tokens", [])

    if not verify_times:
        print("[PROFILE] No verify attention calls recorded.")
        print("[PROFILE] Possible causes:")
        print("  - CUDA graph captured the attention calls (try --disable-cuda-graph)")
        print("  - The subprocess didn't apply the profiling patch")
        return None

    total_verify_attn_ms = sum(verify_times)
    avg_verify_attn_ms = total_verify_attn_ms / len(verify_times)
    total_other_attn_ms = sum(other_times) if other_times else 0
    avg_q_tokens = sum(verify_q_tokens) / len(verify_q_tokens) if verify_q_tokens else 0

    total_time_ms = total_time * 1000
    attn_pct_of_total = (total_verify_attn_ms / total_time_ms) * 100 if total_time_ms > 0 else 0

    attn_per_iteration_ms = total_verify_attn_ms / total_verify_ct if total_verify_ct > 0 else 0
    iter_time_ms = (total_time / total_verify_ct) * 1000 if total_verify_ct > 0 else 0
    attn_pct_of_iteration = (attn_per_iteration_ms / iter_time_ms) * 100 if iter_time_ms > 0 else 0

    print("\n" + "-" * 70)
    print("ATTENTION PROFILING (layer 0 only):")
    print(f"  Verify attention calls: {len(verify_times)}")
    print(f"  Other attention calls:  {len(other_times)}")
    print(f"  Avg verify attn time:   {avg_verify_attn_ms:.4f} ms  (layer 0)")
    print(f"  Total verify attn time: {total_verify_attn_ms:.2f} ms (layer 0)")
    print(f"  Avg Q tokens per call:  {avg_q_tokens:.0f}")
    print(f"  Attn per iteration:     {attn_per_iteration_ms:.4f} ms (layer 0)")
    print(f"  Iteration time:         {iter_time_ms:.4f} ms")
    print(f"  Attn % of iteration:    {attn_pct_of_iteration:.3f}%  (layer 0)")
    print(f"  Attn % of total time:   {attn_pct_of_total:.3f}%  (layer 0)")

    return {
        "verify_attn_calls": len(verify_times),
        "other_attn_calls": len(other_times),
        "avg_verify_attn_ms": avg_verify_attn_ms,
        "total_verify_attn_ms": total_verify_attn_ms,
        "avg_q_tokens": avg_q_tokens,
        "attn_per_iteration_ms": attn_per_iteration_ms,
        "attn_pct_of_iteration": attn_pct_of_iteration,
        "attn_pct_of_total": attn_pct_of_total,
    }


def main():
    args = parse_args()

    need_patch = args.profile_attention

    # Set up environment for subprocess communication
    profile_file = None
    if need_patch:
        os.environ["SGLANG_BENCH_SCRIPTS_DIR"] = _SCRIPTS_DIR

        flags = []
        if args.profile_attention:
            profile_file = tempfile.mktemp(suffix=".json", prefix="sglang_attn_profile_")
            os.environ["SGLANG_ATTN_PROFILE_FILE"] = profile_file
            flags.append("profile")
            print(f"[PROFILE] Attention profiling enabled, output: {profile_file}")
        os.environ["SGLANG_BENCH_FLAGS"] = ",".join(flags)

    # Load prompts
    if args.dataset:
        prompts = detect_and_load(args.dataset, args.num_prompts,
                                  max_tokens=args.max_context_tokens)
    else:
        prompts = SAMPLE_PROMPTS
        if args.num_prompts is not None:
            prompts = prompts[:args.num_prompts]
    print(f"Running with {len(prompts)} prompts")

    # Resolve speculative decoding parameters based on algorithm
    algo = args.speculative_algorithm
    if algo == "STANDALONE":
        num_steps = args.num_steps if args.num_steps is not None else 5
        eagle_topk = args.eagle_topk if args.eagle_topk is not None else 1
        num_draft_tokens = args.num_draft_tokens if args.num_draft_tokens is not None else num_steps + 1
    else:  # EAGLE3
        num_steps = args.num_steps if args.num_steps is not None else 5
        eagle_topk = args.eagle_topk if args.eagle_topk is not None else 8
        num_draft_tokens = args.num_draft_tokens if args.num_draft_tokens is not None else 63

    print(f"\nTarget model: {args.target_model}")
    print(f"Draft model: {args.draft_model}")
    print(f"TP size: {args.tp_size}")
    print(f"Algorithm: {algo}")
    fly_info = ""
    if args.fly_enabled:
        kernel_str = "cuda" if args.fly_use_cuda_kernel else "python"
        fly_info = (f", FLy: threshold={args.fly_entropy_threshold}, "
                    f"window={args.fly_window_size}, kernel={kernel_str}")
    print(f"Spec params: num_steps={num_steps}, "
          f"eagle_topk={eagle_topk}, "
          f"num_draft_tokens={num_draft_tokens}{fly_info}")

    # Create SGLang engine
    engine_kwargs = dict(
        model_path=args.target_model,
        speculative_algorithm=algo,
        speculative_draft_model_path=args.draft_model,
        speculative_num_steps=num_steps,
        speculative_eagle_topk=eagle_topk,
        speculative_num_draft_tokens=num_draft_tokens,
        tp_size=args.tp_size,
        max_running_requests=args.max_num_seqs,
        dtype=args.dtype,
        disable_cuda_graph=args.disable_cuda_graph,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        log_level=args.log_level,
        watchdog_timeout=1800,
    )
    if args.fly_enabled:
        engine_kwargs["speculative_fly_enabled"] = True
        engine_kwargs["speculative_fly_entropy_threshold"] = args.fly_entropy_threshold
        engine_kwargs["speculative_fly_window_size"] = args.fly_window_size
        if args.fly_use_cuda_kernel:
            engine_kwargs["speculative_fly_use_cuda_kernel"] = True
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static
    if args.attention_backend:
        engine_kwargs["attention_backend"] = args.attention_backend
        print(f"[CONFIG] Forcing attention backend: {args.attention_backend}")

    if need_patch:
        from sglang.srt.entrypoints.engine import Engine as SGLangEngine

        class PatchedEngine(SGLangEngine):
            run_scheduler_process_func = staticmethod(_patched_run_scheduler_process)

        engine = PatchedEngine(**engine_kwargs)
    else:
        engine = sgl.Engine(**engine_kwargs)

    sampling_params = {
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    # Run benchmark
    print(f"\nGenerating up to {args.max_tokens} tokens per prompt...")
    start_time = time.perf_counter()
    outputs = engine.generate(prompts, sampling_params)
    end_time = time.perf_counter()

    total_time = end_time - start_time

    # Extract per-request metrics
    total_output_tokens = 0
    total_verify_ct = 0
    total_accepted_tokens = 0
    has_spec_metrics = False
    total_fly_deferred = 0
    total_fly_accepted = 0
    total_fly_rejected = 0
    total_fly_entropy_gate = 0
    total_fly_depth_insufficient = 0
    has_fly_metrics = False

    for output in outputs:
        completion_tokens = output["meta_info"]["completion_tokens"]
        total_output_tokens += completion_tokens

        if "spec_verify_ct" in output["meta_info"]:
            has_spec_metrics = True
            total_verify_ct += output["meta_info"]["spec_verify_ct"]

        if "spec_accepted_tokens" in output["meta_info"]:
            total_accepted_tokens += output["meta_info"]["spec_accepted_tokens"]

        if "fly_deferred_count" in output["meta_info"]:
            has_fly_metrics = True
            total_fly_deferred += output["meta_info"]["fly_deferred_count"]
            total_fly_accepted += output["meta_info"]["fly_deferred_accepted"]
            total_fly_rejected += output["meta_info"]["fly_deferred_rejected"]
            total_fly_entropy_gate += output["meta_info"]["fly_entropy_gate_rejections"]
            total_fly_depth_insufficient += output["meta_info"].get("fly_depth_insufficient", 0)

    throughput = total_output_tokens / total_time

    # Print results
    print("\n" + "=" * 70)
    print(f"BENCHMARK RESULTS (SGLang {algo})")
    print("=" * 70)
    print(f"Target model: {args.target_model}")
    print(f"Draft model: {args.draft_model}")
    print(f"Algorithm: {algo}, num_steps={num_steps}, topk={eagle_topk}, "
          f"num_draft_tokens={num_draft_tokens}")
    if args.fly_enabled:
        print(f"FLy: enabled, entropy_threshold={args.fly_entropy_threshold}, "
              f"window_size={args.fly_window_size}")
    print(f"Num prompts: {len(prompts)}")
    print("-" * 70)
    print(f"Total time: {total_time:.2f}s")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Throughput: {throughput:.2f} tokens/s")

    if has_spec_metrics:
        avg_acceptance_length = total_output_tokens / total_verify_ct if total_verify_ct > 0 else 0
        time_per_iteration_ms = (total_time / total_verify_ct) * 1000 if total_verify_ct > 0 else 0
        time_per_token_ms = (total_time / total_output_tokens) * 1000 if total_output_tokens > 0 else 0

        print("-" * 70)
        print("SPECULATION METRICS:")
        print(f"  Total verify rounds: {total_verify_ct}")
        print(f"  Total accepted tokens: {total_accepted_tokens}")
        print(f"  Avg acceptance length: {avg_acceptance_length:.2f}")
        print("-" * 70)
        print("SPEEDUP METRICS:")
        print(f"  Num iterations: {total_verify_ct}")
        print(f"  Time per iteration: {time_per_iteration_ms:.4f} ms")
        print(f"  Tokens per iteration (avg acceptance): {avg_acceptance_length:.2f}")
        print(f"  Time per token: {time_per_token_ms:.4f} ms")

    if has_fly_metrics and (total_fly_deferred > 0 or total_fly_entropy_gate > 0
                            or total_fly_depth_insufficient > 0):
        fly_accept_rate = total_fly_accepted / total_fly_deferred * 100 if total_fly_deferred > 0 else 0
        print("-" * 70)
        print("FLY METRICS:")
        print(f"  Deferrals attempted: {total_fly_deferred}")
        print(f"  Deferrals accepted (window passed): {total_fly_accepted}")
        print(f"  Deferrals rejected (window failed): {total_fly_rejected}")
        print(f"  Entropy gate rejections: {total_fly_entropy_gate}")
        print(f"  Depth insufficient (skipped): {total_fly_depth_insufficient}")
        print(f"  Deferral success rate: {fly_accept_rate:.1f}%")

    # Read and display attention profiling results
    profile_metrics = None
    if args.profile_attention and profile_file:
        engine.shutdown()
        profile_metrics = read_profile_results(profile_file, total_time, total_verify_ct)
        try:
            os.unlink(profile_file)
        except OSError:
            pass
        engine = None

    print("=" * 70)

    # Try to get server-level aggregate metrics
    if engine is not None:
        try:
            server_info = engine.get_server_info()
            internal_states = server_info.get("internal_states", [{}])
            if internal_states:
                avg_spec = internal_states[0].get("avg_spec_accept_length")
                if avg_spec is not None:
                    print(f"\nServer avg_spec_accept_length: {avg_spec:.2f}")
        except Exception:
            pass

    # Print sample outputs
    print("\n--- Sample outputs ---")
    for i in range(min(3, len(outputs))):
        text = outputs[i]["text"][:200]
        print(f"\nPrompt {i}: {prompts[i][:80]}...")
        print(f"Output: {text}...")

    # JSON output
    json_metrics = {
        "framework": "sglang",
        "speculative_algorithm": algo,
        "num_steps": num_steps,
        "eagle_topk": eagle_topk,
        "num_draft_tokens": num_draft_tokens,
        "total_time_s": total_time,
        "total_tokens": total_output_tokens,
        "throughput_tokens_per_s": throughput,
        "num_prompts": len(prompts),
    }

    if has_spec_metrics:
        json_metrics["total_verify_rounds"] = total_verify_ct
        json_metrics["total_accepted_tokens"] = total_accepted_tokens
        json_metrics["avg_acceptance_length"] = avg_acceptance_length
        json_metrics["time_per_iteration_ms"] = time_per_iteration_ms
        json_metrics["time_per_token_ms"] = time_per_token_ms

    if has_fly_metrics:
        json_metrics["fly"] = {
            "enabled": args.fly_enabled,
            "entropy_threshold": args.fly_entropy_threshold,
            "window_size": args.fly_window_size,
            "total_deferrals": total_fly_deferred,
            "deferrals_accepted": total_fly_accepted,
            "deferrals_rejected": total_fly_rejected,
            "entropy_gate_rejections": total_fly_entropy_gate,
            "depth_insufficient": total_fly_depth_insufficient,
        }

    if profile_metrics:
        json_metrics["profile"] = profile_metrics

    print(f"\nJSON metrics: {json.dumps(json_metrics)}")

    if engine is not None:
        engine.shutdown()


if __name__ == "__main__":
    main()

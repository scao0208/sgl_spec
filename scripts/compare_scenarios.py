"""
Compare benchmark results from Scenario A (EAGLE expanded tree) and
Scenario B (fixed mc_sim_8b_512 tree).

Reads JSON metrics from benchmark output files and produces a side-by-side
comparison table with speedup analysis.

Usage:
    # After running benchmarks and saving output to files:
    python compare_scenarios.py \
        --results result_a_unified.json result_a_eagle.json \
                  result_b_unified.json result_b_eagle.json

    # Or parse from benchmark log files:
    python compare_scenarios.py \
        --logs scenario_a_unified.log scenario_a_eagle.log \
               scenario_b_unified.log scenario_b_eagle.log

    # Or provide individual files:
    python compare_scenarios.py \
        --a-unified result_a_unified.json \
        --a-eagle result_a_eagle.json \
        --b-unified result_b_unified.json \
        --b-eagle result_b_eagle.json
"""

import argparse
import json
import sys


def parse_json_from_log(filepath):
    """Extract JSON metrics from a benchmark log file.

    Looks for lines starting with 'JSON metrics: ' and parses the JSON.
    """
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("JSON metrics:"):
                json_str = line[len("JSON metrics:"):].strip()
                return json.loads(json_str)
    raise ValueError(f"No 'JSON metrics:' line found in {filepath}")


def load_result(filepath):
    """Load a result from either a JSON file or a benchmark log file."""
    with open(filepath) as f:
        content = f.read().strip()

    # Try direct JSON first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try parsing as log file
    return parse_json_from_log(filepath)


def fmt_float(v, precision=2):
    if v is None:
        return "N/A"
    return f"{v:.{precision}f}"


def fmt_pct(v, precision=1):
    if v is None:
        return "N/A"
    return f"{v*100:.{precision}f}%"


def print_table(headers, rows, col_widths=None):
    """Print a formatted table."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            w = len(h)
            for row in rows:
                if i < len(row):
                    w = max(w, len(str(row[i])))
            col_widths.append(w + 2)

    # Header
    header_line = ""
    for h, w in zip(headers, col_widths):
        header_line += h.center(w) + "|"
    print(header_line.rstrip("|"))
    print("-" * len(header_line))

    # Rows
    for row in rows:
        line = ""
        for i, (cell, w) in enumerate(zip(row, col_widths)):
            cell_str = str(cell)
            if i == 0:
                line += cell_str.ljust(w) + "|"
            else:
                line += cell_str.rjust(w) + "|"
        print(line.rstrip("|"))


def compute_speedup(baseline, optimized, metric, higher_is_better=True):
    """Compute speedup ratio between two results."""
    base_val = baseline.get(metric)
    opt_val = optimized.get(metric)
    if base_val is None or opt_val is None or base_val == 0:
        return None
    if higher_is_better:
        return opt_val / base_val
    else:
        return base_val / opt_val


def compare(results):
    """Print comparison table from a dict of results.

    Expected keys: 'a_unified', 'a_eagle', 'b_unified', 'b_eagle'
    Any key can be None if that result is not available.
    """
    print("=" * 90)
    print("SCENARIO COMPARISON: ea_attn_exp Tree Kernel vs vLLM Unified Attention")
    print("=" * 90)

    # Overview
    for key, label in [
        ("a_unified", "Scenario A (EAGLE tree) - vLLM unified"),
        ("a_eagle", "Scenario A (EAGLE tree) - ea_attn_exp"),
        ("b_unified", "Scenario B (mc_sim_512) - vLLM unified"),
        ("b_eagle", "Scenario B (mc_sim_512) - ea_attn_exp"),
    ]:
        r = results.get(key)
        if r:
            print(f"\n  {label}:")
            print(f"    Tree: {r.get('tree_size', '?')} nodes, depth {r.get('tree_depth', '?')}")
            print(f"    Prompts: {r.get('num_prompts', '?')}")

    # Main comparison table
    print("\n" + "-" * 90)
    print("PERFORMANCE COMPARISON")
    print("-" * 90)

    headers = ["Metric", "A:unified", "A:ea_attn", "A:speedup", "B:unified", "B:ea_attn", "B:speedup"]
    rows = []

    metric_defs = [
        ("Throughput (tok/s)", "throughput_tokens_per_s", True, fmt_float),
        ("Time/token (ms)", "time_per_token_ms", False, lambda v: fmt_float(v, 4)),
        ("Time/iteration (ms)", "time_per_iteration_ms", False, lambda v: fmt_float(v, 4)),
        ("Total time (s)", "total_time_s", False, fmt_float),
        ("Avg accept length", "avg_acceptance_length", True, fmt_float),
        ("Draft efficiency", "draft_efficiency", True, fmt_pct),
        ("Tokens/iteration", "tokens_per_iteration", True, fmt_float),
        ("Total tokens", "total_tokens", True, lambda v: str(int(v)) if v else "N/A"),
    ]

    for label, metric, higher_is_better, fmt_fn in metric_defs:
        a_uni = results.get("a_unified", {}).get(metric)
        a_eag = results.get("a_eagle", {}).get(metric)
        b_uni = results.get("b_unified", {}).get(metric)
        b_eag = results.get("b_eagle", {}).get(metric)

        a_speedup = None
        if a_uni and a_eag and a_uni != 0:
            if higher_is_better:
                a_speedup = a_eag / a_uni
            else:
                a_speedup = a_uni / a_eag

        b_speedup = None
        if b_uni and b_eag and b_uni != 0:
            if higher_is_better:
                b_speedup = b_eag / b_uni
            else:
                b_speedup = b_uni / b_eag

        rows.append([
            label,
            fmt_fn(a_uni) if a_uni is not None else "N/A",
            fmt_fn(a_eag) if a_eag is not None else "N/A",
            f"{a_speedup:.3f}x" if a_speedup is not None else "N/A",
            fmt_fn(b_uni) if b_uni is not None else "N/A",
            fmt_fn(b_eag) if b_eag is not None else "N/A",
            f"{b_speedup:.3f}x" if b_speedup is not None else "N/A",
        ])

    print_table(headers, rows)

    # Cross-scenario comparison (EAGLE vs Fixed tree)
    print("\n" + "-" * 90)
    print("CROSS-SCENARIO: Scenario A (EAGLE tree) vs Scenario B (Fixed tree)")
    print("-" * 90)

    headers2 = ["Metric", "A (best)", "B (best)", "A/B ratio"]
    rows2 = []

    for label, metric, higher_is_better, fmt_fn in metric_defs:
        # Pick the best kernel result for each scenario
        a_uni = results.get("a_unified", {}).get(metric)
        a_eag = results.get("a_eagle", {}).get(metric)
        b_uni = results.get("b_unified", {}).get(metric)
        b_eag = results.get("b_eagle", {}).get(metric)

        if higher_is_better:
            a_best = max(filter(None, [a_uni, a_eag]), default=None)
            b_best = max(filter(None, [b_uni, b_eag]), default=None)
        else:
            a_best = min(filter(None, [a_uni, a_eag]), default=None)
            b_best = min(filter(None, [b_uni, b_eag]), default=None)

        ratio = None
        if a_best and b_best and b_best != 0:
            if higher_is_better:
                ratio = a_best / b_best
            else:
                ratio = b_best / a_best

        rows2.append([
            label,
            fmt_fn(a_best) if a_best is not None else "N/A",
            fmt_fn(b_best) if b_best is not None else "N/A",
            f"{ratio:.3f}x" if ratio is not None else "N/A",
        ])

    print_table(headers2, rows2)

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    for scenario, key_uni, key_eag, name in [
        ("A", "a_unified", "a_eagle", "EAGLE expanded tree"),
        ("B", "b_unified", "b_eagle", "Fixed mc_sim_8b_512"),
    ]:
        r_uni = results.get(key_uni)
        r_eag = results.get(key_eag)

        if r_uni and r_eag:
            tp_speedup = compute_speedup(r_uni, r_eag, "throughput_tokens_per_s", True)
            lat_speedup = compute_speedup(r_uni, r_eag, "time_per_token_ms", False)
            accept_uni = r_uni.get("avg_acceptance_length", 0)
            accept_eag = r_eag.get("avg_acceptance_length", 0)

            print(f"\n  Scenario {scenario} ({name}):")
            print(f"    Throughput speedup: {tp_speedup:.3f}x" if tp_speedup else "    Throughput speedup: N/A")
            print(f"    Latency speedup:    {lat_speedup:.3f}x" if lat_speedup else "    Latency speedup: N/A")
            print(f"    Acceptance length:  {accept_uni:.2f} (unified) vs {accept_eag:.2f} (ea_attn)")
            if accept_uni == accept_eag:
                print(f"    -> Acceptance identical (kernel swap only, no accuracy change)")
            elif accept_eag > accept_uni:
                print(f"    -> WARNING: Acceptance changed, kernel may affect numerics")

    print("\n" + "=" * 90)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Scenario A vs B benchmark results"
    )
    parser.add_argument(
        "--results", nargs="+", metavar="FILE",
        help="Result files (JSON or log). Auto-detect scenario/kernel from content."
    )
    parser.add_argument("--a-unified", type=str, help="Scenario A, vLLM unified result file")
    parser.add_argument("--a-eagle", type=str, help="Scenario A, ea_attn_exp result file")
    parser.add_argument("--b-unified", type=str, help="Scenario B, vLLM unified result file")
    parser.add_argument("--b-eagle", type=str, help="Scenario B, ea_attn_exp result file")
    return parser.parse_args()


def classify_result(r):
    """Classify a result dict into scenario+kernel key."""
    scenario = r.get("scenario", "")
    kernel = r.get("kernel", "")

    if "A" in scenario or "eagle" in scenario.lower():
        if "ea_attn" in kernel:
            return "a_eagle"
        return "a_unified"
    elif "B" in scenario or "mc_sim" in scenario.lower() or "fixed" in scenario.lower():
        if "ea_attn" in kernel:
            return "b_eagle"
        return "b_unified"
    return None


def main():
    args = parse_args()
    results = {}

    # Load from named args
    for key, filepath in [
        ("a_unified", args.a_unified),
        ("a_eagle", args.a_eagle),
        ("b_unified", args.b_unified),
        ("b_eagle", args.b_eagle),
    ]:
        if filepath:
            results[key] = load_result(filepath)

    # Load from positional results (auto-classify)
    if args.results:
        for filepath in args.results:
            r = load_result(filepath)
            key = classify_result(r)
            if key:
                results[key] = r
            else:
                print(f"WARNING: Could not classify result from {filepath}, skipping")

    if not results:
        print("ERROR: No result files provided.")
        print("Usage examples:")
        print("  python compare_scenarios.py --results a_uni.json a_eagle.json b_uni.json b_eagle.json")
        print("  python compare_scenarios.py --a-unified a_uni.json --a-eagle a_eagle.json \\")
        print("                              --b-unified b_uni.json --b-eagle b_eagle.json")
        sys.exit(1)

    compare(results)

    # Save combined results
    output_file = "comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCombined results saved to {output_file}")


if __name__ == "__main__":
    main()

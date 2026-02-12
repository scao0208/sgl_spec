"""
Correctness test: verify ea_attn_exp kernel matches PyTorch reference
for the mc_sim_8b_512 tree structure.

Tests:
1. Tree mask construction correctness
2. Kernel output vs PyTorch reference with mc_sim_8b_512 mask
3. Various past context lengths (100, 500, 1000)
4. GQA configurations (32 Q heads, 8 KV heads)

Usage:
    cd scripts
    conda activate vllm-spec
    python test_kernel_correctness.py
"""

import torch
import math
from choices import mc_sim_8b_512
from eagle_tree_attention import _attention_strided


def build_tree_attn_mask(tree_choices, device="cuda"):
    """Build tree attention mask in additive format (0=attend, -inf=masked).

    Matches vLLM's _prepare_tree_attn_bias logic.
    """
    tree_len = len(tree_choices) + 1  # +1 for root
    mask = torch.full((tree_len, tree_len), float("-inf"), device=device)

    # Diagonal: self-attention
    for i in range(tree_len):
        mask[i, i] = 0.0

    # Column 0: all attend to root
    mask[:, 0] = 0.0

    # Ancestors
    for i, path in enumerate(tree_choices):
        pos = i + 1
        for c in range(len(path) - 1):
            ancestor_path = path[: c + 1]
            # Find ancestor index
            for j, tc in enumerate(tree_choices):
                if tc == ancestor_path:
                    mask[pos, j + 1] = 0.0
                    break

    return mask


def pytorch_tree_attention(q, k, v, tree_mask, sm_scale):
    """PyTorch reference: tree attention with past context + tree region."""
    B, H, N_CTX_Q, D = q.shape
    N_CTX_KV = k.shape[2]
    past_len = N_CTX_KV - N_CTX_Q

    attn = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    full_mask = torch.zeros((1, 1, N_CTX_Q, N_CTX_KV), dtype=q.dtype, device=q.device)
    full_mask[:, :, :, past_len:] = tree_mask.to(q.dtype)
    attn = attn + full_mask
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


class Results:
    def __init__(self):
        self.tests = []

    def add(self, name, passed, max_diff=None):
        self.tests.append((name, passed, max_diff))

    def summary(self):
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        all_pass = True
        for name, passed, diff in self.tests:
            s = "PASS" if passed else "FAIL"
            d = f" (max_diff={diff:.6f})" if diff is not None else ""
            print(f"  [{s}] {name}{d}")
            all_pass = all_pass and passed
        print("\n" + ("All tests passed!" if all_pass else "Some tests FAILED!"))
        return all_pass


def test_tree_mask_structure(results):
    """Test 1: Verify mc_sim_8b_512 mask structure."""
    print("\n" + "=" * 70)
    print("TEST 1: mc_sim_8b_512 tree mask structure")
    print("=" * 70)

    tree = mc_sim_8b_512
    mask = build_tree_attn_mask(tree, "cuda")
    tree_len = len(tree) + 1

    print(f"  Tree: {len(tree)} nodes + root = {tree_len} total")
    print(f"  Max depth: {max(len(n) for n in tree)}")
    print(f"  Mask shape: {mask.shape}")

    ok = True

    # Root sees only itself
    root_visible = (mask[0, :] == 0).sum().item()
    if root_visible != 1:
        print(f"  ERROR: Root sees {root_visible} positions, expected 1")
        ok = False
    else:
        print(f"  Root sees only itself: OK")

    # All tokens see root
    see_root = (mask[:, 0] == 0).sum().item()
    if see_root != tree_len:
        print(f"  ERROR: {see_root}/{tree_len} tokens see root")
        ok = False
    else:
        print(f"  All tokens see root: OK")

    # Check a few specific nodes
    # (0,) at pos 1 should see root(0) + self(1) = 2
    vis_1 = (mask[1, :] == 0).sum().item()
    if vis_1 != 2:
        print(f"  ERROR: (0,) sees {vis_1}, expected 2")
        ok = False

    # (0,0) should see root(0) + (0,)(1) + self = 3
    idx_00 = tree.index((0, 0)) + 1
    vis_00 = (mask[idx_00, :] == 0).sum().item()
    if vis_00 != 3:
        print(f"  ERROR: (0,0) at pos {idx_00} sees {vis_00}, expected 3")
        ok = False
    else:
        print(f"  (0,0) sees root + (0,) + self = 3: OK")

    # Siblings should NOT see each other: (0,) and (1,) at pos 1 and 2
    if mask[1, 2] == 0:
        print(f"  ERROR: (0,) sees sibling (1,)")
        ok = False
    else:
        print(f"  Siblings don't see each other: OK")

    results.add("Tree mask structure", ok)


def test_kernel_vs_reference(results, past_len, num_q_heads, num_kv_heads, tree_subset=None):
    """Test kernel output vs PyTorch reference."""
    label = f"past={past_len}, Q_heads={num_q_heads}, KV_heads={num_kv_heads}"
    if tree_subset:
        label += f", tree_nodes={tree_subset}"

    print(f"\n--- {label} ---")

    torch.manual_seed(42)
    B, D = 1, 128
    sm_scale = 1.0 / math.sqrt(D)

    tree = mc_sim_8b_512
    if tree_subset:
        tree = tree[:tree_subset]

    tree_mask_2d = build_tree_attn_mask(tree, "cuda")
    N_CTX_Q = tree_mask_2d.shape[0]
    N_CTX_KV = past_len + N_CTX_Q

    # Use Q heads for both Q and KV in the kernel (handle GQA by expansion)
    H = num_q_heads

    q = torch.randn(B, H, N_CTX_Q, D, dtype=torch.float16, device="cuda")
    # For GQA test: generate KV with num_kv_heads, then expand
    k_kv = torch.randn(B, num_kv_heads, N_CTX_KV, D, dtype=torch.float16, device="cuda")
    v_kv = torch.randn(B, num_kv_heads, N_CTX_KV, D, dtype=torch.float16, device="cuda")

    if num_q_heads != num_kv_heads:
        repeat = num_q_heads // num_kv_heads
        k = k_kv.repeat_interleave(repeat, dim=1)
        v = v_kv.repeat_interleave(repeat, dim=1)
    else:
        k = k_kv
        v = v_kv

    tree_mask_4d = tree_mask_2d.unsqueeze(0).unsqueeze(0)

    # Reference
    ref_out = pytorch_tree_attention(
        q.float(), k.float(), v.float(), tree_mask_4d, sm_scale
    ).half()

    # Kernel
    triton_out = _attention_strided.apply(
        q, k, v, tree_mask_4d.to(torch.float16), sm_scale
    )

    max_diff = (ref_out - triton_out).abs().max().item()
    mean_diff = (ref_out - triton_out).abs().mean().item()

    passed = max_diff < 0.02
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] max_diff={max_diff:.6f}, mean_diff={mean_diff:.8f}")

    if not passed:
        # Find worst positions
        diff = (ref_out - triton_out).abs()
        for pos in range(min(10, N_CTX_Q)):
            pos_diff = diff[0, :, pos, :].max().item()
            if pos_diff > 0.01:
                print(f"    Position {pos}: diff={pos_diff:.6f}")

    results.add(label, passed, max_diff)


def main():
    print("=" * 70)
    print("KERNEL CORRECTNESS TEST: ea_attn_exp vs PyTorch reference")
    print(f"Tree: mc_sim_8b_512 ({len(mc_sim_8b_512)} nodes)")
    print("=" * 70)

    results = Results()

    # Test 1: Mask structure
    test_tree_mask_structure(results)

    # Test 2: Small tree subset (fast sanity check)
    print("\n" + "=" * 70)
    print("TEST 2: Small tree subset (64 nodes)")
    print("=" * 70)
    test_kernel_vs_reference(results, past_len=100, num_q_heads=32,
                             num_kv_heads=32, tree_subset=64)

    # Test 3: Full 512-node tree with various past lengths
    print("\n" + "=" * 70)
    print("TEST 3: Full mc_sim_8b_512 tree, various past context lengths")
    print("=" * 70)
    for past_len in [100, 500, 1000]:
        test_kernel_vs_reference(results, past_len=past_len,
                                 num_q_heads=32, num_kv_heads=32)

    # Test 4: GQA (LLaMA 3 style: 32 Q heads, 8 KV heads)
    print("\n" + "=" * 70)
    print("TEST 4: GQA configuration (32Q / 8KV heads)")
    print("=" * 70)
    test_kernel_vs_reference(results, past_len=200, num_q_heads=32,
                             num_kv_heads=8, tree_subset=64)
    test_kernel_vs_reference(results, past_len=500, num_q_heads=32,
                             num_kv_heads=8)

    return results.summary()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

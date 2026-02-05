"""
Benchmark: Compare batched vs sequential OT solving.

This script demonstrates the speedup achieved by solve_OT_batched over
sequential calls to solve_OT.
"""

import time

import numpy as np
import torch

from mdot_tnt import solve_OT, solve_OT_batched


def generate_problems(n_problems: int, size: int, device: str, dtype: torch.dtype, seed: int):
    """Generate random OT problems with shared cost matrix."""
    torch.manual_seed(seed)

    # Random marginals (normalized to sum to 1)
    rs = torch.rand(n_problems, size, device=device, dtype=dtype)
    rs = rs / rs.sum(dim=-1, keepdim=True)

    cs = torch.rand(n_problems, size, device=device, dtype=dtype)
    cs = cs / cs.sum(dim=-1, keepdim=True)

    # Shared cost matrix (scaled to [0, 1])
    C = torch.rand(size, size, device=device, dtype=dtype)

    return rs, cs, C


def solve_sequential(rs, cs, C, **kwargs):
    """Solve OT problems one at a time."""
    results = []
    for i in range(rs.shape[0]):
        result = solve_OT(rs[i], cs[i], C, **kwargs)
        results.append(result)
    return torch.stack(results)


def benchmark(
    n_problems: int = 32,
    size: int = 512,
    n_seeds: int = 5,
    device: str = "cuda",
    dtype: torch.dtype = torch.float64,
    gamma_f: float = 1024.0,
):
    """Run benchmark comparing sequential vs batched solving."""
    print(f"Benchmark: {n_problems} problems, size {size}×{size}, gamma_f={gamma_f}")
    if device == "cuda" and torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        print("Device: CPU")
    print("-" * 60)

    seq_times = []
    batched_times = []
    max_diffs = []

    kwargs = {"gamma_f": gamma_f, "return_plan": False, "round": True}

    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}")
        rs, cs, C = generate_problems(n_problems, size, device, dtype, seed)

        # Warmup
        if seed == 0:
            _ = solve_OT(rs[0], cs[0], C, **kwargs)
            _ = solve_OT_batched(rs[:2], cs[:2], C, **kwargs)
            if device == "cuda":
                torch.cuda.synchronize()

        # Sequential
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        results_seq = solve_sequential(rs, cs, C, **kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        t_seq = time.perf_counter() - t0
        seq_times.append(t_seq)
        print(f"  Sequential: {t_seq:.3f}s")

        # Batched
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        results_batched = solve_OT_batched(rs, cs, C, **kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        t_batched = time.perf_counter() - t0
        batched_times.append(t_batched)
        print(f"  Batched:    {t_batched:.3f}s")

        # Verify results match
        max_diff = (results_seq - results_batched).abs().max().item()
        max_diffs.append(max_diff)
        print(f"  Max diff:   {max_diff:.2e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    seq_median = np.median(seq_times)
    batched_median = np.median(batched_times)
    speedup = seq_median / batched_median

    print(f"Sequential - Median: {seq_median:.3f}s")
    print(f"Batched    - Median: {batched_median:.3f}s")
    print(f"Speedup:   {speedup:.1f}x")
    print(f"Max error: {max(max_diffs):.2e}")

    return {
        "sequential_times": seq_times,
        "batched_times": batched_times,
        "speedup": speedup,
        "max_error": max(max_diffs),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark batched vs sequential OT solving")
    parser.add_argument("--n-problems", type=int, default=32, help="Number of problems")
    parser.add_argument("--size", type=int, default=512, help="Problem size (n=m)")
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--gamma-f", type=float, default=1024.0, help="Temperature parameter")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"
    benchmark(
        n_problems=args.n_problems,
        size=args.size,
        n_seeds=args.n_seeds,
        device=device,
        gamma_f=args.gamma_f,
    )

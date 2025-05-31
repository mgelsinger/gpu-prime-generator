#!/usr/bin/env python3
"""
gpu_prime_generator.py
----------------------
Stream ordered prime numbers to a file using your NVIDIA GPU.

• Press Ctrl-C to stop safely (the file is closed and a tally printed).
• Resumes automatically if the output file already exists.
"""

import argparse
import math
import os
import signal
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import cuda

# ────────────────────────────────────────────────────────────────
# GPU kernel: mark 1 if n is prime, 0 otherwise.
# Each thread tests exactly one candidate number.
# ----------------------------------------------------------------
@cuda.jit
def is_prime_kernel(candidates, small_primes, results):
    i = cuda.grid(1)
    if i >= candidates.size:
        return

    n = candidates[i]
    if n < 2:
        results[i] = 0
        return
    if n == 2:
        results[i] = 1
        return
    if n % 2 == 0:
        results[i] = 0
        return

    is_prime = 1
    # Loop over small primes (already in global memory)
    for j in range(small_primes.size):
        p = small_primes[j]
        if p * p > n:
            break
        if n % p == 0:
            is_prime = 0
            break
    results[i] = is_prime


# ────────────────────────────────────────────────────────────────
# Helpers
# ----------------------------------------------------------------
def sieve_cpu(limit: int) -> list[int]:
    """Classical sieve on the CPU – fast for 'small' limits (<10 M)."""
    mask = np.ones(limit + 1, dtype=bool)
    mask[:2] = False
    for p in range(2, int(math.isqrt(limit)) + 1):
        if mask[p]:
            mask[p * p :: p] = False
    return np.flatnonzero(mask).tolist()


def next_segment(start: int, seg_len: int) -> tuple[np.ndarray, int]:
    """Return an *inclusive* segment [start, start+seg_len) as int64 array."""
    end = start + seg_len
    return np.arange(start, end, dtype=np.int64), end


# ────────────────────────────────────────────────────────────────
# Main loop
# ----------------------------------------------------------------
def generate_primes(
    outfile: Path,
    seg_len: int = 1_000_000,
    threads_per_block: int = 256,
):
    # Resume if file exists
    last_prime = 1
    total_found = 0
    if outfile.exists() and outfile.stat().st_size:
        with outfile.open("r") as f:
            f.seek(0, os.SEEK_END)
            # read last non-empty line
            while f.tell() > 0:
                f.seek(f.tell() - 2, os.SEEK_SET)
                if f.read(1) == "\n":
                    line = f.readline().strip()
                    if line:
                        last_prime = int(line)
                        total_found = sum(1 for _ in open(outfile, "r"))
                        break
    else:
        outfile.parent.mkdir(parents=True, exist_ok=True)

    # Pre-seed small primes list up to 10 000
    small_primes = sieve_cpu(10_000)
    max_small = small_primes[-1]

    print(
        f"Starting at n={last_prime+1:,} "
        f"(already have {total_found:,} primes in {outfile})"
    )

    start = last_prime + 1 if last_prime % 2 else last_prime + 2  # stay on odd
    big_start_time = perf_counter()

    try:
        with outfile.open("a", buffering=1) as f:  # line-buffered
            while True:
                # Extend the CPU sieve if needed
                seg_end = start + seg_len
                needed_limit = int(math.isqrt(seg_end)) + 1
                if needed_limit > max_small:
                    small_primes = sieve_cpu(needed_limit)
                    max_small = small_primes[-1]

                # Transfer small primes to device
                d_small_primes = cuda.to_device(np.array(small_primes, dtype=np.int64))

                # Prepare candidate numbers (odd only for efficiency)
                if start % 2 == 0:
                    start += 1
                candidates_host, next_start = next_segment(start, seg_len)
                d_candidates = cuda.to_device(candidates_host)
                d_results = cuda.device_array(len(candidates_host), dtype=np.int8)

                # Launch kernel
                blocks = (len(candidates_host) + threads_per_block - 1) // threads_per_block
                is_prime_kernel[blocks, threads_per_block](
                    d_candidates, d_small_primes, d_results
                )
                cuda.synchronize()

                mask = d_results.copy_to_host().astype(bool)
                primes_in_seg = candidates_host[mask]

                # Write to file
                for p in primes_in_seg:
                    f.write(f"{int(p)}\n")
                total_found += primes_in_seg.size

                # Progress indicator
                if primes_in_seg.size:
                    print(
                        f"{primes_in_seg.size:>7} primes   "
                        f"up to {primes_in_seg[-1]:,}   "
                        f"│ total {total_found:,}"
                    )

                start = next_start
    except KeyboardInterrupt:
        duration = perf_counter() - big_start_time
        rate = total_found / duration if duration else 0
        print(
            "\n────────────────────────────────────────────────────────\n"
            f"Stopped by user. Found {total_found:,} primes "
            f"in {duration:,.1f} s  ({rate:,.0f} p/s)"
        )


# ────────────────────────────────────────────────────────────────
# CLI entry point
# ----------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="GPU prime-stream generator (CUDA).")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("primes.txt"),
        help="destination file (appends or creates)",
    )
    p.add_argument(
        "-n",
        "--segment",
        type=int,
        default=1_000_000,
        help="how many integers to test per GPU batch",
    )
    args = p.parse_args()
    generate_primes(outfile=args.output, seg_len=args.segment)


if __name__ == "__main__":
    # Make Ctrl-C work instantly even in tight CUDA loops
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()

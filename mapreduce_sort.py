#!/usr/bin/env python3
"""
mapreduce_sort.py
Parallel Sorting (MapReduce-style) supporting multithreading and multiprocessing.

Usage:
    python mapreduce_sort.py --mode thread --workers 4 --size 131072

Notes:
- Map phase: divide array into chunks, sort each chunk in parallel (merge sort via .sort())
- Reduce phase: merge sorted chunks using heapq.merge to produce final sorted array
- IPC (process mode): multiprocessing.Queue to send sorted chunks from workers to reducer
- Memory/time measurement: uses psutil if available; otherwise resource.getrusage (UNIX).
"""

import argparse
import random
import time
import heapq
import sys

# try to import psutil for memory measurement
try:
    import psutil
except Exception:
    psutil = None

from typing import List

# -------------------------
# Utility measurement funcs
# -------------------------
def current_mem_mb():
    if psutil:
        p = psutil.Process()
        return p.memory_info().rss / (1024*1024)
    else:
        # fallback for UNIX-like systems
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return -1.0

# -------------------------
# Map worker implementations
# -------------------------
def chunk_indices(total_len, n_chunks):
    base = total_len // n_chunks
    remainder = total_len % n_chunks
    indices = []
    start = 0
    for i in range(n_chunks):
        end = start + base + (1 if i < remainder else 0)
        indices.append((start, end))
        start = end
    return indices

# -------------
# Thread version
# -------------
def map_sort_threads(arr: List[int], workers: int):
    """Map phase using threads. Returns list of sorted chunks."""
    import threading
    results = [None] * workers
    def worker_func(i, start, end):
        # short pseudocode:
        # - take arr[start:end], sort in-place or copy-and-sort, store result into results[i]
        sub = arr[start:end]
        sub.sort()
        results[i] = sub

    threads = []
    for i, (s, e) in enumerate(chunk_indices(len(arr), workers)):
        t = threading.Thread(target=worker_func, args=(i, s, e))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    return results

# -----------------
# Process version
# -----------------
def map_sort_processes(arr: List[int], workers: int):
    """Map phase using multiprocessing. Returns list of sorted chunks."""
    from multiprocessing import Process, Queue
    q = Queue()
    def worker_proc(start, end, q, idx):
        sub = arr[start:end]  # arr is pickled for child (costly but fine for this assignment)
        sub.sort()
        q.put((idx, sub))
    procs = []
    for i, (s, e) in enumerate(chunk_indices(len(arr), workers)):
        p = Process(target=worker_proc, args=(s, e, q, i))
        procs.append(p)
        p.start()
    results = [None] * workers
    for _ in range(workers):
        idx, sub = q.get()
        results[idx] = sub
    for p in procs:
        p.join()
    return results

# -------------
# Reduce phase
# -------------
def reduce_merge(sorted_chunks: List[List[int]]):
    """Merge sorted chunks into one sorted list using heapq.merge."""
    # heapq.merge accepts multiple sorted iterables and returns a lazy iterator
    merged = list(heapq.merge(*sorted_chunks))
    return merged

# -------------------------
# Whole runner for sorting
# -------------------------
def run_sort(mode: str, workers: int, size:int):
    assert mode in ('thread','process')
    # generate data
    arr = [random.randint(0, 10**9) for _ in range(size)]

    # correctness test size small: for S=32 we will check the final result matches Python sort
    check_correct = (size <= 1024)

    mem_before = current_mem_mb()
    t0 = time.perf_counter()

    # Map phase
    map_start = time.perf_counter()
    if mode == 'thread':
        sorted_chunks = map_sort_threads(arr, workers)
    else:
        sorted_chunks = map_sort_processes(arr, workers)
    map_end = time.perf_counter()

    # Reduce phase
    reduce_start = time.perf_counter()
    final_sorted = reduce_merge(sorted_chunks)
    reduce_end = time.perf_counter()

    t1 = time.perf_counter()
    mem_after = current_mem_mb()

    # correctness check
    if check_correct:
        expected = sorted(arr)
        assert final_sorted == expected, "Sorting incorrect!"

    print(f"MODE={mode} workers={workers} size={size}")
    print(f"map_time_s: {map_end - map_start:.6f}")
    print(f"reduce_time_s: {reduce_end - reduce_start:.6f}")
    print(f"total_time_s: {t1 - t0:.6f}")
    if mem_before >= 0 and mem_after >= 0:
        print(f"mem_before_mb: {mem_before:.2f} mem_after_mb: {mem_after:.2f} delta_mb: {mem_after-mem_before:.2f}")
    else:
        print("Memory: measurement not available on this platform.")


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['thread','process'], default='thread')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--size', type=int, default=131072)
    args = parser.parse_args()
    run_sort(args.mode, args.workers, args.size)

if __name__ == '__main__':
    main()

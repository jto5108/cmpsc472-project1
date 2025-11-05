#!/usr/bin/env python3
"""
max_aggregation.py
Compute global maximum with constrained shared memory (single integer) using threading or multiprocessing.

Usage:
    python max_aggregation.py --mode thread --workers 4 --size 131072

Notes:
- Each worker computes local max of its chunk.
- Shared buffer stores a single integer (current global max).
- Workers must read current value, compare, and update only if local > current.
- Synchronization: threading.Lock for threads; multiprocessing.Value + Lock for processes.
- Measures timing and memory similar to mapreduce_sort.
"""

import argparse
import random
import time
import sys

try:
    import psutil
except Exception:
    psutil = None

from typing import List

def current_mem_mb():
    if psutil:
        p = psutil.Process()
        return p.memory_info().rss / (1024*1024)
    else:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return -1.0

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

# -------------------------
# Thread-based implementation
# -------------------------
def max_aggregation_threads(arr, workers):
    import threading
    # shared buffer: single-element list to hold int (acts as "single integer buffer")
    shared_buf = [min(arr) - 1]  # initial value below possible min
    lock = threading.Lock()

    def worker(i, start, end):
        local_max = max(arr[start:end]) if end>start else float('-inf')
        # critical region: read, compare, maybe write
        with lock:
            if local_max > shared_buf[0]:
                shared_buf[0] = local_max

    threads = []
    for i,(s,e) in enumerate(chunk_indices(len(arr), workers)):
        t = threading.Thread(target=worker, args=(i,s,e))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    return shared_buf[0]

# -------------------------
# Process-based implementation
# -------------------------
def max_aggregation_processes(arr, workers):
    from multiprocessing import Process, Value, Lock
    # shared single integer Value (C int)
    shared_val = Value('q', min(arr)-1)  # use 'q' (signed long long) to be safe for large ints
    lock = Lock()
    def proc_worker(start, end, sv, lock):
        # compute local max
        local_max = max(arr[start:end]) if end>start else -9223372036854775808
        # synchronization: acquire lock, read, compare, update
        with lock:
            if local_max > sv.value:
                sv.value = local_max

    procs = []
    for s,e in chunk_indices(len(arr), workers):
        p = Process(target=proc_worker, args=(s,e,shared_val,lock))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()
    return shared_val.value

# -------------------------
# Runner
# -------------------------
def run_max(mode, workers, size):
    assert mode in ('thread','process')
    # generate data
    arr = [random.randint(-10**9, 10**9) for _ in range(size)]

    mem_before = current_mem_mb()
    t0 = time.perf_counter()

    map_start = time.perf_counter()
    if mode == 'thread':
        final = max_aggregation_threads(arr, workers)
    else:
        final = max_aggregation_processes(arr, workers)
    map_end = time.perf_counter()

    t1 = time.perf_counter()
    mem_after = current_mem_mb()

    # verify correctness versus builtin max (small sizes always)
    if size <= 1024:
        assert final == max(arr), f"Wrong result: got {final} expected {max(arr)}"

    print(f"MODE={mode} workers={workers} size={size}")
    print(f"map_time_s (incl updates): {map_end - map_start:.6f}")
    print(f"total_time_s: {t1 - t0:.6f}")
    if mem_before >= 0 and mem_after >= 0:
        print(f"mem_before_mb: {mem_before:.2f} mem_after_mb: {mem_after:.2f} delta_mb: {mem_after-mem_before:.2f}")
    else:
        print("Memory: measurement not available on this platform.")
    print(f"global_max: {final}")

# CLI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['thread','process'], default='thread')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--size', type=int, default=131072)
    args = parser.parse_args()
    run_max(args.mode, args.workers, args.size)

if __name__ == '__main__':
    main()

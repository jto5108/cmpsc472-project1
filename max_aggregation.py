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

# -----------------------------------------------
# Helper: Return current process memory in MB
# -----------------------------------------------
def current_mem_mb():
    if psutil:
        # psutil gives detailed memory info
        p = psutil.Process()
        return p.memory_info().rss / (1024*1024)
    else:
        # fallback using resource if psutil not available
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return -1.0

# -----------------------------------------------
# Helper: Compute chunk start/end indices for each worker
# -----------------------------------------------
def chunk_indices(total_len, n_chunks):
    base = total_len // n_chunks
    remainder = total_len % n_chunks
    indices = []
    start = 0
    # distribute remainder one-by-one to first few chunks
    for i in range(n_chunks):
        end = start + base + (1 if i < remainder else 0)
        indices.append((start, end))
        start = end
    return indices

# ============================================================
# THREAD-BASED IMPLEMENTATION
# ============================================================
def max_aggregation_threads(arr, workers):
    """
    Compute the global max using threads.
    Shared memory is simulated by a one-element list (shared_buf).
    Lock ensures only one thread updates the shared buffer at a time.
    """
    import threading

    # shared single integer buffer (simulated using list of one element)
    shared_buf = [min(arr) - 1]  # initial value below any array element
    lock = threading.Lock()      # ensures synchronized access

    # Worker function
    def worker(i, start, end):
        # Step 1: compute local max for this chunk
        local_max = max(arr[start:end]) if end > start else float('-inf')

        # Step 2: critical region (compare and possibly update)
        with lock:
            if local_max > shared_buf[0]:
                shared_buf[0] = local_max

    # Step 3: Create and start worker threads
    threads = []
    for i, (s, e) in enumerate(chunk_indices(len(arr), workers)):
        t = threading.Thread(target=worker, args=(i, s, e))
        threads.append(t)
        t.start()

    # Step 4: Wait for all threads to complete
    for t in threads:
        t.join()

    # Step 5: Return final max from shared buffer
    return shared_buf[0]

# ============================================================
# PROCESS-BASED IMPLEMENTATION
# ============================================================
def max_aggregation_processes(arr, workers):
    """
    Compute global max using multiprocessing.
    Shared value and lock are created using multiprocessing primitives.
    """
    from multiprocessing import Process, Value, Lock

    # Shared single integer using multiprocessing.Value
    shared_val = Value('q', min(arr) - 1)  # 'q' = long long integer
    lock = Lock()

    # Worker function (executed in a separate process)
    def proc_worker(start, end, sv, lock):
        # Step 1: compute local max of this segment
        local_max = max(arr[start:end]) if end > start else -9223372036854775808

        # Step 2: critical region (compare & update shared value)
        with lock:
            if local_max > sv.value:
                sv.value = local_max

    # Step 3: Create and start worker processes
    procs = []
    for s, e in chunk_indices(len(arr), workers):
        p = Process(target=proc_worker, args=(s, e, shared_val, lock))
        procs.append(p)
        p.start()

    # Step 4: Wait for all processes to finish
    for p in procs:
        p.join()

    # Step 5: Return final result stored in shared value
    return shared_val.value

# ============================================================
# RUNNER FUNCTION
# ============================================================
def run_max(mode, workers, size):
    """
    Run the selected mode (thread or process) and measure:
    - execution time
    - memory usage
    - correctness (for small input)
    """
    assert mode in ('thread', 'process')

    # Step 1: Generate random data
    arr = [random.randint(-10**9, 10**9) for _ in range(size)]

    # Step 2: Measure memory before computation
    mem_before = current_mem_mb()
    t0 = time.perf_counter()

    # Step 3: Perform computation
    map_start = time.perf_counter()
    if mode == 'thread':
        final = max_aggregation_threads(arr, workers)
    else:
        final = max_aggregation_processes(arr, workers)
    map_end = time.perf_counter()

    # Step 4: Measure time and memory after computation
    t1 = time.perf_counter()
    mem_after = current_mem_mb()

    # Step 5: Validate correctness for small datasets
    if size <= 1024:
        assert final == max(arr), f"Wrong result: got {final}, expected {max(arr)}"

    # Step 6: Display performance results
    print(f"MODE={mode} workers={workers} size={size}")
    print(f"map_time_s (incl updates): {map_end - map_start:.6f}")
    print(f"total_time_s: {t1 - t0:.6f}")
    if mem_before >= 0 and mem_after >= 0:
        print(f"mem_before_mb: {mem_before:.2f} mem_after_mb: {mem_after:.2f} delta_mb: {mem_after - mem_before:.2f}")
    else:
        print("Memory: measurement not available on this platform.")
    print(f"global_max: {final}")

# ============================================================
# COMMAND-LINE INTERFACE (ENTRY POINT)
# ============================================================
def main():
    """
    Parse command-line arguments and execute run_max().
    Example:
        python max_aggregation.py --mode thread --workers 4 --size 131072
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['thread', 'process'], default='thread',
                        help='Select between threading and multiprocessing modes')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker threads or processes')
    parser.add_argument('--size', type=int, default=131072,
                        help='Size of random input data array')
    args = parser.parse_args()

    # Run program
    run_max(args.mode, args.workers, args.size)


# Run main if executed as script
if __name__ == '__main__':
    main()

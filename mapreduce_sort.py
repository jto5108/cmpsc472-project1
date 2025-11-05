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
- Memory/time measurement: uses psutil if available; otherwise resource.getrusage (UNIX)
"""

import argparse
import random
import time
import heapq
import sys

# optional memory measurement library
try:
    import psutil
except Exception:
    psutil = None

from typing import List

# -------------------------
# Utility: Measure current memory usage in MB
# -------------------------
def current_mem_mb():
    """Return current process memory usage in MB"""
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
# Helper: Compute chunk indices for splitting array among workers
# -------------------------
def chunk_indices(total_len, n_chunks):
    """Divide total_len into n_chunks as evenly as possible"""
    base = total_len // n_chunks
    remainder = total_len % n_chunks
    indices = []
    start = 0
    for i in range(n_chunks):
        end = start + base + (1 if i < remainder else 0)  # distribute remainder
        indices.append((start, end))
        start = end
    return indices

# ============================================================
# THREAD-BASED MAP PHASE
# ============================================================
def map_sort_threads(arr: List[int], workers: int):
    """
    Map phase using threads.
    - Each thread sorts a chunk of the array.
    - Results stored in shared list 'results'.
    """
    import threading
    results = [None] * workers

    def worker_func(i, start, end):
        """Worker thread: sort chunk and store in results[i]"""
        sub = arr[start:end]
        sub.sort()  # in-place sort for efficiency
        results[i] = sub

    # spawn threads for each chunk
    threads = []
    for i, (s, e) in enumerate(chunk_indices(len(arr), workers)):
        t = threading.Thread(target=worker_func, args=(i, s, e))
        threads.append(t)
        t.start()

    # wait for all threads to complete
    for t in threads:
        t.join()

    return results

# ============================================================
# PROCESS-BASED MAP PHASE
# ============================================================
def map_sort_processes(arr: List[int], workers: int):
    """
    Map phase using processes.
    - Each process sorts a chunk of the array.
    - Results returned via multiprocessing.Queue.
    """
    from multiprocessing import Process, Queue
    q = Queue()

    def worker_proc(start, end, q, idx):
        """Worker process: sort chunk and put into queue"""
        sub = arr[start:end]  # array is pickled and sent to child
        sub.sort()
        q.put((idx, sub))

    # spawn processes for each chunk
    procs = []
    for i, (s, e) in enumerate(chunk_indices(len(arr), workers)):
        p = Process(target=worker_proc, args=(s, e, q, i))
        procs.append(p)
        p.start()

    # collect sorted chunks from queue
    results = [None] * workers
    for _ in range(workers):
        idx, sub = q.get()
        results[idx] = sub

    # ensure all processes finished
    for p in procs:
        p.join()

    return results

# ============================================================
# REDUCE PHASE
# ============================================================
def reduce_merge(sorted_chunks: List[List[int]]):
    """
    Reduce phase: merge sorted chunks into final sorted array
    - Uses heapq.merge for efficient k-way merge
    """
    merged = list(heapq.merge(*sorted_chunks))  # lazy iterator converted to list
    return merged

# ============================================================
# FULL RUNNER
# ============================================================
def run_sort(mode: str, workers: int, size:int):
    """
    Run MapReduce-style parallel sort with timing and memory measurement.
    - mode: 'thread' or 'process'
    - workers: number of threads/processes
    - size: size of input array
    """
    assert mode in ('thread','process')

    # Step 1: generate random input array
    arr = [random.randint(0, 10**9) for _ in range(size)]

    # Step 2: correctness check only for small arrays
    check_correct = (size <= 1024)

    # Step 3: measure memory and start timer
    mem_before = current_mem_mb()
    t0 = time.perf_counter()

    # -------------------------
    # Map Phase
    # -------------------------
    map_start = time.perf_counter()
    if mode == 'thread':
        sorted_chunks = map_sort_threads(arr, workers)
    else:
        sorted_chunks = map_sort_processes(arr, workers)
    map_end = time.perf_counter()

    # -------------------------
    # Reduce Phase
    # -------------------------
    reduce_start = time.perf_counter()
    final_sorted = reduce_merge(sorted_chunks)
    reduce_end = time.perf_counter()

    # Step 4: total elapsed time and memory
    t1 = time.perf_counter()
    mem_after = current_mem_mb()

    # Step 5: correctness check
    if check_correct:
        expected = sorted(arr)
        assert final_sorted == expected, "Sorting incorrect!"

    # Step 6: display performance metrics
    print(f"MODE={mode} workers={workers} size={size}")
    print(f"map_time_s: {map_end - map_start:.6f}")
    print(f"reduce_time_s: {reduce_end - reduce_start:.6f}")
    print(f"total_time_s: {t1 - t0:.6f}")
    if mem_before >= 0 and mem_after >= 0:
        print(f"mem_before_mb: {mem_before:.2f} mem_after_mb: {mem_after:.2f} delta_mb: {mem_after-mem_before:.2f}")
    else:
        print("Memory: measurement not available on this platform.")

# ============================================================
# CLI ENTRY POINT
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['thread','process'], default='thread',
                        help='Select threading or multiprocessing')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of threads or processes')
    parser.add_argument('--size', type=int, default=131072,
                        help='Input array size')
    args = parser.parse_args()
    run_sort(args.mode, args.workers, args.size)

if __name__ == '__main__':
    main()

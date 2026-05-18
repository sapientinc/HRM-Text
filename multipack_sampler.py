from typing import Optional

import torch.distributed as dist
from torch.utils.data import Sampler

import numpy as np
import numba


@numba.njit
def lpt_check(heap: np.ndarray, A: np.ndarray, c: int, n: int):
    # LPT (Longest processing time first scheduling)
    # Time: O(|A| log |A| + |A| log n)

    A = np.sort(A)[::-1]
    heap.fill(0)
    for size in A:
        # Put into smallest element
        heap[1] += size
        if heap[1] > c:
            return False

        # Heapify (Sink)
        # https://stackoverflow.com/questions/20397674/replacing-element-in-min-heap
        u = 1
        while (u << 1) <= n:
            v = u << 1  # lch
            rch = (u << 1) | 1
            if rch <= n and heap[rch] < heap[v]:
                v = rch
            
            if heap[u] <= heap[v]:
                break

            heap[u], heap[v] = heap[v], heap[u]
            u = v

    return True


@numba.njit
def lpt_with_result(heap: np.ndarray, A: np.ndarray, n: int, rank: int):
    # LPT (Longest processing time first scheduling)
    # Time: O(|A| log |A| + |A| log n)

    result = []

    indices = np.argsort(A, kind="mergesort")[::-1]  # Stable sort for coherence across machine and versions
    A = A[indices]

    heap.fill(0)
    heap_id = np.arange(-1, n, dtype=A.dtype)
    for idx, size in enumerate(A):
        # Put into smallest element
        heap[1] += size
        if heap_id[1] == rank:
            result.append(indices[idx])

        # Heapify (Sink)
        # https://stackoverflow.com/questions/20397674/replacing-element-in-min-heap
        u = 1
        while (u << 1) <= n:
            v = u << 1  # lch
            rch = (u << 1) | 1
            if rch <= n and heap[rch] < heap[v]:
                v = rch
            
            if heap[u] <= heap[v]:
                break

            heap[u], heap[v] = heap[v], heap[u]
            heap_id[u], heap_id[v] = heap_id[v], heap_id[u]
            u = v

    return np.array(result)


@numba.njit
def allocate(heap: np.ndarray, start_index: int, lengths: np.ndarray, rank: int, c: int, n: int):
    # Dynamic batch allocator, binary search + LPT
    # ~99.5% efficiency on OpenChat training set (12 * 2048 ctx len)

    # Linear scan
    total_len = 0
    end_index = start_index
    while end_index < lengths.size:
        l = lengths[end_index]
        total_len += l
        end_index += 1

        if total_len >= c * n:
            break

    a = lengths[start_index: end_index]

    # binary search [l, r)
    l = 1
    r = 1 + a.size

    while r - l > 1:
        m = (l + r) // 2
        if lpt_check(heap, a[:m], c, n):
            l = m
        else:
            r = m

    # use length l
    batch = start_index + lpt_with_result(heap, a[:l], n, rank)

    return l >= n, l, batch, np.sum(a[:l])


class MultipackDistributedBatchSampler(Sampler):
    """Unpadded length sampling using Multipack V2, for models with quadratic attention complexity.
       It also tries to evenly distribute the sequences using LPT, so that quadratic load is more balanced.

       Approximate (at most 1.33x ?) the optimal solution of the identical-machines scheduling problem, which is NP-hard.

       Time Complexity: O(n log n log k)
       n = maximum number of sequences per batch, k = number of nodes
    """

    def __init__(
        self,
        batch_max_length: int,
        lengths: np.ndarray,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        drop_last_batch: bool = False
    ):
        # Get rank
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last_batch = drop_last_batch

        self.batch_max_length = batch_max_length
        self.lengths = lengths
        assert isinstance(self.lengths, np.ndarray)

        # statistics
        self.eff_total_used = 0
        self.eff_total_slots = 0

    def iter(self):
        # Allocate workspace
        heap = np.empty(self.num_replicas + 1, dtype=self.lengths.dtype)

        start_index = 0
        while start_index < self.lengths.size:
            is_full, global_numseq, batch, batch_totlen = allocate(heap,
                                                                   start_index, self.lengths,
                                                                   rank=self.rank, c=self.batch_max_length, n=self.num_replicas)
            start_index += global_numseq

            if not is_full:  # Skip batch with empty GPUs
                break
            if self.drop_last_batch and (start_index >= self.lengths.size):  # Skip last batch
                break

            self.eff_total_used += batch_totlen
            self.eff_total_slots += self.num_replicas * self.batch_max_length
            
            yield batch
            
    def estimate_num_batches(self):
        # Rough estimate, don't consider packing efficiency
        return round(np.sum(self.lengths) / (self.num_replicas * self.batch_max_length))

    def efficiency(self):
        return self.eff_total_used / self.eff_total_slots

"""Example: resource allocation with QES compute allocator.

This example treats a set of jobs as candidate demand profiles and allocates a
shared compute budget according to each job's permission, uncertainty, risk,
and value.
"""
from __future__ import annotations

from qes.compute_allocator import ComputeAllocator, RoomComputeProfile


def main() -> None:
    allocator = ComputeAllocator()
    jobs = [
        RoomComputeProfile(permission=0.95, uncertainty=0.82, risk=1.3, value=8.0),
        RoomComputeProfile(permission=0.72, uncertainty=0.91, risk=0.7, value=6.5),
        RoomComputeProfile(permission=0.60, uncertainty=0.70, risk=2.2, value=4.2),
        RoomComputeProfile(permission=0.88, uncertainty=0.62, risk=1.0, value=7.8),
    ]

    allocations = allocator.allocate_with_floor(jobs, total=100.0, min_share=0.10)
    priorities = [allocator.priority(job) for job in jobs]

    print("QES resource allocation")
    print("=" * 28)
    pairs = zip(jobs, priorities, allocations, strict=True)
    for index, (_job, priority, allocation) in enumerate(pairs, start=1):
        print(f"Job {index}: priority={priority:.4f}, allocation={allocation:.2f}")

    print(f"Total: {allocations.sum():.2f}")


if __name__ == "__main__":
    main()

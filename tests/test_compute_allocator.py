import numpy as np
import pytest

from qes.compute_allocator import ComputeAllocator, RoomComputeProfile


def test_priority_increases_with_each_factor():
    allocator = ComputeAllocator()
    low = RoomComputeProfile(permission=0.1, uncertainty=0.1, risk=0.0, value=0.1)
    high = RoomComputeProfile(permission=0.9, uncertainty=0.9, risk=5.0, value=0.9)
    assert allocator.priority(high) > allocator.priority(low)


def test_allocate_sums_to_total_budget():
    allocator = ComputeAllocator()
    profiles = [
        RoomComputeProfile(permission=0.9, uncertainty=0.5, risk=0.1, value=0.7),
        RoomComputeProfile(permission=0.2, uncertainty=0.9, risk=2.0, value=0.1),
        RoomComputeProfile(permission=0.5, uncertainty=0.5, risk=0.5, value=0.5),
    ]
    allocation = allocator.allocate(profiles, total=100.0)
    assert allocation.sum() == pytest.approx(100.0)


def test_allocate_gives_more_to_higher_priority_room():
    allocator = ComputeAllocator()
    profiles = [
        RoomComputeProfile(permission=0.9, uncertainty=0.9, risk=5.0, value=0.9),
        RoomComputeProfile(permission=0.1, uncertainty=0.1, risk=0.0, value=0.1),
    ]
    allocation = allocator.allocate(profiles, total=100.0)
    assert allocation[0] > allocation[1]


def test_check_resource_constraint():
    requested = np.array([10.0, 20.0, 30.0])
    assert ComputeAllocator.check_resource_constraint(requested, capacity=60.0)
    assert not ComputeAllocator.check_resource_constraint(requested, capacity=50.0)


def test_clip_to_capacity_scales_down_proportionally():
    requested = np.array([10.0, 30.0])
    clipped = ComputeAllocator.clip_to_capacity(requested, capacity=20.0)
    assert clipped.sum() == 20.0
    np.testing.assert_allclose(clipped / clipped.sum(), requested / requested.sum())


def test_clip_to_capacity_noop_when_within_budget():
    requested = np.array([1.0, 2.0])
    clipped = ComputeAllocator.clip_to_capacity(requested, capacity=100.0)
    np.testing.assert_allclose(clipped, requested)


def test_allocate_with_floor_reduces_to_plain_allocate_at_zero_min_share():
    allocator = ComputeAllocator()
    profiles = [
        RoomComputeProfile(permission=0.9, uncertainty=0.9, risk=5.0, value=0.9),
        RoomComputeProfile(permission=0.1, uncertainty=0.1, risk=0.0, value=0.1),
    ]
    plain = allocator.allocate(profiles, total=100.0)
    floored = allocator.allocate_with_floor(profiles, total=100.0, min_share=0.0)
    np.testing.assert_allclose(plain, floored)


def test_allocate_with_floor_guarantees_minimum_share_to_low_priority_room():
    allocator = ComputeAllocator()
    profiles = [
        RoomComputeProfile(permission=0.9, uncertainty=0.9, risk=5.0, value=0.9),
        RoomComputeProfile(permission=0.0001, uncertainty=0.0001, risk=0.0, value=0.0001),
    ]
    plain = allocator.allocate(profiles, total=100.0)
    floored = allocator.allocate_with_floor(profiles, total=100.0, min_share=0.5)
    assert floored[1] > plain[1]
    assert floored.sum() == pytest.approx(100.0)


def test_allocate_with_floor_handles_empty_profiles():
    allocator = ComputeAllocator()
    result = allocator.allocate_with_floor([], total=100.0)
    assert result.shape == (0,)


def test_allocate_returns_zeros_when_priority_sum_is_non_positive():
    allocator = ComputeAllocator()
    allocation = allocator.allocate([], total=100.0)
    assert allocation.shape == (0,)


def test_allocate_multi_resource_returns_per_resource_allocations():
    allocator = ComputeAllocator()
    profiles = [
        RoomComputeProfile(permission=0.9, uncertainty=0.9, risk=5.0, value=0.9),
        RoomComputeProfile(permission=0.1, uncertainty=0.1, risk=0.0, value=0.1),
    ]
    result = allocator.allocate_multi_resource(
        profiles, capacities={"cpu": 100.0, "gpu": 10.0}
    )
    assert set(result.keys()) == {"cpu", "gpu"}
    assert result["cpu"].sum() == pytest.approx(100.0)
    assert result["gpu"].sum() == pytest.approx(10.0)

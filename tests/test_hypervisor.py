import pytest

from qes.hypervisor import CosmicVP


def test_virtual_resource_explicit_id_is_preserved():
    from qes.hypervisor import VirtualResource

    resource = VirtualResource(kind="vm", id="custom-id")
    assert resource.id == "custom-id"


def test_h11_must_be_positive():
    with pytest.raises(ValueError):
        CosmicVP(h11=0.0)


def test_create_resources_of_each_kind():
    vp = CosmicVP(h11=1.0, capacity=10)
    vm = vp.create_vm()
    net = vp.create_network()
    storage = vp.create_storage()
    container = vp.create_container()
    assert {vm.kind, net.kind, storage.kind, container.kind} == {"vm", "network", "storage", "container"}
    assert len(vp.resources) == 4


def test_create_raises_when_capacity_exceeded():
    vp = CosmicVP(h11=1.0, capacity=2)
    vp.create_vm()
    vp.create_vm()
    with pytest.raises(RuntimeError):
        vp.create_vm()


def test_expand_raises_on_non_positive_nodes():
    vp = CosmicVP()
    with pytest.raises(ValueError):
        vp.expand(0)


def test_expand_increases_capacity_headroom():
    vp = CosmicVP(h11=1.0, capacity=1)
    vp.create_vm()
    vp.expand(1)
    # now capacity is nodes(2) * capacity(1) = 2, room for one more
    vp.create_vm()
    assert len(vp.resources) == 2


def test_stabilize_within_and_beyond_threshold():
    vp = CosmicVP(h11=2.0)
    assert vp.stabilize(delta_load=1.0, threshold=1.0)
    assert not vp.stabilize(delta_load=10.0, threshold=1.0)


def test_security_check_both_halves():
    vp = CosmicVP(h11=1.0)
    assert vp.security_check(n=5.0, n_prime=1.0, phi=1.0)
    assert not vp.security_check(n=0.0, n_prime=1.0, phi=1.0)


def test_drift_map_and_of():
    vp = CosmicVP(h11=2.0)
    resource = vp.create("vm", phi=4.0)
    assert vp.drift_of(resource.id) == pytest.approx(2.0)
    updated = vp.map_drift(resource.id, phi=6.0)
    assert updated == pytest.approx(3.0)
    assert vp.drift_of("missing") == 0.0


def test_map_drift_unknown_resource_raises():
    vp = CosmicVP()
    with pytest.raises(KeyError):
        vp.map_drift("nope", phi=1.0)


def test_acros_execute_runs_executor():
    vp = CosmicVP()
    result = vp.acros_execute("program", lambda p: p.upper())
    assert result == "PROGRAM"
    assert ("acros_execute", "program") in vp.event_log


def test_null_recover_replaces_resource():
    vp = CosmicVP(h11=1.0, capacity=10)
    resource = vp.create_vm(spec={"cpu": 4})
    replacement = vp.null_recover(resource.id)
    assert replacement.id != resource.id
    assert replacement.kind == "vm"
    assert replacement.spec == {"cpu": 4}
    assert resource.id not in vp.resources


def test_null_recover_unknown_resource_defaults_to_vm():
    vp = CosmicVP(h11=1.0, capacity=10)
    replacement = vp.null_recover("missing")
    assert replacement.kind == "vm"


def test_request_cycle_success():
    vp = CosmicVP(h11=1.0, capacity=10)
    result = vp.request_cycle("req", executor=lambda r: r + "-done")
    assert result.output == "req-done"
    assert result.recovered is False


def test_request_cycle_expands_when_at_capacity():
    vp = CosmicVP(h11=1.0, capacity=1)
    vp.create_vm()
    result = vp.request_cycle("req", executor=lambda r: r)
    assert result.expanded is True


def test_request_cycle_recovers_from_executor_failure():
    vp = CosmicVP(h11=1.0, capacity=10)
    calls = {"n": 0}

    def flaky_executor(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return "recovered"

    result = vp.request_cycle("req", executor=flaky_executor)
    assert result.recovered is True
    assert result.output == "recovered"


def test_virtual_resource_rejects_invalid_kind_spec_and_id():
    from qes.hypervisor import VirtualResource

    with pytest.raises(TypeError):
        VirtualResource(kind=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        VirtualResource(kind="   ")
    with pytest.raises(TypeError):
        VirtualResource(kind="vm", spec=[])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        VirtualResource(kind="vm", id=123)  # type: ignore[arg-type]


def test_cosmic_vp_validates_h11_capacity_and_threshold_inputs():
    with pytest.raises(TypeError):
        CosmicVP(h11=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CosmicVP(h11=float("inf"))
    with pytest.raises(TypeError):
        CosmicVP(capacity=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CosmicVP(capacity=0)

    vp = CosmicVP()
    with pytest.raises(ValueError):
        vp.stabilize(delta_load=1.0, threshold=-0.1)


def test_hypervisor_methods_validate_resource_ids_and_executors():
    vp = CosmicVP()

    with pytest.raises(TypeError):
        vp.drift_of(1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        vp.map_drift(1, phi=1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        vp.acros_execute("program", None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        vp.null_recover(1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        vp.request_cycle("req", executor=None)  # type: ignore[arg-type]

import pytest

from qroute.circuit import Circuit
from qroute.hardware import HardwareGraph
from qroute.mapping import Mapping
from qroute.routers.greedy import GreedyRouter
from qroute.routers.naive import NaiveRouter
from qroute.validation import (
    ValidationError,
    validate_adjacency,
    validate_hardware_indices,
    validate_mapping,
    validate_routing_result,
    verify_equivalence_statevector,
)


def _sample_circuit():
    c = Circuit(4)
    c.h(0)
    c.cx(0, 3)
    c.cx(1, 2)
    c.cx(0, 1)
    return c


class TestBasicChecks:
    def test_valid_mapping_passes(self):
        validate_mapping(Mapping.identity(3))

    def test_hardware_indices_out_of_range_rejected(self):
        c = Circuit(2)
        c.add_gate("H", (0,))
        hw = HardwareGraph.from_line(2)
        # Manually build a bad circuit referencing an out-of-range physical qubit.
        bad = Circuit(5)
        bad.add_gate("H", (4,))
        with pytest.raises(ValidationError):
            validate_hardware_indices(bad, hw)

    def test_non_adjacent_two_qubit_gate_rejected(self):
        hw = HardwareGraph.from_line(3)
        bad = Circuit(3)
        bad.add_gate("CX", (0, 2))
        with pytest.raises(ValidationError):
            validate_adjacency(bad, hw)


class TestRoutingResultValidation:
    def test_naive_router_produces_valid_result(self):
        c = _sample_circuit()
        hw = HardwareGraph.from_line(4)
        result = NaiveRouter(hw).route(c)
        validate_routing_result(c, result, hw)

    def test_greedy_router_produces_valid_result_with_reordering(self):
        c = _sample_circuit()
        hw = HardwareGraph.from_line(4)
        result = GreedyRouter(hw).route(c)
        validate_routing_result(c, result, hw)

    def test_detects_swapped_gate_order_violation(self):
        # Deliberately construct a broken "routing result" where two gates
        # sharing a qubit have been reordered, and confirm the validator
        # catches it.
        from qroute.metrics import compute_metrics
        from qroute.routers.base import RoutingResult

        c = Circuit(2)
        c.h(0)
        c.x(0)  # depends on the H above (same qubit)
        hw = HardwareGraph.from_line(2)

        broken = Circuit(2)
        broken.add_gate("X", (0,))  # wrong order: X before H
        broken.add_gate("H", (0,))

        mapping = Mapping.identity(2)
        metrics = compute_metrics(c, broken, [], 0.0)
        result = RoutingResult(
            routed_circuit=broken,
            inserted_swaps=[],
            initial_mapping=mapping,
            final_mapping=mapping,
            metrics=metrics,
            router_name="broken",
        )
        with pytest.raises(ValidationError):
            validate_routing_result(c, result, hw)

    def test_detects_wrong_final_mapping(self):
        from qroute.metrics import compute_metrics
        from qroute.routers.base import RoutingResult

        c = Circuit(2)
        c.cx(0, 1)
        hw = HardwareGraph.from_line(2)
        routed = Circuit(2)
        routed.add_gate("CX", (0, 1))
        wrong_final = Mapping({0: 1, 1: 0})  # claims a swap happened, but none did
        metrics = compute_metrics(c, routed, [], 0.0)
        result = RoutingResult(
            routed_circuit=routed,
            inserted_swaps=[],
            initial_mapping=Mapping.identity(2),
            final_mapping=wrong_final,
            metrics=metrics,
            router_name="broken",
        )
        with pytest.raises(ValidationError):
            validate_routing_result(c, result, hw)


class TestStatevectorEquivalence:
    def test_equivalent_after_routing(self):
        c = _sample_circuit()
        hw = HardwareGraph.from_line(4)
        result = NaiveRouter(hw).route(c)
        assert verify_equivalence_statevector(c, result, hw) is True

    def test_more_hardware_qubits_than_logical(self):
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)
        hw = HardwareGraph.from_line(4)
        result = NaiveRouter(hw).route(c)
        assert verify_equivalence_statevector(c, result, hw) is True

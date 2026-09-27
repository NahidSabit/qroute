import pytest

from qroute.circuit import Circuit
from qroute.generators import random_circuit
from qroute.hardware import HardwareGraph
from qroute.mapping import Mapping
from qroute.routers.astar import AStarRouter
from qroute.routers.greedy import GreedyRouter
from qroute.routers.lookahead import LookaheadRouter
from qroute.routers.naive import NaiveRouter
from qroute.routers.shortest_path import ShortestPathRouter
from qroute.validation import validate_routing_result, verify_equivalence_statevector

MAIN_ROUTERS = [NaiveRouter, ShortestPathRouter, GreedyRouter, LookaheadRouter]
ALL_ROUTERS = MAIN_ROUTERS + [AStarRouter]


@pytest.fixture(params=MAIN_ROUTERS)
def router_class(request):
    return request.param


class TestNoSwapNeeded:
    def test_already_adjacent_gate_needs_no_swap(self, router_class):
        hw = HardwareGraph.from_line(3)
        c = Circuit(2)
        c.cx(0, 1)
        result = router_class(hw).route(c)
        assert result.inserted_swaps == []
        validate_routing_result(c, result, hw)


class TestOneSwapNeeded:
    def test_distance_two_needs_exactly_one_swap(self, router_class):
        hw = HardwareGraph.from_line(3)
        c = Circuit(3)
        c.cx(0, 2)
        result = router_class(hw).route(c)
        assert len(result.inserted_swaps) == 1
        validate_routing_result(c, result, hw)


class TestLineTopology:
    def test_line_topology_routes_correctly(self, router_class):
        hw = HardwareGraph.from_line(5)
        c = random_circuit(5, 20, two_qubit_fraction=0.6, seed=1)
        result = router_class(hw).route(c)
        validate_routing_result(c, result, hw)


class TestGridTopology:
    def test_grid_topology_routes_correctly(self, router_class):
        hw = HardwareGraph.from_grid(3, 3)
        c = random_circuit(6, 25, two_qubit_fraction=0.6, seed=2)
        result = router_class(hw).route(c)
        validate_routing_result(c, result, hw)


class TestRepeatedInteractions:
    def test_repeated_gate_between_same_pair(self, router_class):
        hw = HardwareGraph.from_line(4)
        c = Circuit(4)
        for _ in range(5):
            c.cx(0, 3)
        result = router_class(hw).route(c)
        validate_routing_result(c, result, hw)


class TestMappingRemainsValid:
    def test_final_mapping_is_valid(self, router_class):
        hw = HardwareGraph.from_grid(3, 3)
        c = random_circuit(7, 20, two_qubit_fraction=0.5, seed=3)
        result = router_class(hw).route(c)
        assert result.final_mapping.is_valid()
        assert result.initial_mapping.is_valid()


class TestEveryTwoQubitGateHardwareValid:
    def test_all_two_qubit_gates_adjacent(self, router_class):
        hw = HardwareGraph.from_ring(6)
        c = random_circuit(6, 20, two_qubit_fraction=0.6, seed=4)
        result = router_class(hw).route(c)
        for gate in result.routed_circuit.two_qubit_gates:
            assert hw.are_adjacent(*gate.qubits)


class TestEdgeCases:
    def test_empty_circuit(self, router_class):
        hw = HardwareGraph.from_line(3)
        c = Circuit(3)
        result = router_class(hw).route(c)
        assert result.inserted_swaps == []
        validate_routing_result(c, result, hw)

    def test_single_qubit_circuit(self, router_class):
        hw = HardwareGraph.from_line(3)
        c = Circuit(1)
        c.h(0)
        c.x(0)
        result = router_class(hw).route(c)
        validate_routing_result(c, result, hw)

    def test_only_single_qubit_gates(self, router_class):
        hw = HardwareGraph.from_line(3)
        c = Circuit(3)
        c.h(0)
        c.x(1)
        c.z(2)
        result = router_class(hw).route(c)
        assert result.inserted_swaps == []
        validate_routing_result(c, result, hw)

    def test_complete_hardware_graph_never_needs_swaps(self, router_class):
        hw = HardwareGraph.from_complete(5)
        c = random_circuit(5, 20, two_qubit_fraction=0.8, seed=5)
        result = router_class(hw).route(c)
        assert result.inserted_swaps == []

    def test_more_logical_qubits_than_physical_raises(self, router_class):
        hw = HardwareGraph.from_line(2)
        c = Circuit(4)
        with pytest.raises(ValueError):
            router_class(hw).route(c)

    def test_fewer_logical_qubits_than_physical(self, router_class):
        hw = HardwareGraph.from_grid(3, 3)
        c = Circuit(3)
        c.cx(0, 1)
        c.cx(1, 2)
        result = router_class(hw).route(c)
        validate_routing_result(c, result, hw)


class TestRandomizedValidation:
    @pytest.mark.parametrize("seed", range(10))
    def test_seeded_random_circuits_validate_on_all_main_routers(self, seed):
        hw = HardwareGraph.from_grid(3, 3)
        c = random_circuit(6, 18, two_qubit_fraction=0.5, seed=seed)
        for router_cls in MAIN_ROUTERS:
            result = router_cls(hw).route(c)
            validate_routing_result(c, result, hw)


class TestQuantumEquivalence:
    @pytest.mark.parametrize("seed", range(5))
    def test_routed_circuit_equivalent_to_original(self, seed):
        hw = HardwareGraph.from_line(4)
        c = random_circuit(4, 10, two_qubit_fraction=0.6, seed=seed)
        for router_cls in MAIN_ROUTERS:
            result = router_cls(hw).route(c)
            assert verify_equivalence_statevector(c, result, hw)


class TestAStar:
    def test_astar_matches_or_beats_naive_on_tiny_instance(self):
        hw = HardwareGraph.from_line(3)
        c = Circuit(3)
        c.cx(0, 2)
        c.cx(1, 2)
        result_astar = AStarRouter(hw).route(c)
        result_naive = NaiveRouter(hw).route(c)
        validate_routing_result(c, result_astar, hw)
        assert len(result_astar.inserted_swaps) <= len(result_naive.inserted_swaps)

    def test_astar_raises_when_state_budget_exceeded(self):
        hw = HardwareGraph.from_line(4)
        c = random_circuit(4, 6, two_qubit_fraction=0.7, seed=9)
        with pytest.raises(RuntimeError):
            AStarRouter(hw, max_explored_states=1).route(c)


class TestInitialMapping:
    def test_custom_initial_mapping_is_respected(self, router_class):
        hw = HardwareGraph.from_line(3)
        c = Circuit(3)
        c.cx(0, 2)
        custom = Mapping({0: 2, 1: 1, 2: 0})
        result = router_class(hw).route(c, initial_mapping=custom)
        assert result.initial_mapping.logical_to_physical == custom.logical_to_physical
        validate_routing_result(c, result, hw)

    def test_default_mapping_is_identity(self, router_class):
        hw = HardwareGraph.from_line(3)
        c = Circuit(2)
        c.cx(0, 1)
        result = router_class(hw).route(c)
        assert result.initial_mapping.logical_to_physical == {0: 0, 1: 1}


class TestLookaheadParameters:
    def test_lookahead_size_and_weight_are_configurable(self):
        hw = HardwareGraph.from_line(6)
        c = random_circuit(6, 20, two_qubit_fraction=0.6, seed=7)
        router = LookaheadRouter(hw, lookahead_size=8, lookahead_weight=0.5)
        result = router.route(c)
        validate_routing_result(c, result, hw)

    def test_zero_lookahead_weight_reduces_to_greedy_like_behavior(self):
        hw = HardwareGraph.from_line(6)
        c = random_circuit(6, 20, two_qubit_fraction=0.6, seed=7)
        result_zero = LookaheadRouter(hw, lookahead_weight=0.0).route(c)
        result_greedy = GreedyRouter(hw).route(c)
        validate_routing_result(c, result_zero, hw)
        # Not required to be identical (candidate tie-breaking differs
        # slightly), but should be in the same ballpark.
        assert abs(len(result_zero.inserted_swaps) - len(result_greedy.inserted_swaps)) <= 5

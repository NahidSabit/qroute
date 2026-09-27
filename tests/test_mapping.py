import pytest

from qroute.generators import star_circuit
from qroute.hardware import HardwareGraph
from qroute.mapping import Mapping


class TestConstruction:
    def test_identity(self):
        m = Mapping.identity(3)
        assert m.physical_of(0) == 0
        assert m.physical_of(2) == 2

    def test_duplicate_physical_assignment_rejected(self):
        with pytest.raises(ValueError):
            Mapping({0: 1, 1: 1})

    def test_negative_index_rejected(self):
        with pytest.raises(ValueError):
            Mapping({0: -1})

    def test_random_mapping_is_valid_and_uses_given_physical_qubits(self):
        m = Mapping.random(3, physical_qubits=[0, 1, 2, 3, 4], seed=0)
        assert m.is_valid()
        assert m.num_logical_qubits() == 3
        for p in m.logical_to_physical.values():
            assert p in {0, 1, 2, 3, 4}

    def test_random_mapping_too_many_logical_qubits_raises(self):
        with pytest.raises(ValueError):
            Mapping.random(5, physical_qubits=[0, 1, 2])


class TestQueries:
    def test_physical_of_and_logical_of_are_inverses(self):
        m = Mapping({0: 2, 1: 0, 2: 1})
        assert m.logical_of(2) == 0
        assert m.logical_of(0) == 1
        assert m.logical_of(1) == 2

    def test_logical_of_unmapped_physical_is_none(self):
        m = Mapping({0: 0})
        assert m.logical_of(5) is None


class TestSwap:
    def test_swap_updates_both_directions(self):
        m = Mapping({0: 0, 1: 1})
        m.apply_swap(0, 1)
        assert m.physical_of(0) == 1
        assert m.physical_of(1) == 0
        assert m.logical_of(0) == 1
        assert m.logical_of(1) == 0
        assert m.is_valid()

    def test_swap_with_unoccupied_physical_qubit(self):
        m = Mapping({0: 0})
        m.apply_swap(0, 5)
        assert m.physical_of(0) == 5
        assert m.logical_of(0) is None
        assert m.logical_of(5) == 0
        assert m.is_valid()

    def test_swap_self_inverse(self):
        m = Mapping({0: 0, 1: 1, 2: 2})
        original = m.copy()
        m.apply_swap(0, 1)
        m.apply_swap(0, 1)
        assert m.logical_to_physical == original.logical_to_physical


class TestCopy:
    def test_copy_is_independent(self):
        m = Mapping({0: 0})
        m2 = m.copy()
        m2.apply_swap(0, 1)
        assert m.physical_of(0) == 0
        assert m2.physical_of(0) == 1


class TestHeuristicMapping:
    def test_hub_qubit_goes_to_highest_degree_physical_qubit(self):
        circuit = star_circuit(5, seed=0)  # qubit 0 is the hub
        hw = HardwareGraph.from_grid(3, 3)  # center qubit (4) has degree 4
        m = Mapping.from_interaction_heuristic(circuit, hw)
        assert m.physical_of(0) == 4
        assert m.is_valid()

    def test_covers_all_logical_qubits(self):
        circuit = star_circuit(4, seed=0)
        hw = HardwareGraph.from_line(4)
        m = Mapping.from_interaction_heuristic(circuit, hw)
        assert m.num_logical_qubits() == 4

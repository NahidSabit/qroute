from qroute.circuit import Circuit
from qroute.dependency import DependencyGraph


class TestFrontLayer:
    def test_gates_sharing_a_qubit_are_ordered(self):
        c = Circuit(1)
        c.h(0)
        c.x(0)
        dep = DependencyGraph(c)
        # gate 1 depends on gate 0
        assert dep.front_layer(set()) == [0]
        assert dep.front_layer({0}) == [1]

    def test_independent_gates_both_in_front_layer(self):
        c = Circuit(2)
        c.h(0)
        c.h(1)
        dep = DependencyGraph(c)
        assert set(dep.front_layer(set())) == {0, 1}

    def test_two_qubit_gate_depends_on_both_qubits(self):
        c = Circuit(2)
        c.h(0)
        c.h(1)
        c.cx(0, 1)
        dep = DependencyGraph(c)
        assert set(dep.front_layer(set())) == {0, 1}
        assert dep.front_layer({0}) == [1]  # gate 2 still needs gate 1 (h on qubit1) done
        assert dep.front_layer({0, 1}) == [2]

    def test_front_layer_updates_as_gates_complete(self):
        c = Circuit(2)
        c.cx(0, 1)
        c.h(0)
        c.h(1)
        dep = DependencyGraph(c)
        assert dep.front_layer(set()) == [0]
        assert set(dep.front_layer({0})) == {1, 2}

    def test_is_done(self):
        c = Circuit(1)
        c.h(0)
        dep = DependencyGraph(c)
        assert not dep.is_done(set())
        assert dep.is_done({0})

    def test_empty_circuit_is_done(self):
        c = Circuit(2)
        dep = DependencyGraph(c)
        assert dep.is_done(set())


class TestUpcoming:
    def test_upcoming_excludes_completed_and_front(self):
        c = Circuit(3)
        c.cx(0, 1)  # 0
        c.cx(1, 2)  # 1
        c.h(0)  # 2
        c.cx(0, 2)  # 3
        dep = DependencyGraph(c)
        front = [0]
        upcoming = dep.upcoming_two_qubit_gates(set(), front, size=5)
        assert upcoming == [1, 3]

    def test_upcoming_respects_size_limit(self):
        c = Circuit(2)
        c.cx(0, 1)
        c.h(0)
        c.cx(0, 1)
        c.cx(0, 1)
        dep = DependencyGraph(c)
        upcoming = dep.upcoming_two_qubit_gates(set(), [0], size=1)
        assert len(upcoming) == 1


import pytest

from qroute.circuit import Circuit, Gate


class TestGate:
    def test_valid_single_qubit_gate(self):
        g = Gate("H", (0,))
        assert g.name == "H"
        assert g.qubits == (0,)
        assert not g.is_two_qubit

    def test_valid_two_qubit_gate(self):
        g = Gate("CX", (0, 1))
        assert g.is_two_qubit

    def test_unsupported_gate_name(self):
        with pytest.raises(ValueError):
            Gate("TOFFOLI", (0, 1, 2))

    def test_single_qubit_gate_wrong_arity(self):
        with pytest.raises(ValueError):
            Gate("H", (0, 1))

    def test_two_qubit_gate_wrong_arity(self):
        with pytest.raises(ValueError):
            Gate("CX", (0,))

    def test_two_qubit_gate_same_qubit_twice(self):
        with pytest.raises(ValueError):
            Gate("CX", (0, 0))

    def test_parametric_gate_requires_param(self):
        with pytest.raises(ValueError):
            Gate("RX", (0,))

    def test_negative_qubit_index_rejected(self):
        with pytest.raises(ValueError):
            Gate("H", (-1,))


class TestCircuit:
    def test_add_gates_via_convenience_methods(self):
        c = Circuit(3)
        c.h(0)
        c.cx(0, 1)
        c.rz(2, 1.23)
        assert c.gate_count() == 3
        assert c.two_qubit_gate_count() == 1

    def test_invalid_qubit_index_raises(self):
        c = Circuit(2)
        with pytest.raises(ValueError):
            c.h(5)

    def test_two_qubit_gates_property(self):
        c = Circuit(3)
        c.h(0)
        c.cx(0, 1)
        c.cz(1, 2)
        assert [g.name for g in c.two_qubit_gates] == ["CX", "CZ"]

    def test_depth_single_qubit_chain(self):
        c = Circuit(1)
        c.h(0)
        c.x(0)
        c.z(0)
        assert c.depth() == 3

    def test_depth_parallel_single_qubit_gates(self):
        c = Circuit(2)
        c.h(0)
        c.h(1)
        assert c.depth() == 1

    def test_depth_two_qubit_gate_synchronizes(self):
        c = Circuit(2)
        c.h(0)
        c.h(1)
        c.h(1)
        c.cx(0, 1)
        # qubit0: slot0 (h). qubit1: slot0,slot1 (h,h). cx needs slot after
        # both -> slot2.
        assert c.depth() == 3

    def test_depth_empty_circuit(self):
        c = Circuit(3)
        assert c.depth() == 0

    def test_copy_is_independent(self):
        c = Circuit(1)
        c.h(0)
        c2 = c.copy()
        c2.x(0)
        assert c.gate_count() == 1
        assert c2.gate_count() == 2


class TestQiskitConversion:
    def test_round_trip(self):
        c = Circuit(3)
        c.h(0)
        c.cx(0, 1)
        c.rz(2, 0.5)
        c.cz(1, 2)
        c.swap(0, 2)

        qc = c.to_qiskit()
        c2 = Circuit.from_qiskit(qc)

        assert c2.num_qubits == c.num_qubits
        assert len(c2.gates) == len(c.gates)
        for g1, g2 in zip(c.gates, c2.gates):
            assert g1.name == g2.name
            assert g1.qubits == g2.qubits
            assert g1.params == pytest.approx(g2.params)

    def test_from_qiskit_rejects_unsupported_gate(self):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.ccx(0, 1, 0) if False else qc.t(0)  # T gate is not in QRoute's gate set
        with pytest.raises(ValueError):
            Circuit.from_qiskit(qc)

    def test_to_qiskit_gate_count_matches(self):
        c = Circuit(2)
        c.h(0)
        c.cx(0, 1)
        qc = c.to_qiskit()
        assert len(qc.data) == 2

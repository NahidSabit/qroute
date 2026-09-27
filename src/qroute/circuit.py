"""A small, explicit representation for quantum circuits.

QRoute does not reuse Qiskit's internal circuit representation for routing.
Instead it defines a minimal ``Circuit``/``Gate`` pair that is easy to read,
easy to test, and easy to translate to and from a Qiskit ``QuantumCircuit``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Gates that act on exactly one qubit and therefore never need routing.
SINGLE_QUBIT_GATES = {"X", "Y", "Z", "H"}

# Parameterized single-qubit rotation gates (one parameter: the angle).
PARAMETRIC_SINGLE_QUBIT_GATES = {"RX", "RY", "RZ"}

# Gates that act on exactly two qubits and therefore require the two
# physical qubits they touch to be adjacent on the hardware graph.
TWO_QUBIT_GATES = {"CX", "CZ", "SWAP"}

ALL_GATES = SINGLE_QUBIT_GATES | PARAMETRIC_SINGLE_QUBIT_GATES | TWO_QUBIT_GATES


@dataclass(frozen=True)
class Gate:
    """A single operation in a :class:`Circuit`.

    Attributes:
        name: One of the supported gate names (see ``ALL_GATES``).
        qubits: The qubit indices the gate acts on, in a fixed, meaningful
            order (e.g. for ``CX`` the order is ``(control, target)``).
        params: Numeric parameters, e.g. a rotation angle for ``RX``.
    """

    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.name not in ALL_GATES:
            raise ValueError(
                f"Unsupported gate '{self.name}'. Supported gates: {sorted(ALL_GATES)}"
            )
        if self.name in SINGLE_QUBIT_GATES or self.name in PARAMETRIC_SINGLE_QUBIT_GATES:
            if len(self.qubits) != 1:
                raise ValueError(f"Gate '{self.name}' requires exactly 1 qubit, got {self.qubits}")
        elif self.name in TWO_QUBIT_GATES:
            if len(self.qubits) != 2:
                raise ValueError(f"Gate '{self.name}' requires exactly 2 qubits, got {self.qubits}")
            if self.qubits[0] == self.qubits[1]:
                raise ValueError(f"Gate '{self.name}' cannot act twice on the same qubit {self.qubits}")
        if self.name in PARAMETRIC_SINGLE_QUBIT_GATES and len(self.params) != 1:
            raise ValueError(f"Gate '{self.name}' requires exactly 1 parameter, got {self.params}")
        if any(q < 0 for q in self.qubits):
            raise ValueError(f"Qubit indices must be non-negative, got {self.qubits}")

    @property
    def is_two_qubit(self) -> bool:
        return self.name in TWO_QUBIT_GATES

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if self.params:
            return f"{self.name}({', '.join(f'{p:.3f}' for p in self.params)}){list(self.qubits)}"
        return f"{self.name}{list(self.qubits)}"


class Circuit:
    """An ordered list of gates over a fixed number of qubits."""

    def __init__(self, num_qubits: int):
        if num_qubits < 0:
            raise ValueError("num_qubits must be non-negative")
        self.num_qubits = num_qubits
        self.gates: list[Gate] = []

    # -- construction -----------------------------------------------------
    def add_gate(self, name: str, qubits: tuple[int, ...], params: tuple[float, ...] = ()) -> Gate:
        for q in qubits:
            if q >= self.num_qubits:
                raise ValueError(
                    f"Qubit index {q} out of range for circuit with {self.num_qubits} qubits"
                )
        gate = Gate(name, tuple(qubits), tuple(params))
        self.gates.append(gate)
        return gate

    def x(self, q: int) -> Gate:
        return self.add_gate("X", (q,))

    def y(self, q: int) -> Gate:
        return self.add_gate("Y", (q,))

    def z(self, q: int) -> Gate:
        return self.add_gate("Z", (q,))

    def h(self, q: int) -> Gate:
        return self.add_gate("H", (q,))

    def rx(self, q: int, theta: float) -> Gate:
        return self.add_gate("RX", (q,), (theta,))

    def ry(self, q: int, theta: float) -> Gate:
        return self.add_gate("RY", (q,), (theta,))

    def rz(self, q: int, theta: float) -> Gate:
        return self.add_gate("RZ", (q,), (theta,))

    def cx(self, control: int, target: int) -> Gate:
        return self.add_gate("CX", (control, target))

    def cz(self, q1: int, q2: int) -> Gate:
        return self.add_gate("CZ", (q1, q2))

    def swap(self, q1: int, q2: int) -> Gate:
        return self.add_gate("SWAP", (q1, q2))

    # -- inspection ---------------------------------------------------------
    @property
    def two_qubit_gates(self) -> list[Gate]:
        return [g for g in self.gates if g.is_two_qubit]

    def gate_count(self) -> int:
        return len(self.gates)

    def two_qubit_gate_count(self) -> int:
        return len(self.two_qubit_gates)

    def depth(self) -> int:
        """Circuit depth: the length of the longest chain of dependent gates.

        Two gates are dependent if they share a qubit. This is computed by
        tracking, for each qubit, the "time slot" of the last gate that used
        it, and placing each new gate one slot after the latest slot among
        its qubits.
        """
        last_slot: dict[int, int] = {}
        max_slot = 0
        for gate in self.gates:
            slot = max((last_slot.get(q, -1) for q in gate.qubits), default=-1) + 1
            for q in gate.qubits:
                last_slot[q] = slot
            max_slot = max(max_slot, slot)
        return max_slot + 1 if self.gates else 0

    def copy(self) -> Circuit:
        new = Circuit(self.num_qubits)
        new.gates = list(self.gates)
        return new

    def __len__(self) -> int:
        return len(self.gates)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Circuit(num_qubits={self.num_qubits}, gates={len(self.gates)})"

    # -- Qiskit interop -------------------------------------------------
    def to_qiskit(self):
        """Convert to a ``qiskit.QuantumCircuit``."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(self.num_qubits)
        for gate in self.gates:
            if gate.name == "X":
                qc.x(gate.qubits[0])
            elif gate.name == "Y":
                qc.y(gate.qubits[0])
            elif gate.name == "Z":
                qc.z(gate.qubits[0])
            elif gate.name == "H":
                qc.h(gate.qubits[0])
            elif gate.name == "RX":
                qc.rx(gate.params[0], gate.qubits[0])
            elif gate.name == "RY":
                qc.ry(gate.params[0], gate.qubits[0])
            elif gate.name == "RZ":
                qc.rz(gate.params[0], gate.qubits[0])
            elif gate.name == "CX":
                qc.cx(gate.qubits[0], gate.qubits[1])
            elif gate.name == "CZ":
                qc.cz(gate.qubits[0], gate.qubits[1])
            elif gate.name == "SWAP":
                qc.swap(gate.qubits[0], gate.qubits[1])
            else:  # pragma: no cover - guarded by Gate.__post_init__
                raise ValueError(f"Unsupported gate '{gate.name}'")
        return qc

    @classmethod
    def from_qiskit(cls, qc) -> Circuit:
        """Build a :class:`Circuit` from a ``qiskit.QuantumCircuit``.

        Only the gate set supported by QRoute (see ``ALL_GATES``) is
        accepted; anything else raises a ``ValueError`` naming the
        unsupported instruction.
        """
        name_map = {
            "x": "X",
            "y": "Y",
            "z": "Z",
            "h": "H",
            "rx": "RX",
            "ry": "RY",
            "rz": "RZ",
            "cx": "CX",
            "cz": "CZ",
            "swap": "SWAP",
        }
        circuit = cls(qc.num_qubits)
        for instruction in qc.data:
            op = instruction.operation
            qiskit_name = op.name
            if qiskit_name not in name_map:
                raise ValueError(
                    f"Qiskit instruction '{qiskit_name}' is not supported by QRoute's "
                    f"circuit representation. Supported: {sorted(name_map)}"
                )
            qubit_indices = tuple(qc.find_bit(q).index for q in instruction.qubits)
            params = tuple(float(p) for p in op.params)
            circuit.add_gate(name_map[qiskit_name], qubit_indices, params)
        return circuit

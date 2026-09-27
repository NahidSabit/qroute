"""Round-trip a Qiskit circuit through QRoute.

Builds a circuit with Qiskit, converts it to QRoute's internal
representation, routes it, converts the result back to Qiskit, and checks
(via statevector simulation) that the routed circuit is equivalent to the
original, up to the final qubit permutation.

Usage:
    python examples/qiskit_example.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qiskit import QuantumCircuit

from qroute.circuit import Circuit
from qroute.hardware import HardwareGraph
from qroute.routers.greedy import GreedyRouter
from qroute.validation import validate_routing_result, verify_equivalence_statevector


def main() -> None:
    # 1. Build a circuit with Qiskit directly.
    qc = QuantumCircuit(5)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 4)
    qc.cx(0, 4)  # long-range interaction
    qc.rz(0.3, 2)

    print("Original Qiskit circuit:")
    print(qc.draw(output="text"))

    # 2. Convert it into QRoute's circuit representation.
    circuit = Circuit.from_qiskit(qc)

    # 3. Route it onto a 3x3 grid device.
    hardware = HardwareGraph.from_grid(3, 3)
    router = GreedyRouter(hardware)
    result = router.route(circuit)
    validate_routing_result(circuit, result, hardware)

    # 4. Convert the routed circuit back to Qiskit.
    routed_qc = result.routed_circuit.to_qiskit()

    print(f"\nInserted {len(result.inserted_swaps)} SWAP(s) to route onto the 3x3 grid.")
    print("\nRouted Qiskit circuit (physical qubits):")
    print(routed_qc.draw(output="text"))

    # 5. Confirm equivalence via statevector simulation (small circuit only).
    equivalent = verify_equivalence_statevector(circuit, result, hardware)
    print(f"\nRouted circuit verified equivalent to original (statevector check): {equivalent}")


if __name__ == "__main__":
    main()

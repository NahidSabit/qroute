"""A minimal end-to-end example: build a circuit, build hardware, route it.

Usage:
    python examples/basic_example.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qroute.circuit import Circuit
from qroute.hardware import HardwareGraph
from qroute.routers.lookahead import LookaheadRouter
from qroute.validation import validate_routing_result


def main() -> None:
    # A small logical circuit: a GHZ-like chain of entangling gates that
    # needs a "long-range" interaction between qubits 0 and 3.
    circuit = Circuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)
    circuit.cx(0, 3)  # not adjacent on a line -- this needs routing

    # Hardware: a 4-qubit line, 0-1-2-3. Qubits 0 and 3 are not adjacent.
    hardware = HardwareGraph.from_line(4)

    router = LookaheadRouter(hardware, lookahead_size=4, lookahead_weight=0.5)
    result = router.route(circuit)

    # Always validate: the router's own claims about what it did are
    # re-derived and checked independently.
    validate_routing_result(circuit, result, hardware)

    print("Original circuit:")
    for gate in circuit.gates:
        print(f"  {gate}")

    print(f"\nInserted {len(result.inserted_swaps)} SWAP(s):")
    for a, b in result.inserted_swaps:
        print(f"  SWAP(physical {a}, physical {b})")

    print("\nRouted circuit (physical qubits):")
    for gate in result.routed_circuit.gates:
        print(f"  {gate}")

    print(f"\nInitial mapping (logical -> physical): {result.initial_mapping.logical_to_physical}")
    print(f"Final mapping   (logical -> physical): {result.final_mapping.logical_to_physical}")

    print("\nMetrics:")
    for key, value in result.metrics.as_dict().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

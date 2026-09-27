"""Metrics describing the cost of a routing solution."""

from __future__ import annotations

from dataclasses import dataclass

from qroute.circuit import Circuit

# QRoute does not simulate a real gate-decomposition compiler; it uses the
# textbook fact that a SWAP can be built from three CX gates as a simple,
# standard way to translate a SWAP count into an estimated CX overhead.
CX_PER_SWAP = 3


@dataclass
class RoutingMetrics:
    """Summary statistics comparing an original circuit to its routed form."""

    original_gate_count: int
    routed_gate_count: int
    inserted_swap_count: int
    original_two_qubit_gate_count: int
    final_two_qubit_gate_count: int
    original_depth: int
    routed_depth: int
    depth_overhead: int
    gate_overhead: int
    estimated_extra_cx: int
    routing_runtime: float

    def as_dict(self) -> dict:
        return {
            "original_gate_count": self.original_gate_count,
            "routed_gate_count": self.routed_gate_count,
            "inserted_swap_count": self.inserted_swap_count,
            "original_two_qubit_gate_count": self.original_two_qubit_gate_count,
            "final_two_qubit_gate_count": self.final_two_qubit_gate_count,
            "original_depth": self.original_depth,
            "routed_depth": self.routed_depth,
            "depth_overhead": self.depth_overhead,
            "gate_overhead": self.gate_overhead,
            "estimated_extra_cx": self.estimated_extra_cx,
            "routing_runtime": self.routing_runtime,
        }


def compute_metrics(
    original_circuit: Circuit,
    routed_circuit: Circuit,
    inserted_swaps: list[tuple[int, int]],
    routing_runtime: float,
) -> RoutingMetrics:
    original_depth = original_circuit.depth()
    routed_depth = routed_circuit.depth()
    original_gate_count = original_circuit.gate_count()
    routed_gate_count = routed_circuit.gate_count()

    return RoutingMetrics(
        original_gate_count=original_gate_count,
        routed_gate_count=routed_gate_count,
        inserted_swap_count=len(inserted_swaps),
        original_two_qubit_gate_count=original_circuit.two_qubit_gate_count(),
        final_two_qubit_gate_count=routed_circuit.two_qubit_gate_count(),
        original_depth=original_depth,
        routed_depth=routed_depth,
        depth_overhead=routed_depth - original_depth,
        gate_overhead=routed_gate_count - original_gate_count,
        estimated_extra_cx=len(inserted_swaps) * CX_PER_SWAP,
        routing_runtime=routing_runtime,
    )

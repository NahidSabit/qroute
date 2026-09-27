"""Shared router interface and result type."""

from __future__ import annotations

from dataclasses import dataclass

from qroute.circuit import Circuit
from qroute.hardware import HardwareGraph
from qroute.mapping import Mapping
from qroute.metrics import RoutingMetrics


@dataclass
class RoutingResult:
    """The output of running a router on a circuit."""

    routed_circuit: Circuit
    inserted_swaps: list[tuple[int, int]]
    initial_mapping: Mapping
    final_mapping: Mapping
    metrics: RoutingMetrics
    router_name: str


class BaseRouter:
    """Common interface implemented by every routing strategy.

    Subclasses implement :meth:`route`. ``default_mapping`` is a shared
    helper: when the caller doesn't supply an initial mapping, every router
    falls back to the identity mapping (logical qubit ``i`` starts on
    physical qubit ``i``).
    """

    name: str = "base"

    def __init__(self, hardware: HardwareGraph):
        self.hardware = hardware

    def default_mapping(self, circuit: Circuit) -> Mapping:
        if circuit.num_qubits > self.hardware.num_qubits:
            raise ValueError(
                f"Circuit has {circuit.num_qubits} logical qubits, but the hardware only "
                f"has {self.hardware.num_qubits} physical qubits"
            )
        return Mapping.identity(circuit.num_qubits)

    def _resolve_initial_mapping(self, circuit: Circuit, initial_mapping: Mapping | None) -> Mapping:
        if initial_mapping is None:
            return self.default_mapping(circuit)
        if circuit.num_qubits > self.hardware.num_qubits:
            raise ValueError(
                f"Circuit has {circuit.num_qubits} logical qubits, but the hardware only "
                f"has {self.hardware.num_qubits} physical qubits"
            )
        return initial_mapping.copy()

    def route(self, circuit: Circuit, initial_mapping: Mapping | None = None) -> RoutingResult:
        raise NotImplementedError

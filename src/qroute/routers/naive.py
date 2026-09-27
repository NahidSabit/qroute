"""The naive baseline router.

This is deliberately the simplest correct approach: it processes gates in
their original program order (it does not even look at which gates are
independent), and whenever a two-qubit gate is blocked, it walks one of the
two logical qubits along a shortest hardware path towards the other,
inserting one SWAP per hop, until they are adjacent.

Its only job is to be obviously correct, so it acts as the reference point
every smarter router is compared against.
"""

from __future__ import annotations

import time

from qroute.circuit import Circuit
from qroute.metrics import compute_metrics
from qroute.routers.base import BaseRouter, RoutingResult


class NaiveRouter(BaseRouter):
    name = "naive"

    def route(self, circuit: Circuit, initial_mapping=None) -> RoutingResult:
        start = time.perf_counter()
        mapping = self._resolve_initial_mapping(circuit, initial_mapping)
        initial_mapping_recorded = mapping.copy()
        routed = Circuit(self.hardware.num_qubits)
        inserted_swaps: list[tuple[int, int]] = []

        for gate in circuit.gates:
            if not gate.is_two_qubit:
                p = mapping.physical_of(gate.qubits[0])
                routed.add_gate(gate.name, (p,), gate.params)
                continue

            l1, l2 = gate.qubits
            p1, p2 = mapping.physical_of(l1), mapping.physical_of(l2)
            # Move the qubit currently at p1 one step at a time towards p2
            # along a shortest path, until the pair is adjacent.
            while not self.hardware.are_adjacent(p1, p2):
                path = self.hardware.shortest_path(p1, p2)
                next_p = path[1]
                routed.add_gate("SWAP", (p1, next_p))
                mapping.apply_swap(p1, next_p)
                inserted_swaps.append((p1, next_p))
                p1 = next_p
            routed.add_gate(gate.name, (p1, p2), gate.params)

        runtime = time.perf_counter() - start
        metrics = compute_metrics(circuit, routed, inserted_swaps, runtime)
        return RoutingResult(
            routed_circuit=routed,
            inserted_swaps=inserted_swaps,
            initial_mapping=initial_mapping_recorded,
            final_mapping=mapping,
            metrics=metrics,
            router_name=self.name,
        )

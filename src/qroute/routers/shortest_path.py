"""The shortest-path router.

Like :class:`~qroute.routers.naive.NaiveRouter`, this router processes gates
in program order rather than reasoning about independent gates. The
difference is in *which* qubit it moves when a gate is blocked.

Instead of always moving one fixed endpoint all the way across, it moves
*both* endpoints towards each other, one hop each, per round. This still
uses exactly ``distance - 1`` SWAPs to bring a blocked pair together (the
same as the naive router), but it spreads the disruption to the mapping
across both sides of the path instead of concentrating it on one side. When
only one hop is needed to reach adjacency, there is nothing to balance, so
the router deterministically moves the first endpoint.
"""

from __future__ import annotations

import time

from qroute.circuit import Circuit
from qroute.metrics import compute_metrics
from qroute.routers.base import BaseRouter, RoutingResult


class ShortestPathRouter(BaseRouter):
    name = "shortest_path"

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

            while not self.hardware.are_adjacent(p1, p2):
                path = self.hardware.shortest_path(p1, p2)
                if len(path) > 3:
                    # Distance >= 3: move both endpoints one hop inward.
                    next_p1 = path[1]
                    routed.add_gate("SWAP", (p1, next_p1))
                    mapping.apply_swap(p1, next_p1)
                    inserted_swaps.append((p1, next_p1))
                    p1 = next_p1

                    # The path may be stale after the first swap only in the
                    # sense that its *endpoints* moved; since we only ever
                    # move along the previously-computed shortest path, the
                    # second-to-last node is still valid to move p2 towards.
                    next_p2 = path[-2]
                    if next_p2 != p1:
                        routed.add_gate("SWAP", (p2, next_p2))
                        mapping.apply_swap(p2, next_p2)
                        inserted_swaps.append((p2, next_p2))
                        p2 = next_p2
                else:
                    # Distance == 2: a single SWAP suffices. Deterministic
                    # tie-break: always move the first endpoint.
                    next_p1 = path[1]
                    routed.add_gate("SWAP", (p1, next_p1))
                    mapping.apply_swap(p1, next_p1)
                    inserted_swaps.append((p1, next_p1))
                    p1 = next_p1
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

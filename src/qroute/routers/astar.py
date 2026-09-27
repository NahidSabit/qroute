"""An A* router, intended only as a small-instance reference point.

This searches over *mapping states* rather than circuits: a state is
``(completed gate indices, current mapping)``, ``g(n)`` is the number of
SWAPs used to reach that state, and

.. math::

    f(n) = g(n) + h(n)

Gates that are already executable (single-qubit gates, or two-qubit gates
whose qubits are already adjacent) are executed for free -- they never
branch the search, since there is nothing to decide. The search only
branches on which SWAP to insert when the front layer contains a blocked
two-qubit gate.

**Heuristic.** ``h(n)`` is the *largest*, not the *sum*, of
``distance - 1`` over the front layer's two-qubit gates. Using the largest
single value is admissible (it never overestimates the true number of
SWAPs remaining): whichever front-layer gate is currently farthest apart
needs at least ``distance - 1`` SWAPs just for that gate, so the true
remaining cost can never be less than that value. A *sum* over all
front-layer gates would be tempting (and more informative), but it is
**not** guaranteed admissible: a single SWAP can simultaneously shrink the
distance of two different front-layer gates that share a qubit, so summing
independent lower bounds can overestimate the true minimum. This router
therefore does not claim to always find the strict global optimum for
*all* graphs and gate sets -- only that its heuristic never overestimates,
which is what A* needs to guarantee optimality given exhaustive search to
completion.

Because the state space grows very quickly, this router is only practical
for a handful of qubits and gates; it exists to sanity-check the other
routers on tiny instances, not to be used at scale.
"""

from __future__ import annotations

import heapq
import itertools
import time

from qroute.circuit import Circuit
from qroute.dependency import DependencyGraph
from qroute.mapping import Mapping
from qroute.metrics import compute_metrics
from qroute.routers.base import BaseRouter, RoutingResult


def _auto_execute(dep: DependencyGraph, hardware, circuit: Circuit, completed: set[int], mapping: Mapping, ops: list):
    """Execute every currently-free gate (single-qubit, or two-qubit and
    already adjacent) until none remain; this never costs a SWAP so it never
    branches the search."""
    completed = set(completed)
    while True:
        front = dep.front_layer(completed)
        progressed = False
        for i in front:
            gate = circuit.gates[i]
            if not gate.is_two_qubit:
                p = mapping.physical_of(gate.qubits[0])
                ops.append(("gate", gate.name, (p,), gate.params))
                completed.add(i)
                progressed = True
            else:
                l1, l2 = gate.qubits
                p1, p2 = mapping.physical_of(l1), mapping.physical_of(l2)
                if hardware.are_adjacent(p1, p2):
                    ops.append(("gate", gate.name, (p1, p2), gate.params))
                    completed.add(i)
                    progressed = True
        if not progressed:
            break
    return completed


class AStarRouter(BaseRouter):
    name = "astar"

    def __init__(self, hardware, max_explored_states: int = 20000):
        super().__init__(hardware)
        self.max_explored_states = max_explored_states

    def route(self, circuit: Circuit, initial_mapping=None) -> RoutingResult:
        start_time = time.perf_counter()
        mapping0 = self._resolve_initial_mapping(circuit, initial_mapping)
        initial_mapping_recorded = mapping0.copy()
        dep = DependencyGraph(circuit)

        ops: list = []
        completed0 = _auto_execute(dep, self.hardware, circuit, set(), mapping0, ops)
        if dep.is_done(completed0):
            return self._build_result(
                circuit, initial_mapping_recorded, mapping0, ops, [], start_time
            )

        def state_key(completed, mapping):
            return (frozenset(completed), tuple(sorted(mapping.physical_to_logical.items())))

        def heuristic(completed, mapping):
            front = dep.front_layer(completed)
            best = 0
            for i in front:
                gate = circuit.gates[i]
                if gate.is_two_qubit:
                    l1, l2 = gate.qubits
                    d = self.hardware.distance(mapping.physical_of(l1), mapping.physical_of(l2))
                    best = max(best, d - 1)
            return best

        counter = itertools.count()
        h0 = heuristic(completed0, mapping0)
        heap = [(h0, next(counter), 0, completed0, mapping0, ops, [])]
        best_g = {state_key(completed0, mapping0): 0}
        explored = 0

        while heap:
            _f, _, g, completed_c, mapping_c, ops_c, swaps_c = heapq.heappop(heap)
            key = state_key(completed_c, mapping_c)
            if best_g.get(key, float("inf")) < g:
                continue  # stale heap entry

            explored += 1
            if explored > self.max_explored_states:
                raise RuntimeError(
                    f"A* search explored more than {self.max_explored_states} states without "
                    "finishing. A* is only intended as a small-instance reference point; use "
                    "a smaller circuit/hardware graph or raise max_explored_states."
                )

            if dep.is_done(completed_c):
                return self._build_result(
                    circuit, initial_mapping_recorded, mapping_c, ops_c, swaps_c, start_time
                )

            front = dep.front_layer(completed_c)
            candidates: set[tuple[int, int]] = set()
            for i in front:
                gate = circuit.gates[i]
                if gate.is_two_qubit:
                    l1, l2 = gate.qubits
                    for p in (mapping_c.physical_of(l1), mapping_c.physical_of(l2)):
                        for nb in self.hardware.neighbors(p):
                            candidates.add(tuple(sorted((p, nb))))

            for a, b in sorted(candidates):
                new_mapping = mapping_c.copy()
                new_mapping.apply_swap(a, b)
                new_ops = ops_c + [("swap", a, b)]
                new_swaps = swaps_c + [(a, b)]
                new_completed = _auto_execute(dep, self.hardware, circuit, completed_c, new_mapping, new_ops)
                new_g = g + 1
                key2 = state_key(new_completed, new_mapping)
                if best_g.get(key2, float("inf")) <= new_g:
                    continue
                best_g[key2] = new_g
                h = heuristic(new_completed, new_mapping)
                heapq.heappush(heap, (new_g + h, next(counter), new_g, new_completed, new_mapping, new_ops, new_swaps))

        raise RuntimeError(  # pragma: no cover - unreachable on a connected hardware graph
            "A* search exhausted without finding a solution; this should not happen on a "
            "connected hardware graph"
        )

    def _build_result(self, circuit, initial_mapping, final_mapping, ops, swaps, start_time) -> RoutingResult:
        routed = Circuit(self.hardware.num_qubits)
        for op in ops:
            if op[0] == "gate":
                _, name, qubits, params = op
                routed.add_gate(name, qubits, params)
            else:
                _, a, b = op
                routed.add_gate("SWAP", (a, b))
        runtime = time.perf_counter() - start_time
        metrics = compute_metrics(circuit, routed, swaps, runtime)
        return RoutingResult(
            routed_circuit=routed,
            inserted_swaps=swaps,
            initial_mapping=initial_mapping,
            final_mapping=final_mapping,
            metrics=metrics,
            router_name=self.name,
        )

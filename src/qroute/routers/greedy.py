"""The greedy heuristic router.

Unlike the naive and shortest-path routers, the greedy router reasons about
the *front layer*: the set of gates that are currently executable given
which gates have already run (see :mod:`qroute.dependency`). When the front
layer contains a blocked two-qubit gate, it considers every SWAP adjacent to
a qubit involved in a front-layer interaction, and picks the one that
minimizes the total hardware distance of all front-layer interactions:

.. math::

    H(s) = \\sum_{g \\in F} d_g(s)

where :math:`F` is the front layer and :math:`d_g(s)` is the hardware
distance between gate :math:`g`'s two qubits after tentatively applying
candidate SWAP :math:`s`.

Loop prevention: a SWAP is never immediately reversed, states are checked
against a small "recently seen" history, and after too many SWAPs in a row
without executing a gate, the router falls back to shortest-path-style
progress on the most-blocked front-layer gate.
"""

from __future__ import annotations

import time

from qroute.circuit import Circuit
from qroute.dependency import DependencyGraph
from qroute.mapping import Mapping
from qroute.metrics import compute_metrics
from qroute.routers.base import BaseRouter, RoutingResult

# Safety valve: if a router inserts more SWAPs than this, something is wrong
# (e.g. a bug producing an infinite loop) rather than merely slow. This is
# far more than any of the circuits used in this project would need.
_ABSOLUTE_SWAP_LIMIT_MULTIPLIER = 200


class GreedyRouter(BaseRouter):
    name = "greedy"

    def __init__(self, hardware, max_no_progress: int = 30):
        super().__init__(hardware)
        self.max_no_progress = max_no_progress

    def route(self, circuit: Circuit, initial_mapping=None) -> RoutingResult:
        start = time.perf_counter()
        mapping = self._resolve_initial_mapping(circuit, initial_mapping)
        initial_mapping_recorded = mapping.copy()
        routed = Circuit(self.hardware.num_qubits)
        inserted_swaps: list[tuple[int, int]] = []

        self._circuit = circuit
        self._dep = DependencyGraph(circuit)
        dep = self._dep

        completed: set[int] = set()
        self._completed = completed  # exposed so subclasses (e.g. LookaheadRouter) can read it
        last_swap: tuple[int, int] | None = None
        no_progress_count = 0
        seen_states: dict[tuple, int] = {}
        swap_limit = max(50, _ABSOLUTE_SWAP_LIMIT_MULTIPLIER * max(1, len(circuit.gates)))

        while not dep.is_done(completed):
            front = dep.front_layer(completed)
            executed_any = False
            blocked: list[int] = []
            for i in front:
                gate = circuit.gates[i]
                if not gate.is_two_qubit:
                    p = mapping.physical_of(gate.qubits[0])
                    routed.add_gate(gate.name, (p,), gate.params)
                    completed.add(i)
                    executed_any = True
                    continue
                l1, l2 = gate.qubits
                p1, p2 = mapping.physical_of(l1), mapping.physical_of(l2)
                if self.hardware.are_adjacent(p1, p2):
                    routed.add_gate(gate.name, (p1, p2), gate.params)
                    completed.add(i)
                    executed_any = True
                else:
                    blocked.append(i)

            if executed_any:
                no_progress_count = 0
            if not blocked:
                continue

            front_after = dep.front_layer(completed)
            candidates = self._generate_candidates(front_after, mapping)
            if not candidates:  # pragma: no cover - only if a qubit is isolated
                raise RuntimeError("No candidate SWAPs available; hardware graph may be disconnected")

            chosen = self._choose_swap(front_after, mapping, candidates, last_swap)
            a, b = chosen
            routed.add_gate("SWAP", (a, b))
            mapping.apply_swap(a, b)
            inserted_swaps.append((a, b))
            last_swap = (a, b)
            if not executed_any:
                no_progress_count += 1

            state_key = (frozenset(completed), tuple(sorted(mapping.physical_to_logical.items())))
            seen_states[state_key] = seen_states.get(state_key, 0) + 1

            if len(inserted_swaps) > swap_limit:
                raise RuntimeError(
                    f"Router exceeded {swap_limit} SWAPs without finishing; this indicates a bug "
                    "rather than a merely hard instance"
                )

            if no_progress_count > self.max_no_progress or seen_states[state_key] > 3:
                self._force_progress(front_after, mapping, routed, inserted_swaps)
                last_swap = inserted_swaps[-1] if inserted_swaps else None
                no_progress_count = 0

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

    # -- helpers, overridable by subclasses --------------------------------
    def _generate_candidates(self, front: list[int], mapping: Mapping) -> set[tuple[int, int]]:
        """Every SWAP adjacent to a physical qubit involved in a front-layer
        two-qubit interaction."""
        touched_physical: set[int] = set()
        for i in front:
            gate = self._circuit.gates[i]
            if gate.is_two_qubit:
                l1, l2 = gate.qubits
                touched_physical.add(mapping.physical_of(l1))
                touched_physical.add(mapping.physical_of(l2))
        candidates: set[tuple[int, int]] = set()
        for p in touched_physical:
            for nb in self.hardware.neighbors(p):
                candidates.add(tuple(sorted((p, nb))))
        return candidates

    def _score(self, front: list[int], mapping: Mapping, a: int, b: int) -> float:
        trial = mapping.copy()
        trial.apply_swap(a, b)
        total = 0.0
        for i in front:
            gate = self._circuit.gates[i]
            if gate.is_two_qubit:
                l1, l2 = gate.qubits
                total += self.hardware.distance(trial.physical_of(l1), trial.physical_of(l2))
        return total

    def _choose_swap(
        self,
        front: list[int],
        mapping: Mapping,
        candidates: set[tuple[int, int]],
        last_swap: tuple[int, int] | None,
    ) -> tuple[int, int]:
        non_reversal = [
            c for c in candidates if last_swap is None or c not in (last_swap, tuple(reversed(last_swap)))
        ]
        pool = non_reversal if non_reversal else list(candidates)
        best = None
        best_score = None
        # Deterministic tie-breaking: iterate candidates in sorted order and
        # only replace the incumbent on a strictly better score.
        for a, b in sorted(pool):
            score = self._score(front, mapping, a, b)
            if best_score is None or score < best_score:
                best_score = score
                best = (a, b)
        return best

    def _force_progress(self, front, mapping, routed, inserted_swaps) -> None:
        """Loop-prevention fallback: pick the front-layer two-qubit gate with
        the largest current hardware distance and move one of its qubits one
        hop closer, shortest-path style (same move the naive router would
        make). This guarantees measurable progress on at least one gate."""
        worst_gate = None
        worst_distance = -1
        for i in front:
            gate = self._circuit.gates[i]
            if gate.is_two_qubit:
                l1, l2 = gate.qubits
                d = self.hardware.distance(mapping.physical_of(l1), mapping.physical_of(l2))
                if d > worst_distance:
                    worst_distance = d
                    worst_gate = gate
        if worst_gate is None:
            return
        l1, l2 = worst_gate.qubits
        p1, p2 = mapping.physical_of(l1), mapping.physical_of(l2)
        path = self.hardware.shortest_path(p1, p2)
        next_p = path[1]
        routed.add_gate("SWAP", (p1, next_p))
        mapping.apply_swap(p1, next_p)
        inserted_swaps.append((p1, next_p))

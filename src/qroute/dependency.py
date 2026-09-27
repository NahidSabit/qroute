"""Gate dependency DAG.

Two gates that act on the same qubit must keep their original relative
order (a qubit's state can only be used in the order its gates were
issued). Gates that share no qubit are independent and can, in principle,
execute in either order. This module builds that dependency DAG once per
circuit and exposes it through pure functions of an explicit ``completed``
set, so several routers (and a search-based router) can share one
implementation without stepping on each other's state.
"""

from __future__ import annotations

import networkx as nx

from qroute.circuit import Circuit


class DependencyGraph:
    """Dependency DAG over the gate indices of a :class:`Circuit`.

    Node ``i`` represents ``circuit.gates[i]``. There is an edge ``i -> j``
    if gate ``i`` is the most recent prior gate touching some qubit that
    gate ``j`` also touches (so ``i`` must execute before ``j``).
    """

    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(len(circuit.gates)))

        last_gate_for_qubit: dict[int, int] = {}
        for i, gate in enumerate(circuit.gates):
            predecessors = set()
            for q in gate.qubits:
                if q in last_gate_for_qubit:
                    predecessors.add(last_gate_for_qubit[q])
                last_gate_for_qubit[q] = i
            for p in predecessors:
                self.graph.add_edge(p, i)

    def front_layer(self, completed: set[int] | frozenset[int]) -> list[int]:
        """Gate indices that are not yet completed but whose predecessors
        all are -- i.e. the gates that are ready to execute right now."""
        return [
            i
            for i in self.graph.nodes
            if i not in completed
            and all(p in completed for p in self.graph.predecessors(i))
        ]

    def upcoming_two_qubit_gates(
        self, completed: set[int] | frozenset[int], front: list[int], size: int
    ) -> list[int]:
        """The next ``size`` two-qubit gates in program order that are
        neither completed nor already in the front layer.

        This is a simple, explainable notion of "upcoming work": scan the
        circuit in its original order and collect the next few two-qubit
        gates. It does not attempt to model exactly which gates become
        schedulable soonest -- that would require re-deriving front layers
        for hypothetical futures, which is unnecessary complexity for a
        lookahead heuristic.
        """
        front_set = set(front)
        result: list[int] = []
        for i, gate in enumerate(self.circuit.gates):
            if i in completed or i in front_set:
                continue
            if gate.is_two_qubit:
                result.append(i)
                if len(result) >= size:
                    break
        return result

    def is_done(self, completed: set[int] | frozenset[int]) -> bool:
        return len(completed) == self.graph.number_of_nodes()

    def __len__(self) -> int:
        return self.graph.number_of_nodes()

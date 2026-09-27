"""The lookahead router: the main algorithmic contribution of this project.

This extends :class:`~qroute.routers.greedy.GreedyRouter` by scoring
candidate SWAPs with both the current front layer *and* a window of
upcoming two-qubit gates:

.. math::

    H(s) = \\frac{1}{|F|} \\sum_{g \\in F} d_g(s)
         + \\lambda \\frac{1}{|E|} \\sum_{g \\in E} d_g(s)

where :math:`F` is the front layer, :math:`E` is a window of the next
``lookahead_size`` upcoming two-qubit gates (see
:meth:`~qroute.dependency.DependencyGraph.upcoming_two_qubit_gates`), and
``lookahead_weight`` (:math:`\\lambda`) controls how much the router should
care about setting up future interactions versus resolving current ones.
Everything else (candidate generation, reversal prevention, loop
prevention) is inherited unchanged from :class:`GreedyRouter`.
"""

from __future__ import annotations

from qroute.mapping import Mapping
from qroute.routers.greedy import GreedyRouter


class LookaheadRouter(GreedyRouter):
    name = "lookahead"

    def __init__(
        self,
        hardware,
        lookahead_size: int = 8,
        lookahead_weight: float = 0.5,
        max_no_progress: int = 30,
    ):
        super().__init__(hardware, max_no_progress=max_no_progress)
        self.lookahead_size = lookahead_size
        self.lookahead_weight = lookahead_weight

    def _score(self, front: list[int], mapping: Mapping, a: int, b: int) -> float:
        trial = mapping.copy()
        trial.apply_swap(a, b)

        front_two_qubit = [i for i in front if self._circuit.gates[i].is_two_qubit]
        front_term = 0.0
        if front_two_qubit:
            total = sum(
                self.hardware.distance(
                    trial.physical_of(self._circuit.gates[i].qubits[0]),
                    trial.physical_of(self._circuit.gates[i].qubits[1]),
                )
                for i in front_two_qubit
            )
            front_term = total / len(front_two_qubit)

        upcoming = self._dep.upcoming_two_qubit_gates(self._completed_snapshot, front, self.lookahead_size)
        future_term = 0.0
        if upcoming:
            total = sum(
                self.hardware.distance(
                    trial.physical_of(self._circuit.gates[i].qubits[0]),
                    trial.physical_of(self._circuit.gates[i].qubits[1]),
                )
                for i in upcoming
            )
            future_term = total / len(upcoming)

        return front_term + self.lookahead_weight * future_term

    def _choose_swap(self, front, mapping, candidates, last_swap):
        # `GreedyRouter.route` exposes the live `completed` set as
        # `self._completed`, which `upcoming_two_qubit_gates` needs to know
        # which gates have already executed.
        self._completed_snapshot = frozenset(self._completed)
        return super()._choose_swap(front, mapping, candidates, last_swap)

"""Logical-to-physical qubit mapping.

A ``Mapping`` tracks which physical qubit currently holds each logical
qubit, and its inverse. Routers repeatedly apply SWAPs to a ``Mapping`` as
they route a circuit, so keeping the two directions consistent (and cheap
to query) matters a lot for correctness.
"""

from __future__ import annotations

import random


class Mapping:
    """A one-to-one placement of logical qubits onto physical qubits.

    Not every physical qubit needs to hold a logical qubit (the hardware may
    have more physical qubits than the circuit has logical qubits), but every
    logical qubit must be placed, and no two logical qubits may share a
    physical qubit.
    """

    def __init__(self, logical_to_physical: dict[int, int]):
        self.logical_to_physical: dict[int, int] = dict(logical_to_physical)
        self.physical_to_logical: dict[int, int] = {
            p: l for l, p in self.logical_to_physical.items()
        }
        self._validate()

    def _validate(self) -> None:
        physical_values = list(self.logical_to_physical.values())
        if len(set(physical_values)) != len(physical_values):
            raise ValueError(
                f"Mapping is not one-to-one: physical qubits {physical_values} "
                "contain duplicates"
            )
        if any(p < 0 for p in physical_values) or any(l < 0 for l in self.logical_to_physical):
            raise ValueError("Logical and physical qubit indices must be non-negative")

    # -- constructors -----------------------------------------------------
    @classmethod
    def identity(cls, num_logical_qubits: int) -> Mapping:
        """Logical qubit ``i`` starts on physical qubit ``i``."""
        return cls({i: i for i in range(num_logical_qubits)})

    @classmethod
    def random(cls, num_logical_qubits: int, physical_qubits: list[int], seed: int | None = None) -> Mapping:
        """A uniformly random assignment of logical qubits to distinct
        physical qubits chosen from ``physical_qubits``."""
        if num_logical_qubits > len(physical_qubits):
            raise ValueError(
                f"Cannot place {num_logical_qubits} logical qubits onto "
                f"{len(physical_qubits)} physical qubits"
            )
        rng = random.Random(seed)
        chosen = rng.sample(list(physical_qubits), num_logical_qubits)
        return cls({logical: physical for logical, physical in enumerate(chosen)})

    # -- queries ------------------------------------------------------------
    def physical_of(self, logical: int) -> int:
        return self.logical_to_physical[logical]

    def logical_of(self, physical: int) -> int | None:
        return self.physical_to_logical.get(physical)

    def num_logical_qubits(self) -> int:
        return len(self.logical_to_physical)

    # -- mutation -------------------------------------------------------
    def apply_swap(self, p1: int, p2: int) -> None:
        """Physically swap whatever logical qubits (if any) sit at physical
        positions ``p1`` and ``p2``."""
        l1 = self.physical_to_logical.get(p1)
        l2 = self.physical_to_logical.get(p2)

        if l1 is not None:
            self.logical_to_physical[l1] = p2
        if l2 is not None:
            self.logical_to_physical[l2] = p1

        if l1 is not None:
            self.physical_to_logical[p2] = l1
        else:
            self.physical_to_logical.pop(p2, None)

        if l2 is not None:
            self.physical_to_logical[p1] = l2
        else:
            self.physical_to_logical.pop(p1, None)

    def copy(self) -> Mapping:
        new = Mapping.__new__(Mapping)
        new.logical_to_physical = dict(self.logical_to_physical)
        new.physical_to_logical = dict(self.physical_to_logical)
        return new

    def is_valid(self) -> bool:
        try:
            self._validate()
        except ValueError:
            return False
        return self.physical_to_logical == {p: l for l, p in self.logical_to_physical.items()}

    @classmethod
    def from_interaction_heuristic(cls, circuit, hardware) -> Mapping:
        """Place frequently-interacting logical qubits near well-connected
        physical qubits.

        Steps:

        1. Build a weighted logical interaction graph: an edge weight equal
           to how many times two logical qubits interact via a two-qubit
           gate anywhere in the circuit.
        2. Rank logical qubits by their weighted degree (total interaction
           weight), descending.
        3. Rank physical qubits by hardware degree (number of hardware
           neighbors), descending.
        4. Assign the most-interacting logical qubits to the
           highest-degree physical qubits, in rank order.

        This is a simple, greedy heuristic -- it does not attempt to solve
        the underlying graph-matching problem optimally, only to give
        routers a better starting point than an arbitrary or identity
        mapping when the circuit's interactions are uneven.
        """
        import networkx as nx

        weighted_graph = nx.Graph()
        weighted_graph.add_nodes_from(range(circuit.num_qubits))
        for gate in circuit.two_qubit_gates:
            a, b = gate.qubits
            if weighted_graph.has_edge(a, b):
                weighted_graph[a][b]["weight"] += 1
            else:
                weighted_graph.add_edge(a, b, weight=1)

        logical_weighted_degree = {
            q: sum(d["weight"] for d in weighted_graph[q].values()) for q in weighted_graph.nodes
        }
        logical_ranked = sorted(
            range(circuit.num_qubits), key=lambda q: (-logical_weighted_degree[q], q)
        )
        physical_ranked = sorted(
            range(hardware.num_qubits), key=lambda p: (-hardware.degree(p), p)
        )

        logical_to_physical = {
            logical: physical_ranked[i] for i, logical in enumerate(logical_ranked)
        }
        return cls(logical_to_physical)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        return self.logical_to_physical == other.logical_to_physical

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Mapping({self.logical_to_physical})"

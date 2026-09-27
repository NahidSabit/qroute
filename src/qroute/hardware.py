"""Hardware connectivity graphs.

Real quantum devices only allow two-qubit gates between specific pairs of
physical qubits. ``HardwareGraph`` wraps a ``networkx.Graph`` describing that
connectivity and precomputes all-pairs shortest-path distances, since every
router needs them repeatedly.
"""

from __future__ import annotations

import networkx as nx


class HardwareGraph:
    """Connectivity graph of physical qubits.

    Nodes are physical qubit indices ``0..num_qubits-1``. An edge means a
    two-qubit gate can be applied directly between that pair of qubits.
    """

    def __init__(self, graph: nx.Graph):
        if graph.number_of_nodes() == 0:
            raise ValueError("Hardware graph must have at least one qubit")
        nodes = sorted(graph.nodes)
        if nodes != list(range(len(nodes))):
            raise ValueError(
                "Hardware graph nodes must be exactly 0..num_qubits-1; "
                f"got {nodes}"
            )
        self.graph = graph
        self.num_qubits = graph.number_of_nodes()
        self._distances: dict[int, dict[int, int]] = dict(
            nx.all_pairs_shortest_path_length(graph)
        )

    # -- constructors -----------------------------------------------------
    @classmethod
    def from_line(cls, n: int) -> HardwareGraph:
        """A 1D chain: 0-1-2-...-(n-1)."""
        return cls(nx.path_graph(n))

    @classmethod
    def from_ring(cls, n: int) -> HardwareGraph:
        """A ring/cycle: 0-1-2-...-(n-1)-0."""
        if n < 3:
            raise ValueError("A ring requires at least 3 qubits")
        return cls(nx.cycle_graph(n))

    @classmethod
    def from_grid(cls, rows: int, cols: int) -> HardwareGraph:
        """A rectangular grid, qubits numbered row-major: (r, c) -> r*cols + c."""
        g = nx.grid_2d_graph(rows, cols)
        mapping = {(r, c): r * cols + c for r in range(rows) for c in range(cols)}
        g = nx.relabel_nodes(g, mapping)
        return cls(g)

    @classmethod
    def from_complete(cls, n: int) -> HardwareGraph:
        """A fully connected device: every pair of qubits is adjacent."""
        return cls(nx.complete_graph(n))

    @classmethod
    def from_edge_list(cls, n: int, edges: list[tuple[int, int]]) -> HardwareGraph:
        """An arbitrary connectivity graph given as an explicit edge list."""
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(edges)
        return cls(g)

    @classmethod
    def from_random_sparse(cls, n: int, p: float, seed: int | None = None) -> HardwareGraph:
        """A random Erdos-Renyi graph, resampled until connected.

        Args:
            n: number of physical qubits.
            p: edge probability.
            seed: random seed for reproducibility.
        """
        rng_seed = seed
        for attempt in range(1000):
            g = nx.erdos_renyi_graph(n, p, seed=None if rng_seed is None else rng_seed + attempt)
            if nx.is_connected(g):
                return cls(g)
        raise RuntimeError(
            f"Could not generate a connected random graph with n={n}, p={p} after 1000 attempts"
        )

    @classmethod
    def from_heavy_hex(cls, distance: int) -> HardwareGraph:
        """A small heavy-hex-inspired topology (IBM-style), for ``distance``
        in {1, 2, 3}. This is a simplified, illustrative version, not an
        exact reproduction of any specific real device.
        """
        if distance == 1:
            edges = [(0, 1), (1, 2), (0, 2)]
            n = 3
        elif distance == 2:
            # A small hexagonal ring with two "spoke" nodes, loosely
            # resembling the heavy-hex motif of alternating data/flag qubits.
            edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6), (3, 7)]
            n = 8
        else:
            # Two hexagons sharing an edge, with spokes - a small illustrative
            # heavy-hex-like lattice fragment.
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
                (3, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 3),
                (0, 11), (7, 12),
            ]
            n = 13
        return cls.from_edge_list(n, edges)

    # -- queries ------------------------------------------------------------
    def neighbors(self, q: int) -> list[int]:
        return list(self.graph.neighbors(q))

    def are_adjacent(self, a: int, b: int) -> bool:
        return self.graph.has_edge(a, b)

    def shortest_path(self, a: int, b: int) -> list[int]:
        """Return one shortest path between physical qubits ``a`` and ``b``."""
        return nx.shortest_path(self.graph, a, b)

    def distance(self, a: int, b: int) -> int:
        return self._distances[a][b]

    def all_distances(self) -> dict[int, dict[int, int]]:
        return self._distances

    def diameter(self) -> int:
        return nx.diameter(self.graph)

    def degree(self, q: int) -> int:
        return self.graph.degree[q]

    def degrees(self) -> dict[int, int]:
        return dict(self.graph.degree())

    def is_connected(self) -> bool:
        return nx.is_connected(self.graph)

    def edges(self) -> list[tuple[int, int]]:
        return list(self.graph.edges())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"HardwareGraph(num_qubits={self.num_qubits}, edges={self.graph.number_of_edges()})"

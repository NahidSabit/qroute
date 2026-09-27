"""Reproducible circuit and interaction-graph generators for experiments.

Everything here is seeded so results in ``experiments/`` can be reproduced
exactly from the recorded seed.
"""

from __future__ import annotations

import random

import networkx as nx

from qroute.circuit import Circuit


def random_circuit(
    num_qubits: int,
    num_gates: int,
    two_qubit_fraction: float = 0.5,
    seed: int | None = None,
) -> Circuit:
    """A circuit with ``num_gates`` gates, each independently two-qubit with
    probability ``two_qubit_fraction`` (a random ``CX`` or ``CZ`` on a random
    distinct pair) and otherwise a random single-qubit gate."""
    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1")
    rng = random.Random(seed)
    circuit = Circuit(num_qubits)
    single_qubit_choices = ["X", "Y", "Z", "H", "RX", "RY", "RZ"]
    two_qubit_choices = ["CX", "CZ"]

    for _ in range(num_gates):
        use_two_qubit = num_qubits >= 2 and rng.random() < two_qubit_fraction
        if use_two_qubit:
            a, b = rng.sample(range(num_qubits), 2)
            name = rng.choice(two_qubit_choices)
            circuit.add_gate(name, (a, b))
        else:
            q = rng.randrange(num_qubits)
            name = rng.choice(single_qubit_choices)
            if name in ("RX", "RY", "RZ"):
                circuit.add_gate(name, (q,), (rng.uniform(0, 2 * 3.141592653589793),))
            else:
                circuit.add_gate(name, (q,))
    return circuit


def _interaction_graph_circuit(
    interaction_graph: nx.Graph,
    repetitions: int,
    seed: int | None,
) -> Circuit:
    """Build a circuit whose two-qubit gates are exactly the edges of
    ``interaction_graph``, repeated ``repetitions`` times in a shuffled
    order each repetition (so the circuit's logical interaction graph is
    exactly ``interaction_graph``)."""
    rng = random.Random(seed)
    n = interaction_graph.number_of_nodes()
    circuit = Circuit(n)
    edges = list(interaction_graph.edges())
    for _ in range(repetitions):
        rng.shuffle(edges)
        for a, b in edges:
            circuit.add_gate("CX", (a, b))
    return circuit


def path_circuit(num_qubits: int, repetitions: int = 1, seed: int | None = None) -> Circuit:
    """Two-qubit gates form a path: (0,1), (1,2), ..., (n-2, n-1)."""
    return _interaction_graph_circuit(nx.path_graph(num_qubits), repetitions, seed)


def cycle_circuit(num_qubits: int, repetitions: int = 1, seed: int | None = None) -> Circuit:
    """Two-qubit gates form a cycle."""
    if num_qubits < 3:
        raise ValueError("A cycle interaction graph requires at least 3 qubits")
    return _interaction_graph_circuit(nx.cycle_graph(num_qubits), repetitions, seed)


def star_circuit(num_qubits: int, repetitions: int = 1, seed: int | None = None) -> Circuit:
    """Two-qubit gates form a star: qubit 0 interacts with every other qubit."""
    return _interaction_graph_circuit(nx.star_graph(num_qubits - 1), repetitions, seed)


def complete_circuit(num_qubits: int, repetitions: int = 1, seed: int | None = None) -> Circuit:
    """Two-qubit gates form a complete graph: every pair interacts."""
    return _interaction_graph_circuit(nx.complete_graph(num_qubits), repetitions, seed)


def random_sparse_circuit(
    num_qubits: int, edge_probability: float, repetitions: int = 1, seed: int | None = None
) -> Circuit:
    """Two-qubit gates form a random Erdos-Renyi interaction graph."""
    g = nx.erdos_renyi_graph(num_qubits, edge_probability, seed=seed)
    return _interaction_graph_circuit(g, repetitions, seed)


def random_regular_circuit(
    num_qubits: int, degree: int, repetitions: int = 1, seed: int | None = None
) -> Circuit:
    """Two-qubit gates form a random ``degree``-regular interaction graph."""
    g = nx.random_regular_graph(degree, num_qubits, seed=seed)
    return _interaction_graph_circuit(g, repetitions, seed)


def logical_interaction_graph(circuit: Circuit) -> nx.Graph:
    """The (unweighted) graph of which logical qubits interact via a
    two-qubit gate somewhere in ``circuit``."""
    g = nx.Graph()
    g.add_nodes_from(range(circuit.num_qubits))
    for gate in circuit.two_qubit_gates:
        g.add_edge(*gate.qubits)
    return g


def graph_features(graph: nx.Graph) -> dict:
    """Structural summary statistics for a graph (logical interaction graph
    or hardware graph), used to study how structure relates to routing cost."""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    degrees = [d for _, d in graph.degree()]
    features = {
        "num_vertices": n,
        "num_edges": m,
        "density": nx.density(graph) if n > 1 else 0.0,
        "average_degree": (sum(degrees) / n) if n > 0 else 0.0,
        "max_degree": max(degrees) if degrees else 0,
        "num_connected_components": nx.number_connected_components(graph),
        "clustering_coefficient": nx.average_clustering(graph) if n > 0 else 0.0,
    }
    if nx.is_connected(graph) and n > 1:
        features["diameter"] = nx.diameter(graph)
    else:
        features["diameter"] = None
    return features

"""Experiment 3: does logical interaction graph structure predict routing
difficulty?

Generates circuits whose logical interaction graphs are paths, cycles,
stars, and random graphs of varying density, routes each with the greedy
router on a fixed grid, and records both the interaction graph's structural
features (density, average degree, ...) and the resulting routing cost.
Produces plots of density/degree vs SWAP count, saved as PNGs, built only
from these actual results (no fabricated numbers).

Usage:
    python experiments/graph_structure_experiment.py
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qroute.generators import (
    cycle_circuit,
    graph_features,
    logical_interaction_graph,
    path_circuit,
    random_sparse_circuit,
    star_circuit,
)
from qroute.hardware import HardwareGraph
from qroute.routers.greedy import GreedyRouter
from qroute.validation import validate_routing_result

OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "results_graph_structure.csv")
OUTPUT_DIR = os.path.dirname(__file__)

NUM_QUBITS = 8
REPETITIONS = 3
SEEDS = list(range(6))
SPARSE_PROBABILITIES = [0.15, 0.25, 0.35, 0.5, 0.7]


def build_circuits() -> list[tuple[str, object]]:
    """Return (label, circuit) pairs covering a spread of interaction
    graph structures, all on NUM_QUBITS logical qubits."""
    circuits: list[tuple[str, object]] = []
    for seed in SEEDS:
        circuits.append((f"path-{seed}", path_circuit(NUM_QUBITS, REPETITIONS, seed=seed)))
        circuits.append((f"cycle-{seed}", cycle_circuit(NUM_QUBITS, REPETITIONS, seed=seed)))
        circuits.append((f"star-{seed}", star_circuit(NUM_QUBITS, REPETITIONS, seed=seed)))
        for p in SPARSE_PROBABILITIES:
            circuits.append(
                (f"random_p{p}-{seed}", random_sparse_circuit(NUM_QUBITS, p, REPETITIONS, seed=seed))
            )
    return circuits


def run() -> list[dict]:
    hardware = HardwareGraph.from_grid(3, 3)
    rows: list[dict] = []
    for label, circuit in build_circuits():
        interaction_graph = logical_interaction_graph(circuit)
        features = graph_features(interaction_graph)
        if features["num_edges"] == 0:
            continue  # nothing to route, not informative
        result = GreedyRouter(hardware).route(circuit)
        validate_routing_result(circuit, result, hardware)
        rows.append(
            {
                "label": label,
                "family": label.split("-")[0],
                **features,
                **result.metrics.as_dict(),
            }
        )
    return rows


def make_plots(rows: list[dict]) -> None:
    densities = [r["density"] for r in rows]
    avg_degrees = [r["average_degree"] for r in rows]
    swaps = [r["inserted_swap_count"] for r in rows]
    num_qubits = [r["num_vertices"] for r in rows]
    depth_overhead = [r["depth_overhead"] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(densities, swaps, color="#023047")
    ax.set_xlabel("logical interaction graph density")
    ax.set_ylabel("inserted SWAP count")
    ax.set_title("Interaction graph density vs SWAP count (greedy router)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "density_vs_swaps.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(avg_degrees, swaps, color="#8338ec")
    ax.set_xlabel("average degree")
    ax.set_ylabel("inserted SWAP count")
    ax.set_title("Average degree vs SWAP count (greedy router)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "degree_vs_swaps.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(num_qubits, depth_overhead, color="#fb8500")
    ax.set_xlabel("circuit size (logical qubits)")
    ax.set_ylabel("depth overhead")
    ax.set_title("Circuit size vs depth overhead (greedy router)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "size_vs_depth_overhead.png"), dpi=150)
    plt.close(fig)


def summarize(rows: list[dict]) -> None:
    from collections import defaultdict

    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)

    print(f"\n{'family':<12}{'n':>4}{'avg density':>14}{'avg degree':>14}{'avg swaps':>12}")
    for family, data in sorted(by_family.items()):
        avg_density = sum(r["density"] for r in data) / len(data)
        avg_degree = sum(r["average_degree"] for r in data) / len(data)
        avg_swaps = sum(r["inserted_swap_count"] for r in data) / len(data)
        print(f"{family:<12}{len(data):>4}{avg_density:>14.3f}{avg_degree:>14.3f}{avg_swaps:>12.2f}")

    densities = [r["density"] for r in rows]
    swaps = [r["inserted_swap_count"] for r in rows]
    n = len(rows)
    mean_d = sum(densities) / n
    mean_s = sum(swaps) / n
    cov = sum((d - mean_d) * (s - mean_s) for d, s in zip(densities, swaps)) / n
    std_d = (sum((d - mean_d) ** 2 for d in densities) / n) ** 0.5
    std_s = (sum((s - mean_s) ** 2 for s in swaps) / n) ** 0.5
    correlation = cov / (std_d * std_s) if std_d > 0 and std_s > 0 else float("nan")
    print(f"\nPearson correlation (density, SWAP count) = {correlation:.3f}")


def main() -> None:
    print("Running graph structure experiment...")
    rows = run()
    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
    make_plots(rows)
    print(f"Saved plots to {OUTPUT_DIR}/*.png")
    summarize(rows)


if __name__ == "__main__":
    main()

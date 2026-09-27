"""Experiment 2: how does routing cost scale with circuit size?

For a fixed hardware topology (a grid, roughly matching each qubit count),
route random circuits of increasing size and record SWAP count, depth
overhead, and runtime.

Usage:
    python experiments/scaling_experiment.py
"""

from __future__ import annotations

import csv
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qroute.generators import random_circuit
from qroute.hardware import HardwareGraph
from qroute.routers.greedy import GreedyRouter
from qroute.routers.lookahead import LookaheadRouter
from qroute.routers.naive import NaiveRouter
from qroute.routers.shortest_path import ShortestPathRouter
from qroute.validation import validate_routing_result

OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "results_scaling.csv")

QUBIT_COUNTS = [4, 6, 8, 10, 12]
GATES_PER_QUBIT = 4  # circuit size grows with qubit count, so difficulty is comparable
TWO_QUBIT_FRACTION = 0.55
SEEDS = list(range(8))

ROUTERS = {
    "naive": NaiveRouter,
    "shortest_path": ShortestPathRouter,
    "greedy": GreedyRouter,
    "lookahead": LookaheadRouter,
}


def grid_for(n: int) -> HardwareGraph:
    """A roughly-square grid with at least n physical qubits."""
    side = math.ceil(math.sqrt(n))
    rows = side
    cols = math.ceil(n / side)
    return HardwareGraph.from_grid(rows, cols)


def run() -> list[dict]:
    rows: list[dict] = []
    for n in QUBIT_COUNTS:
        hardware = grid_for(n)
        num_gates = n * GATES_PER_QUBIT
        for seed in SEEDS:
            circuit = random_circuit(n, num_gates, TWO_QUBIT_FRACTION, seed=seed)
            for router_name, router_cls in ROUTERS.items():
                router = router_cls(hardware)
                result = router.route(circuit)
                validate_routing_result(circuit, result, hardware)
                rows.append(
                    {
                        "num_qubits": n,
                        "hardware_size": hardware.num_qubits,
                        "seed": seed,
                        "router": router_name,
                        **result.metrics.as_dict(),
                    }
                )
        print(f"  n={n}: done ({len(SEEDS)} seeds x {len(ROUTERS)} routers)")
    return rows


def summarize(rows: list[dict]) -> None:
    from collections import defaultdict

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["router"], row["num_qubits"])].append(row)

    print("\nSWAP count and runtime as circuit size grows:")
    header = f"{'router':<15}{'qubits':>8}{'avg swaps':>12}{'avg depth overhead':>22}{'avg runtime (ms)':>20}"
    print(header)
    for router_name in ROUTERS:
        for n in QUBIT_COUNTS:
            data = grouped[(router_name, n)]
            if not data:
                continue
            avg_swaps = sum(r["inserted_swap_count"] for r in data) / len(data)
            avg_depth_overhead = sum(r["depth_overhead"] for r in data) / len(data)
            avg_runtime_ms = sum(r["routing_runtime"] for r in data) / len(data) * 1000
            print(f"{router_name:<15}{n:>8}{avg_swaps:>12.2f}{avg_depth_overhead:>22.2f}{avg_runtime_ms:>20.3f}")


def main() -> None:
    print("Running scaling experiment...")
    rows = run()
    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
    summarize(rows)


if __name__ == "__main__":
    main()

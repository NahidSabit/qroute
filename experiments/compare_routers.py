"""Experiment 1: compare the main routers on the same set of circuits.

Runs several seeded random circuits through every main router (naive,
shortest-path, greedy, lookahead), and A* on the smallest instances only
(it does not scale). Records SWAP count, depth overhead, gate overhead, and
runtime, and saves everything to a CSV so the numbers in the README/report
are never hand-typed.

Usage:
    python experiments/compare_routers.py
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from qroute.generators import random_circuit
from qroute.hardware import HardwareGraph
from qroute.routers.astar import AStarRouter
from qroute.routers.greedy import GreedyRouter
from qroute.routers.lookahead import LookaheadRouter
from qroute.routers.naive import NaiveRouter
from qroute.routers.shortest_path import ShortestPathRouter
from qroute.validation import validate_routing_result

OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "results_compare_routers.csv")

# Circuits used for the main comparison (all four main routers).
NUM_QUBITS = 6
NUM_GATES = 25
TWO_QUBIT_FRACTION = 0.55
SEEDS = list(range(20))

# A smaller set of tiny instances where A* is actually tractable.
ASTAR_NUM_QUBITS = 4
ASTAR_NUM_GATES = 8
ASTAR_SEEDS = list(range(10))

HARDWARE_BUILDERS = {
    "line": lambda n: HardwareGraph.from_line(n),
    "grid": lambda n: HardwareGraph.from_grid(2, (n + 1) // 2) if n > 2 else HardwareGraph.from_line(n),
}


def run() -> list[dict]:
    rows: list[dict] = []
    main_routers = {
        "naive": NaiveRouter,
        "shortest_path": ShortestPathRouter,
        "greedy": GreedyRouter,
        "lookahead": LookaheadRouter,
    }

    for hw_name, build_hw in HARDWARE_BUILDERS.items():
        hardware = build_hw(NUM_QUBITS)
        for seed in SEEDS:
            circuit = random_circuit(NUM_QUBITS, NUM_GATES, TWO_QUBIT_FRACTION, seed=seed)
            for router_name, router_cls in main_routers.items():
                router = router_cls(hardware)
                result = router.route(circuit)
                validate_routing_result(circuit, result, hardware)
                rows.append(
                    {
                        "hardware": hw_name,
                        "seed": seed,
                        "router": router_name,
                        "num_qubits": NUM_QUBITS,
                        "num_gates": NUM_GATES,
                        **result.metrics.as_dict(),
                    }
                )

    # A* comparison on tiny instances only.
    astar_hardware = HardwareGraph.from_line(ASTAR_NUM_QUBITS)
    all_astar_routers = {
        "naive": NaiveRouter,
        "shortest_path": ShortestPathRouter,
        "greedy": GreedyRouter,
        "lookahead": LookaheadRouter,
        "astar": AStarRouter,
    }
    for seed in ASTAR_SEEDS:
        circuit = random_circuit(ASTAR_NUM_QUBITS, ASTAR_NUM_GATES, TWO_QUBIT_FRACTION, seed=seed)
        for router_name, router_cls in all_astar_routers.items():
            router = router_cls(astar_hardware)
            try:
                result = router.route(circuit)
            except RuntimeError as exc:
                print(f"  [astar-set] {router_name} seed={seed} failed to complete: {exc}")
                continue
            validate_routing_result(circuit, result, astar_hardware)
            rows.append(
                {
                    "hardware": "line_tiny",
                    "seed": seed,
                    "router": router_name,
                    "num_qubits": ASTAR_NUM_QUBITS,
                    "num_gates": ASTAR_NUM_GATES,
                    **result.metrics.as_dict(),
                }
            )
    return rows


def summarize(rows: list[dict]) -> None:
    from collections import defaultdict

    by_router: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["hardware"] != "line_tiny":
            by_router[row["router"]].append(row)

    print(f"\nMain comparison ({NUM_QUBITS} qubits, {NUM_GATES} gates, {len(SEEDS)} seeds x 2 topologies):")
    print(f"{'router':<15}{'avg swaps':>12}{'avg depth overhead':>22}{'avg runtime (ms)':>20}")
    for router_name in ["naive", "shortest_path", "greedy", "lookahead"]:
        data = by_router[router_name]
        avg_swaps = sum(r["inserted_swap_count"] for r in data) / len(data)
        avg_depth_overhead = sum(r["depth_overhead"] for r in data) / len(data)
        avg_runtime_ms = sum(r["routing_runtime"] for r in data) / len(data) * 1000
        print(f"{router_name:<15}{avg_swaps:>12.2f}{avg_depth_overhead:>22.2f}{avg_runtime_ms:>20.3f}")

    tiny_rows = [r for r in rows if r["hardware"] == "line_tiny"]
    if tiny_rows:
        by_router_tiny: dict[str, list[dict]] = defaultdict(list)
        for row in tiny_rows:
            by_router_tiny[row["router"]].append(row)
        print(f"\nSmall-instance comparison including A* ({ASTAR_NUM_QUBITS} qubits, {len(ASTAR_SEEDS)} seeds, line topology):")
        print(f"{'router':<15}{'avg swaps':>12}{'instances':>12}")
        for router_name in ["naive", "shortest_path", "greedy", "lookahead", "astar"]:
            data = by_router_tiny.get(router_name, [])
            if not data:
                continue
            avg_swaps = sum(r["inserted_swap_count"] for r in data) / len(data)
            print(f"{router_name:<15}{avg_swaps:>12.2f}{len(data):>12}")


def main() -> None:
    print("Running router comparison experiment...")
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

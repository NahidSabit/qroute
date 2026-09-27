# QRoute

**Graph-based routing and SWAP optimization for quantum circuits on connectivity-constrained hardware.**

QRoute is a personal undergraduate project exploring quantum circuit
routing as a graph problem: implementing several routing heuristics from
scratch, testing them rigorously, and experimentally studying when each
one works well.

## Motivation

Real quantum devices don't let every pair of qubits interact directly.
A logical circuit might require a two-qubit gate between qubits that
aren't adjacent on the hardware:

```
Logical circuit requires:

q0 -------- q3

Hardware provides:

0 --- 1 --- 2 --- 3
```

Since `q0` and `q3` aren't hardware-adjacent, the gate can't execute as
written. The standard fix is to insert `SWAP` gates that physically move
qubit states around until the two qubits that need to interact are
adjacent -- at the cost of extra gates and extra circuit depth. QRoute
treats this as a graph-routing problem: given a hardware connectivity
graph and a circuit's sequence of required interactions, find a
sequence of SWAPs (and an initial placement) that satisfies every
interaction while keeping the SWAP count and depth overhead low.

## What I implemented

- An internal circuit representation (`Gate`/`Circuit`) with round-trip
  conversion to and from Qiskit.
- `HardwareGraph`: connectivity graphs (line, ring, grid, complete,
  arbitrary edge list, random sparse, a small heavy-hex-inspired graph)
  built on NetworkX, with precomputed all-pairs distances.
- `Mapping`: a logical-to-physical qubit placement, plus a simple
  interaction-weighted heuristic for choosing a good initial placement.
- A gate dependency DAG (`DependencyGraph`) used to compute front layers
  and lookahead windows.
- **Four routers**, in increasing sophistication: naive, shortest-path,
  greedy, and lookahead -- the last being the main algorithmic
  contribution of this project (see `docs/algorithms.md`).
- An **optional A\* router**, with an explicitly admissible heuristic,
  used only as a small-instance reference point.
- A validation module that independently re-derives, from a routing
  result, whether it's actually correct: hardware adjacency, mapping
  consistency, and (via Qiskit statevector simulation, for small
  circuits) full unitary equivalence to the original circuit.
- Circuit generators for random circuits and for circuits with specific
  logical interaction graph structures (path, cycle, star, complete,
  random sparse, random regular), plus graph-feature extraction (density,
  average degree, clustering, diameter, ...).
- Three experiments comparing routers, studying scaling behavior, and
  studying how interaction graph structure predicts routing cost.
- 143 pytest tests covering the circuit representation, hardware graphs,
  mapping, dependency graph, every router, and validation -- including
  randomized and quantum-equivalence checks.

This is an educational project, not a production compiler pass: the
point was to implement and understand the algorithms myself, not to
outperform mature tools like Qiskit's transpiler.

## Mathematical formulation

Hardware graph: $G_H = (V_H, E_H)$. Logical interaction graph (which
logical qubits need to interact): $G_L = (V_L, E_L)$. A mapping
$\pi: V_L \to V_H$ places each logical qubit on a physical qubit. An
interaction $(u, v)$ can be executed directly if and only if
$(\pi(u), \pi(v)) \in E_H$. Routing inserts SWAP operations that modify
$\pi$ over time so that every interaction in the circuit can eventually
execute while respecting $E_H$ at the moment it runs. Loosely, the
objective being minimized is

$$\alpha \, N_{\text{swap}} + \beta \, D_{\text{overhead}}$$

-- some combination of SWAP count and depth overhead -- subject to every
two-qubit gate in the routed circuit satisfying the adjacency
constraint above. None of the routers here explicitly optimize this
combined objective; each uses its own heuristic (see
`docs/algorithms.md`) as a proxy for it.

## Algorithms

| Router | Main idea |
|---|---|
| Naive | Move one qubit along a shortest path, gate by gate, in program order |
| Shortest Path | Move both endpoints toward each other, still in program order |
| Greedy | Pick the SWAP minimizing total distance over the current front layer |
| Lookahead | Greedy, plus a weighted term for a window of upcoming gates |
| A* (optional) | Small-instance search reference, with an admissible heuristic |

See `docs/algorithms.md` for the exact algorithms, pseudocode, and the
reasoning behind each design choice (including why A*'s heuristic uses
`max` rather than `sum`, and what that does and doesn't guarantee).

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/qroute.git
cd qroute
pip install -e .
```

For running tests and linting:

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from qroute.circuit import Circuit
from qroute.hardware import HardwareGraph
from qroute.routers.lookahead import LookaheadRouter
from qroute.validation import validate_routing_result

circuit = Circuit(4)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)
circuit.cx(2, 3)
circuit.cx(0, 3)  # not adjacent on a line -- needs routing

hardware = HardwareGraph.from_line(4)
router = LookaheadRouter(hardware, lookahead_size=4, lookahead_weight=0.5)
result = router.route(circuit)

validate_routing_result(circuit, result, hardware)  # raises if anything is wrong
print(f"Inserted {len(result.inserted_swaps)} SWAPs")
print(result.metrics)
```

See `examples/basic_example.py` and `examples/qiskit_example.py` for
complete, runnable walkthroughs (including the Qiskit round-trip and a
statevector equivalence check).

## Experiments

```bash
python experiments/compare_routers.py
python experiments/scaling_experiment.py
python experiments/graph_structure_experiment.py
```

- **`compare_routers.py`** routes the same seeded random circuits with
  every main router (plus A* on tiny instances) and records SWAP count,
  depth overhead, gate overhead, and runtime to
  `experiments/results_compare_routers.csv`. On this project's default
  settings (6 qubits, 25 gates, line and grid hardware, 20 seeds each),
  average SWAP counts were: naive 14.07, shortest-path 13.15, greedy
  11.53, lookahead 9.45 -- lookahead beat naive by about a third on
  average. On tiny 4-qubit instances where A* is tractable, A* averaged
  2.00 SWAPs versus lookahead's 2.10, naive's 2.80, and greedy's 2.30,
  confirming the heuristics are close to (and in this case matching)
  optimal on small cases.
- **`scaling_experiment.py`** routes circuits of growing size (4 to 12
  qubits, scaled proportionally) on a grid and records how SWAP count,
  depth overhead, and runtime grow, to `experiments/results_scaling.csv`.
  Lookahead's advantage over the other routers widened as circuits grew
  larger in this project's runs.
- **`graph_structure_experiment.py`** generates circuits with path,
  cycle, star, and random-sparse interaction graphs at several
  densities, routes each with the greedy router, and studies whether
  interaction graph structure predicts SWAP count, saving
  `experiments/results_graph_structure.csv` and three plots
  (`density_vs_swaps.png`, `degree_vs_swaps.png`,
  `size_vs_depth_overhead.png`). In this project's runs, interaction
  graph density correlated strongly with SWAP count (Pearson
  correlation ≈ 0.92).

All of the numbers above came directly from running these scripts, not
from hand-typed estimates; re-running them will reproduce the same
results given the fixed seeds.

## Example research questions

- How does logical interaction density affect routing overhead?
- How does hardware topology affect routing difficulty?
- When does lookahead improve over greedy routing, and by how much?
- How does routing cost scale with circuit size?
- How sensitive are results to the initial mapping?

## Limitations

- This is an educational/research-oriented personal project, not a
  production compiler pass, and it is not intended to replace or
  outperform mature quantum compilers.
- None of the heuristic routers (naive, shortest-path, greedy,
  lookahead) guarantee a globally minimal SWAP count; A* does guarantee
  optimality given its admissible heuristic, but only if it finishes,
  and it is not scalable beyond small instances.
- The greedy/lookahead candidate generation only considers SWAPs
  adjacent to qubits already involved in a front-layer interaction; it
  does not search further afield.
- The lookahead window (`upcoming_two_qubit_gates`) is a simple
  program-order scan, not a re-derivation of hypothetical future front
  layers -- it is meant to be an explainable heuristic hint, not an
  exact forecast.
- Experiments here use simulated circuits and idealized hardware graphs;
  they don't model gate error rates, calibration data, or other
  real-device effects.

## Future work

- Noise-aware routing (weighting SWAPs by hardware error rates).
- Hardware calibration-aware coupling weights.
- Beam search over candidate SWAP sequences instead of pure greedy
  selection.
- Token-swapping-style formulations for provably better worst-case
  bounds.
- A better initial-placement heuristic (e.g. via graph matching rather
  than the current greedy degree-ranking approach).
- Learned (ML-based) router selection, choosing a router per circuit
  based on its structure.

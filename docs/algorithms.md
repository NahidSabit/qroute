# Routing algorithms

This document explains, precisely, how each router in QRoute works. It
assumes you've read the README's "Motivation" and "Mathematical
formulation" sections: there's a hardware graph $G_H = (V_H, E_H)$, a
logical circuit, a mapping $\pi: V_L \to V_H$ from logical to physical
qubits, and a two-qubit gate $(u, v)$ can execute directly only if
$(\pi(u), \pi(v)) \in E_H$.

Two building blocks are shared by more than one router:

- **Front layer.** Given the set of gates already executed
  ("completed"), the *front layer* is the set of gates that are ready to
  run right now: every gate that shares a qubit with them has already
  run. This comes from a dependency DAG built once per circuit (see
  `qroute/dependency.py`): gate $i$ has an edge to gate $j$ if $i$ is the
  most recent earlier gate touching a qubit that $j$ also touches.
- **Distance.** $d(p, q)$ is the hardware graph's shortest-path distance
  between physical qubits $p$ and $q$, precomputed once per
  `HardwareGraph`.

## 1. Naive router

The simplest possible correct router. It does **not** use the front
layer at all -- it just walks the circuit's gates in their original
order:

```
mapping = initial_mapping
for gate in circuit.gates:
    if gate is single-qubit:
        emit gate on mapping.physical_of(gate.qubit)
    else:  # two-qubit gate on logical qubits (u, v)
        p, q = mapping.physical_of(u), mapping.physical_of(v)
        while not adjacent(p, q):
            path = shortest_path(p, q)
            next_p = path[1]          # one hop towards q
            emit SWAP(p, next_p)
            mapping.apply_swap(p, next_p)
            p = next_p
        emit gate on (p, q)
```

Every blocked gate costs exactly `distance(p, q) - 1` SWAPs, all applied
to one side of the pair. This router exists purely as a correctness
reference and a worst-plausible-case baseline; it is never expected to be
the best router.

## 2. Shortest-path router

Also processes gates in original program order (no front layer), but
instead of moving one endpoint all the way across, it moves **both**
endpoints one hop towards each other per round:

```
while not adjacent(p, q):
    path = shortest_path(p, q)
    if len(path) > 3:            # distance >= 3
        emit SWAP(p, path[1]);      p = path[1]
        emit SWAP(q, path[-2]);     q = path[-2]
    else:                         # distance == 2
        emit SWAP(p, path[1]);      p = path[1]   # deterministic tie-break
```

This still uses exactly `distance(p, q) - 1` SWAPs in total (the same
count as the naive router), but it spreads the disruption to the mapping
across both sides of the path rather than concentrating it on one side.
When only one hop is needed there's nothing to balance, so the router
always moves the first endpoint -- a simple, explicit, deterministic
tie-break.

## 3. Greedy router

This is where the front layer starts to matter. On each iteration:

1. Execute every front-layer gate that's currently free (single-qubit,
   or two-qubit and already adjacent) -- this costs nothing.
2. If any front-layer two-qubit gates remain blocked, generate every SWAP
   adjacent to a physical qubit involved in one of those gates.
3. Score each candidate SWAP $s$ by simulating it and computing

$$H(s) = \sum_{g \in F} d_g(s)$$

   where $F$ is the (post-execution) front layer and $d_g(s)$ is the
   hardware distance between gate $g$'s two qubits after tentatively
   applying $s$.
4. Apply the candidate with the lowest score. Ties are broken
   deterministically by iterating candidates in sorted `(physical_a,
   physical_b)` order and only replacing the incumbent on a strictly
   better score.

**Loop prevention** (see "Loop prevention" below).

## 4. Lookahead router -- the main algorithmic contribution

`LookaheadRouter` subclasses `GreedyRouter` and only overrides the
scoring function (everything else -- candidate generation, tie-breaking,
loop prevention -- is inherited unchanged). Instead of scoring purely on
the front layer, it also considers a window of upcoming two-qubit gates:

$$H(s) = \frac{1}{|F|}\sum_{g \in F} d_g(s) + \lambda \frac{1}{|E|}\sum_{g \in E} d_g(s)$$

where $F$ is the front layer, $E$ is the next `lookahead_size` upcoming
two-qubit gates in program order (excluding ones already in the front
layer -- see `DependencyGraph.upcoming_two_qubit_gates`), and
`lookahead_weight` ($\lambda$) trades off "resolve what's blocking me
right now" against "don't make life harder for gates coming up soon."
Both terms are averaged (not summed) so the score doesn't trivially grow
just because more gates happen to be in the window.

"Upcoming" is deliberately simple: a scan of the circuit's original
order for the next few two-qubit gates, skipping ones already completed
or already in the front layer. A more precise notion (e.g. re-deriving
hypothetical future front layers) would be more accurate but much harder
to explain and reason about, for a heuristic that only needs to be a
"decent hint," not exact.

## Loop prevention

Greedy-style routers can in principle oscillate (swap back and forth)
or get stuck if the front layer's distance can't be reduced by any
single candidate. `GreedyRouter` guards against this with three
layered, understandable mechanisms:

1. **No immediate reversal.** The SWAP just applied cannot be
   immediately undone by the next SWAP (unless every candidate would be
   a reversal, in which case the guard is dropped rather than block
   progress entirely).
2. **Repeated-state detection.** After each SWAP, a hashable key
   `(completed gates, mapping)` is recorded. If the exact same state
   is revisited more than 3 times, or if more than `max_no_progress`
   SWAPs have passed without executing a gate, the router falls back to
   a forced move (next point).
3. **Fallback to shortest-path-style progress.** The fallback picks the
   currently-most-distant blocked front-layer gate and moves one of its
   endpoints one hop closer, exactly like the naive router would. This
   guarantees measurable progress on at least one gate, breaking any
   cycle.

There's also a hard circuit breaker: if the number of inserted SWAPs
ever exceeds a generous multiple of the circuit's gate count, the router
raises an error rather than looping forever. This should never trigger
on a connected hardware graph; it exists to fail loudly if a bug were to
break the above guarantees, instead of hanging.

## 5. A* router (optional, small instances only)

`AStarRouter` searches over **states**, not circuits directly. A state
is `(set of completed gate indices, current mapping)`. From a state,
every gate that's currently free to execute (single-qubit, or two-qubit
and already adjacent) is executed automatically -- this never costs
anything, so it never branches the search. The search only branches
when the front layer contains a blocked two-qubit gate, generating the
same kind of candidate SWAPs as the greedy router.

$$f(n) = g(n) + h(n)$$

- $g(n)$: number of SWAPs used to reach state $n$.
- $h(n) = \max_{g \in F}\big(d_g(n) - 1,\ 0\big)$: the **largest**, not
  the sum, of `distance - 1` over the front layer's two-qubit gates.

**Why `max` and not `sum`, and why that matters for admissibility.**
Whichever front-layer gate is currently farthest apart needs at least
`distance - 1` SWAPs just for that gate, so the true remaining cost can
never be less than that single value -- `max` is therefore a valid lower
bound (admissible). A sum over all front-layer gates would be a more
informative heuristic, but it is **not** guaranteed admissible: a single
SWAP can simultaneously reduce the distance of two front-layer gates
that share a qubit, so summing independent per-gate lower bounds can
overestimate the true minimum SWAP count. Because `h` here is
admissible, A* is guaranteed to return an optimal (minimum-SWAP)
solution *if it finishes* -- but see the caveat below.

**Caveat: this router does not claim to scale.** The number of reachable
states grows very quickly with circuit size and hardware size. A hard
cap, `max_explored_states`, stops the search and raises a `RuntimeError`
rather than hanging. A* in this project exists to sanity-check the
other routers' SWAP counts on tiny instances (a handful of qubits and
gates), not to be used as a general-purpose router.

## A note on program order vs. dependency order

The naive and shortest-path routers process gates in the circuit's
original order. The greedy, lookahead, and A* routers only respect the
*dependency* order (gates sharing a qubit must stay in relative order);
independent gates may be executed in a different order than they
appeared in the original circuit. `qroute.validation.validate_routing_result`
is written to allow this: it checks per-logical-qubit ordering rather
than requiring an exact match to the original gate sequence.

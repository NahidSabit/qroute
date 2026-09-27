"""Correctness checks for a routing result.

A router is only as trustworthy as its validator. This module re-derives,
from a :class:`~qroute.routers.base.RoutingResult`, whether the routed
circuit is actually a legal, meaning-preserving implementation of the
original logical circuit on the given hardware.
"""

from __future__ import annotations

from qroute.circuit import Circuit
from qroute.hardware import HardwareGraph
from qroute.mapping import Mapping


class ValidationError(Exception):
    """Raised when a routed circuit fails a correctness check."""


def validate_mapping(mapping: Mapping) -> None:
    """Check that a mapping is a valid one-to-one placement."""
    if not mapping.is_valid():
        raise ValidationError(f"Mapping is not a valid one-to-one placement: {mapping}")


def validate_hardware_indices(circuit: Circuit, hardware: HardwareGraph) -> None:
    """Check every qubit index used in ``circuit`` is a valid physical qubit."""
    for gate in circuit.gates:
        for q in gate.qubits:
            if not (0 <= q < hardware.num_qubits):
                raise ValidationError(
                    f"Gate {gate} references physical qubit {q}, outside "
                    f"range [0, {hardware.num_qubits})"
                )


def validate_adjacency(circuit: Circuit, hardware: HardwareGraph) -> None:
    """Check every two-qubit gate in ``circuit`` acts on hardware-adjacent qubits."""
    for gate in circuit.gates:
        if gate.is_two_qubit and not hardware.are_adjacent(*gate.qubits):
            raise ValidationError(
                f"Gate {gate} acts on physical qubits {gate.qubits}, which are "
                "not adjacent on the hardware graph"
            )


def validate_routing_result(original_circuit: Circuit, result, hardware: HardwareGraph) -> None:
    """Fully validate a :class:`~qroute.routers.base.RoutingResult`.

    This replays the routed circuit gate by gate, using the router's own
    ``inserted_swaps`` list (in order) to distinguish SWAP gates inserted
    for routing from any ``SWAP`` gates that were already part of the
    original logical circuit. It checks that:

    1. every physical qubit index used is valid;
    2. every two-qubit gate (routing SWAPs included) acts on adjacent
       hardware qubits;
    3. replaying the inserted SWAPs on the initial mapping reproduces the
       router's reported final mapping;
    4. the non-SWAP-insertion gates of the routed circuit, translated back
       to logical qubits through the mapping active at the time they
       executed, are a valid linearization of the original circuit: gates
       that share a logical qubit occur in the same relative order as in
       the original circuit (dependencies preserved), and every original
       gate is executed exactly once, unmodified. Independent gates (no
       shared qubit) may legally appear in a different order than in the
       original circuit -- routers are free to reorder those.
    """
    routed = result.routed_circuit
    validate_hardware_indices(routed, hardware)
    validate_adjacency(routed, hardware)
    validate_mapping(result.initial_mapping)
    validate_mapping(result.final_mapping)

    # Per logical qubit, the original gate indices that touch it, in the
    # original circuit's order. Matching against these pointers (rather than
    # a single global pointer) is what allows independent gates to be
    # reordered while still catching any violation of per-qubit ordering.
    per_qubit_original_order: dict[int, list[int]] = {}
    for i, gate in enumerate(original_circuit.gates):
        for q in gate.qubits:
            per_qubit_original_order.setdefault(q, []).append(i)
    pointer = {q: 0 for q in per_qubit_original_order}

    mapping = result.initial_mapping.copy()
    swap_ptr = 0
    inserted_swaps = result.inserted_swaps
    matched_count = 0

    for gate in routed.gates:
        is_routing_swap = (
            gate.name == "SWAP"
            and swap_ptr < len(inserted_swaps)
            and gate.qubits == tuple(inserted_swaps[swap_ptr])
        )
        if is_routing_swap:
            mapping.apply_swap(*gate.qubits)
            swap_ptr += 1
            continue

        logical_qubits = tuple(mapping.logical_of(p) for p in gate.qubits)
        if any(l is None for l in logical_qubits):
            raise ValidationError(
                f"Gate {gate} acts on a physical qubit with no logical qubit currently mapped "
                "to it"
            )

        candidate_indices = set()
        for q in logical_qubits:
            qlist = per_qubit_original_order.get(q, [])
            p = pointer.get(q, 0)
            if p >= len(qlist):
                raise ValidationError(
                    f"Found an extra gate touching logical qubit {q} that has no corresponding "
                    "unmatched gate in the original circuit"
                )
            candidate_indices.add(qlist[p])
        if len(candidate_indices) != 1:
            raise ValidationError(
                f"Gate {gate} (logical qubits {logical_qubits}) does not match a single, "
                "consistent next expected original gate for all of its qubits -- this means "
                "per-qubit gate order was not preserved"
            )
        orig_idx = candidate_indices.pop()
        expected = original_circuit.gates[orig_idx]
        if gate.name != expected.name or gate.params != expected.params or logical_qubits != expected.qubits:
            raise ValidationError(
                f"Gate {gate} (decoded to logical qubits {logical_qubits}) does not match "
                f"expected original gate {expected} at index {orig_idx}"
            )
        for q in logical_qubits:
            pointer[q] += 1
        matched_count += 1

    if swap_ptr != len(inserted_swaps):
        raise ValidationError(
            f"Only {swap_ptr}/{len(inserted_swaps)} reported inserted SWAPs were found, in "
            "order, in the routed circuit"
        )
    if matched_count != len(original_circuit.gates):
        raise ValidationError(
            f"Only {matched_count}/{len(original_circuit.gates)} original gates were found in "
            "the routed circuit"
        )
    if mapping.logical_to_physical != result.final_mapping.logical_to_physical:
        raise ValidationError(
            "Replaying inserted SWAPs on the initial mapping does not reproduce the "
            f"reported final mapping: got {mapping}, expected {result.final_mapping}"
        )


def verify_equivalence_statevector(original_circuit: Circuit, result, hardware: HardwareGraph, atol: float = 1e-6) -> bool:
    """Optional, expensive check: simulate statevectors to confirm the routed
    circuit implements the same unitary as the original circuit, up to the
    qubit permutation recorded by ``result.final_mapping`` and up to global
    phase. Intended for small circuits only (exponential in qubit count).

    Returns True if the states match; raises ``ValidationError`` otherwise.
    """
    from qiskit.circuit.library import PermutationGate
    from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity

    n = original_circuit.num_qubits
    N = hardware.num_qubits

    orig_sv = Statevector.from_instruction(original_circuit.to_qiskit())
    routed_sv = Statevector.from_instruction(result.routed_circuit.to_qiskit())

    final_map = result.final_mapping.logical_to_physical  # logical -> physical
    logical_order_physical = [final_map[l] for l in range(n)]
    ancilla_physical = [p for p in range(N) if p not in logical_order_physical]

    # PermutationGate pattern: pattern[k] = m means qubit m moves to position k.
    # We want physical qubit `logical_order_physical[l]` to land at position l,
    # and each ancilla physical qubit to land at some position >= n.
    pattern = logical_order_physical + ancilla_physical
    from qiskit import QuantumCircuit

    perm_circuit = QuantumCircuit(N)
    perm_circuit.append(PermutationGate(pattern), range(N))
    reordered_sv = routed_sv.evolve(perm_circuit)

    if N > n:
        reduced = partial_trace(reordered_sv, list(range(n, N)))
        ancilla_dm = partial_trace(reordered_sv, list(range(n)))
        expected_ancilla = DensityMatrix.from_label("0" * (N - n))
        if state_fidelity(ancilla_dm, expected_ancilla) < 1 - 1e-6:
            raise ValidationError("Ancilla (unused hardware) qubits are not left in |0>")
        logical_state = reduced
    else:
        logical_state = DensityMatrix(reordered_sv)

    fidelity = state_fidelity(logical_state, DensityMatrix(orig_sv))
    if fidelity < 1 - atol:
        raise ValidationError(
            f"Routed circuit is not equivalent to the original circuit "
            f"(state fidelity {fidelity:.6f} < {1 - atol:.6f})"
        )
    return True

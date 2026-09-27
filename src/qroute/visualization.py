"""Simple NetworkX/Matplotlib visualizations.

These are deliberately plain: the goal is a clear, static picture, not a UI.
Every function returns the ``matplotlib`` ``Figure`` it drew on, so callers
can show it, save it, or embed it in a larger figure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx


def plot_hardware_graph(hardware, ax=None, title: str | None = None):
    """Draw a hardware connectivity graph."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    pos = nx.spring_layout(hardware.graph, seed=0)
    nx.draw_networkx_nodes(hardware.graph, pos, ax=ax, node_color="#8ecae6", node_size=500)
    nx.draw_networkx_edges(hardware.graph, pos, ax=ax, edge_color="#555555")
    nx.draw_networkx_labels(hardware.graph, pos, ax=ax, font_size=10)
    ax.set_title(title or "Hardware connectivity graph")
    ax.axis("off")
    return fig


def plot_interaction_graph(circuit, ax=None, title: str | None = None):
    """Draw a circuit's logical interaction graph."""
    from qroute.generators import logical_interaction_graph

    graph = logical_interaction_graph(circuit)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    pos = nx.spring_layout(graph, seed=0)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color="#ffb703", node_size=500)
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#555555")
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=10)
    ax.set_title(title or "Logical interaction graph")
    ax.axis("off")
    return fig


def plot_mapping(hardware, mapping, ax=None, title: str | None = None):
    """Draw the hardware graph with each physical qubit labeled by the
    logical qubit currently mapped to it (or blank if unmapped)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    pos = nx.spring_layout(hardware.graph, seed=0)
    labels = {}
    colors = []
    for p in hardware.graph.nodes:
        logical = mapping.logical_of(p)
        labels[p] = f"P{p}\n(L{logical})" if logical is not None else f"P{p}\n(-)"
        colors.append("#8ecae6" if logical is not None else "#dddddd")
    nx.draw_networkx_nodes(hardware.graph, pos, ax=ax, node_color=colors, node_size=800)
    nx.draw_networkx_edges(hardware.graph, pos, ax=ax, edge_color="#555555")
    nx.draw_networkx_labels(hardware.graph, pos, ax=ax, labels=labels, font_size=8)
    ax.set_title(title or "Mapping")
    ax.axis("off")
    return fig


def plot_initial_and_final_mapping(hardware, result, title: str | None = None):
    """Side-by-side plot of the initial and final mapping for a routing result."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_mapping(hardware, result.initial_mapping, ax=axes[0], title="Initial mapping")
    plot_mapping(hardware, result.final_mapping, ax=axes[1], title="Final mapping")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_metric_vs_feature(x_values, y_values, x_label: str, y_label: str, title: str | None = None, ax=None):
    """A simple scatter plot for experiment analysis (e.g. graph density vs
    SWAP count)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    ax.scatter(x_values, y_values, color="#023047")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title or f"{y_label} vs {x_label}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_router_comparison_bars(router_names, values, y_label: str, title: str | None = None, ax=None):
    """A bar chart comparing one metric across routers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    ax.bar(router_names, values, color="#219ebc")
    ax.set_ylabel(y_label)
    ax.set_title(title or y_label)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig

"""Routing strategies for QRoute.

Four main routers -- :class:`NaiveRouter`, :class:`ShortestPathRouter`,
:class:`GreedyRouter`, :class:`LookaheadRouter` -- plus an optional
:class:`AStarRouter` reference point for tiny instances.
"""

from qroute.routers.astar import AStarRouter
from qroute.routers.base import BaseRouter, RoutingResult
from qroute.routers.greedy import GreedyRouter
from qroute.routers.lookahead import LookaheadRouter
from qroute.routers.naive import NaiveRouter
from qroute.routers.shortest_path import ShortestPathRouter

__all__ = [
    "AStarRouter",
    "BaseRouter",
    "GreedyRouter",
    "LookaheadRouter",
    "NaiveRouter",
    "RoutingResult",
    "ShortestPathRouter",
]

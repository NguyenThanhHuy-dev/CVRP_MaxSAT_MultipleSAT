"""
Simple encoder to produce a Weighted CNF via PySAT's WCNF object.
This example implements a compact mapping of route-selection variables only:
- We treat each candidate route from heuristic as an optional bundle variable x_r.
- Hard constraints ensure each customer is covered at least once (ALO on selected routes covering that customer).
- Each route variable has weight equal to the route cost (so selecting routes accumulates cost).

This is a simplified encoding useful as a first step before implementing full wijv/tiv/uibv scheme from the paper.
"""
from pysat.formula import WCNF
from typing import List, Tuple
import numpy as np


def encode_routes_as_wcnf(routes: List[List[int]], D: np.ndarray) -> Tuple[WCNF, dict]:
    """
    routes: list of routes (each route is list of node indices, starting and ending with 0)
    D: distance matrix
    Returns: WCNF object and mapping route_index -> var
    """
    wcnf = WCNF()
    route_vars = {}
    for r_idx, route in enumerate(routes, start=1):
        cost = sum(D[route[i], route[i+1]] for i in range(len(route)-1))
        # assign positive integer var id
        route_vars[r_idx] = r_idx
        # soft clause: selecting route costs 'cost'
        # we store integer costs as int (round)
        wcnf.append([r_idx], weight=int(round(cost)))

    # At-least-one coverage constraint per customer (customers 1..n)
    n = max(max(r) for r in routes)
    for customer in range(1, n+1):
        covering = [ridx for ridx, route in enumerate(routes, start=1) if customer in route]
        if not covering:
            # this customer is not covered by any route; model invalid
            raise ValueError(f'Customer {customer} not covered by any candidate route')
        # hard clause: at least one selected route covers this customer
        wcnf.append(covering)
    return wcnf, route_vars


def write_wcnf_to_file(wcnf: WCNF, path: str):
    wcnf.to_file(path)
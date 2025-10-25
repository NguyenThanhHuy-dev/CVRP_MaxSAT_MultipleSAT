"""
Clarke-Wright (parallel) + 2-opt implementation.
Functions:
 - clarke_wright_savings(D, demands, capacity) -> list of routes
 - two_opt(route, D) -> improved route
 - route_cost(route, D) -> cost
 - total_distance(routes, D) -> total cost
"""
from typing import List
import numpy as np
from itertools import combinations


def clarke_wright_savings(D: np.ndarray, demands: List[int], capacity: int) -> List[List[int]]:
    n = D.shape[0] - 1  # including depot at 0, customers 1..n
    # Start with each customer in its own route: [0, i, 0]
    routes = [[0, i, 0] for i in range(1, n + 1)]

    # Helper - find route index containing node x
    def find_route(routes, x):
        for idx, r in enumerate(routes):
            if x in r[1:-1]:
                return idx
        return None

    # Compute savings
    savings = []
    for i, j in combinations(range(1, n + 1), 2):
        s = D[i, 0] + D[0, j] - D[i, j]
        savings.append((s, i, j))
    savings.sort(reverse=True, key=lambda x: x[0])

    # Greedily merge
    for s, i, j in savings:
        ri = find_route(routes, i)
        rj = find_route(routes, j)
        if ri is None or rj is None or ri == rj:
            continue
        route_i = routes[ri]
        route_j = routes[rj]
        # Check if i is at one end and j at one end to merge without reorientation issues
        if route_i[-2] == i and route_j[1] == j:
            load_i = sum(demands[k] for k in route_i if k != 0)
            load_j = sum(demands[k] for k in route_j if k != 0)
            if load_i + load_j <= capacity:
                new_route = route_i[:-1] + route_j[1:]
                routes.pop(max(ri, rj))
                routes.pop(min(ri, rj))
                routes.append(new_route)
    return routes


def route_cost(route: List[int], D: np.ndarray) -> float:
    return sum(D[route[i], route[i + 1]] for i in range(len(route) - 1))


def total_distance(routes: List[List[int]], D: np.ndarray) -> float:
    return sum(route_cost(r, D) for r in routes)


def two_opt(route: List[int], D: np.ndarray, max_iter: int = 100) -> List[int]:
    best = route[:]
    best_cost = route_cost(best, D)
    n = len(route)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                new_cost = route_cost(new_route, D)
                if new_cost + 1e-9 < best_cost:
                    best = new_route
                    best_cost = new_cost
                    improved = True
        it += 1
    return best


if __name__ == '__main__':
    # Quick manual test
    D = np.array([
        [0, 10, 12, 11, 8],
        [10, 0, 5, 13, 10],
        [12, 5, 0, 7, 9],
        [11, 13, 7, 0, 6],
        [8, 10, 9, 6, 0],
    ], dtype=float)
    demands = [0, 5, 8, 4, 6]
    C = 15
    routes = clarke_wright_savings(D, demands, C)
    print('CW routes:', routes)
    print('Cost:', total_distance(routes, D))
    improved = [two_opt(r, D) for r in routes]
    print('Improved:', improved)
    print('Final cost:', total_distance(improved, D))
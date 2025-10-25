"""
Simple data loader for CVRPLIB-like instances and simple CSV/JSON instances.
Provides an `Instance` dataclass with distance matrix, demands, depot, etc.
"""
from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np
import os

@dataclass
class Instance:
    name: str
    n: int
    depot: int
    coords: List[Tuple[float, float]]
    demands: List[int]
    capacity: int
    dist_matrix: np.ndarray


def euclidean_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def compute_distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            D[i, j] = euclidean_distance(coords[i], coords[j])
    return D


def from_coords_and_demands(name: str, coords: List[Tuple[float, float]], demands: List[int], capacity: int, depot: int = 0) -> Instance:
    if len(coords) != len(demands):
        raise ValueError("coords and demands must have same length")
    D = compute_distance_matrix(coords)
    return Instance(name=name, n=len(coords), depot=depot, coords=coords, demands=demands, capacity=capacity, dist_matrix=D)


def read_vrplib(path: str) -> Instance:
    """
    Minimal parser for some CVRPLIB .vrp formats (Euclidean). Not full-featured.
    Expects NODE_COORD_SECTION and DEMAND_SECTION. Use for small experiments.
    """
    name = os.path.basename(path)
    coords = []
    demands = []
    capacity = None
    depot = 0
    reading_coords = False
    reading_demands = False
    coord_map = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('CAPACITY') or line.startswith('VEHICLE_CAPACITY'):
                parts = line.replace(':', ' ').split()
                for p in parts:
                    if p.isdigit():
                        capacity = int(p)
                        break
            if line.startswith('NODE_COORD_SECTION'):
                reading_coords = True
                reading_demands = False
                continue
            if line.startswith('DEMAND_SECTION'):
                reading_demands = True
                reading_coords = False
                continue
            if line.startswith('DEPOT_SECTION'):
                break
            if reading_coords:
                parts = line.split()
                idx = int(parts[0])
                x = float(parts[1]); y = float(parts[2])
                coord_map[idx] = (x, y)
            if reading_demands:
                parts = line.split()
                idx = int(parts[0])
                d = int(parts[1])
                demands.append((idx, d))

    if not coord_map:
        raise ValueError('No NODE_COORD_SECTION found or unsupported format')
    # Convert by index order
    max_idx = max(coord_map.keys())
    coords = [coord_map[i] for i in range(1, max_idx + 1)]
    # demands list may be unordered
    demand_map = {i: 0 for i in range(1, max_idx + 1)}
    for idx, d in demands:
        demand_map[idx] = d
    demand_list = [demand_map[i] for i in range(1, max_idx + 1)]
    # Prepend depot at index 0
    # Note: many .vrp use 1-based indexing with depot at 1; we map to 0-based
    instance_coords = coords
    instance_demands = [0] + demand_list[1:] if len(demand_list) >= 1 else [0]

    # Fallback capacity
    if capacity is None:
        capacity = max(sum(instance_demands), 999999)

    # Build final instance (if depot is node 1 in file)
    coords0 = [(0.0, 0.0)] + instance_coords[1:]
    demands0 = [0] + instance_demands[1:]
    D = compute_distance_matrix(coords0)
    return Instance(name=name, n=len(coords0), depot=0, coords=coords0, demands=demands0, capacity=capacity, dist_matrix=D)
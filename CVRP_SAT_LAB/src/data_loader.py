from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import numpy as np
import os
import re


@dataclass
class Instance:
    name: str
    n: int
    depot: int
    coords: List[Tuple[float, float]]
    demands: List[int]
    capacity: int
    dist_matrix: np.ndarray
    bks: float = 0.0  # Best Known Solution


def euclidean_distance(a, b):
    # CVRPLIB Convention: Euclidean distance rounded to nearest integer
    return int(round(math.hypot(a[0] - b[0], a[1] - b[1])))


def compute_distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n), dtype=int)    # <--- Sửa thành int
    for i in range(n):
        for j in range(n):
            D[i, j] = euclidean_distance(coords[i], coords[j])
    return D


def from_coords_and_demands(
    name: str,
    coords: List[Tuple[float, float]],
    demands: List[int],
    capacity: int,
    depot: int = 0,
) -> Instance:
    if len(coords) != len(demands):
        raise ValueError("coords and demands must have same length")
    D = compute_distance_matrix(coords)
    return Instance(
        name=name,
        n=len(coords),
        depot=depot,
        coords=coords,
        demands=demands,
        capacity=capacity,
        dist_matrix=D,
    )


def read_vrplib(path: str) -> Instance:
    name = os.path.basename(path).split(".")[0]
    coords = []
    demands = []
    edge_weights = []
    capacity = None
    bks = 0.0
    dimension = 0
    edge_weight_format = None

    # States
    reading_coords = False
    reading_demands = False
    reading_edge_weights = False
    
    coord_map = {}
    demand_map = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # --- HEADER PARSING ---
            if "Optimal value" in line:
                match = re.search(r"Optimal value:?\s*(\d+)", line)
                if match:
                    bks = float(match.group(1))

            if line.startswith("DIMENSION"):
                parts = line.replace(":", " ").split()
                for p in parts:
                    if p.isdigit():
                        dimension = int(p)
                        break

            if line.startswith("CAPACITY") or line.startswith("VEHICLE_CAPACITY"):
                parts = line.replace(":", " ").split()
                for p in parts:
                    if p.isdigit():
                        capacity = int(p)
                        break
            
            if line.startswith("EDGE_WEIGHT_FORMAT"):
                parts = line.replace(":", " ").split()
                if len(parts) >= 2:
                    edge_weight_format = parts[1].strip()

            # --- SECTION DETECTING ---
            if line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                reading_demands = False
                reading_edge_weights = False
                continue
            
            if line.startswith("DEMAND_SECTION"):
                reading_demands = True
                reading_coords = False
                reading_edge_weights = False
                continue
            
            if line.startswith("EDGE_WEIGHT_SECTION"):
                reading_edge_weights = True
                reading_coords = False
                reading_demands = False
                continue

            if line.startswith("DEPOT_SECTION") or line.startswith("EOF"):
                reading_edge_weights = False
                reading_coords = False
                reading_demands = False
                break

            # --- DATA PARSING ---
            if reading_coords:
                parts = line.split()
                try:
                    vals = [float(x) for x in parts]
                    if len(vals) >= 3:
                        idx = int(vals[0])
                        x, y = vals[1], vals[2]
                        coord_map[idx] = (x, y)
                except: pass

            elif reading_demands:
                parts = line.split()
                try:
                    # Xử lý trường hợp dòng chỉ có demand hoặc có cả index
                    vals = [int(x) for x in parts]
                    if len(vals) >= 2:
                        idx = int(vals[0])
                        d = int(vals[1])
                        demand_map[idx] = d
                    elif len(vals) == 1:
                        # Nếu file chỉ liệt kê demand mà không có index (ít gặp)
                        pass 
                except: pass

            elif reading_edge_weights:
                # Đọc toàn bộ số trong section này vào một list phẳng
                parts = line.split()
                for p in parts:
                    try:
                        # Thử ép về int nếu có thể, hoặc làm tròn
                        edge_weights.append(int(float(p) + 0.5)) 
                    except: pass

    # --- POST-PROCESSING ---
    
    # 1. Xử lý Demands
    # Nếu demand_map rỗng (chưa đọc được), thử cách khác hoặc báo lỗi
    if not demand_map and demands:
        # Fallback logic cũ
        pass
    
    # Chuẩn hóa demand về list (index 0 là depot)
    # CVRPLIB thường index từ 1 đến dimension
    if not demand_map and dimension > 0:
         # Tạo demand giả định bằng 0 nếu không tìm thấy (để tránh crash)
         final_demands = [0] * dimension
    else:
        final_demands = [0] * dimension
        for i in range(1, dimension + 1):
            final_demands[i-1] = demand_map.get(i, 0)
    
    # Do logic code của bạn giả định depot ở đầu danh sách và index=0
    # Ta sẽ giữ nguyên thứ tự này.
    
    # 2. Xử lý Distance Matrix & Coords
    D = None
    final_coords = []

    if coord_map:
        # Trường hợp 1: Có tọa độ (EUC_2D)
        final_coords = [coord_map[i] for i in range(1, dimension + 1)]
        # Code cũ của bạn thêm (0,0) vào đầu để làm depot 0? 
        # CVRPLIB: Node 1 thường là Depot.
        # Logic trong encoder/heuristic của bạn dùng index 0..n-1.
        # Nên ta sẽ dùng trực tiếp list này. Node 0 trong list = Node 1 trong file.
        D = compute_distance_matrix(final_coords)
    
    elif edge_weights:
        # Trường hợp 2: Có ma trận trọng số (EXPLICIT)
        # Tạo tọa độ giả để không bị lỗi code (Plot sẽ bị sai nhưng solver chạy đúng)
        final_coords = [(0.0, 0.0) for _ in range(dimension)]
        
        D = np.zeros((dimension, dimension), dtype=int)
        
        if edge_weight_format == "LOWER_ROW":
            # Parse Lower Triangular Matrix
            # K: counter cho edge_weights
            k = 0
            for i in range(dimension):
                for j in range(i):
                    val = edge_weights[k]
                    D[i, j] = val
                    D[j, i] = val
                    k += 1
        elif edge_weight_format == "FULL_MATRIX":
            k = 0
            for i in range(dimension):
                for j in range(dimension):
                    D[i, j] = edge_weights[k]
                    k += 1
        else:
            # Mặc định thử parse theo FULL hoặc LOWER nếu format ko rõ
            # Nhưng E-n31-k7 là LOWER_ROW
            pass

    if D is None:
        raise ValueError("Could not construct Distance Matrix (No coords and no edge weights found)")

    # Fallback capacity
    if capacity is None:
        capacity = 999999

    return Instance(
        name=name,
        n=dimension,
        depot=0, # Index 0 trong list coords/demands
        coords=final_coords,
        demands=final_demands,
        capacity=capacity,
        dist_matrix=D,
        bks=bks,
    )
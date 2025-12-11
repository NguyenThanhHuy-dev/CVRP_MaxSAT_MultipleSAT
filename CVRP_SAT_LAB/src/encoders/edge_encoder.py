from pysat.formula import WCNF
from pysat.card import CardEnc
from typing import List, Tuple, Dict
import numpy as np

def encode_edges_as_wcnf(nodes: List[int], 
                         edges: List[Tuple[int, int]], 
                         dist_matrix: np.ndarray,
                         subtour_cuts: List[List[Tuple[int, int]]] = None,
                         min_vehicles: int = 1) -> Tuple[WCNF, Dict[Tuple[int, int], int]]: # <--- MỚI: Default là 1
    
    wcnf = WCNF()
    edge_to_var = {}
    var_to_edge = {} # (Nếu bạn cần dùng biến này để debug ngược lại)
    counter = 1
    
    # 1. Map Variables & Soft Clauses (Minimize Cost)
    for u, v in edges:
        edge_to_var[(u, v)] = counter
        var_to_edge[counter] = (u, v)
        
        cost = int(round(dist_matrix[u, v]))
        wcnf.append([-counter], weight=cost)
        counter += 1
        
    top_var = counter - 1

    # Gom cạnh
    in_edges = {i: [] for i in nodes}
    out_edges = {i: [] for i in nodes}
    for u, v in edges:
        if u in out_edges: out_edges[u].append(edge_to_var[(u, v)])
        if v in in_edges: in_edges[v].append(edge_to_var[(u, v)])

    # 2. Hard Clauses: Degree Constraints
    for i in nodes:
        # Nếu node không có cạnh nối -> Vô nghiệm
        if not in_edges[i] or not out_edges[i]:
            wcnf.append([]) # Force UNSAT
            continue

        if i == 0:
            # --- FIX QUAN TRỌNG: Ràng buộc cho Depot ---
            # Depot phải có ít nhất K xe đi ra và K xe đi vào.
            # Thay vì bound=1 (ít nhất 1 xe), ta dùng bound=min_vehicles
            
            # Out-degree >= min_vehicles
            cnf_out = CardEnc.atleast(lits=out_edges[i], bound=min_vehicles, top_id=top_var)
            wcnf.extend(cnf_out.clauses)
            top_var = cnf_out.nv
            
            # In-degree >= min_vehicles
            cnf_in = CardEnc.atleast(lits=in_edges[i], bound=min_vehicles, top_id=top_var)
            wcnf.extend(cnf_in.clauses)
            top_var = cnf_in.nv
        else:
            # Khách hàng: Exactly 1 Out, Exactly 1 In
            cnf_out = CardEnc.equals(lits=out_edges[i], bound=1, top_id=top_var)
            wcnf.extend(cnf_out.clauses)
            top_var = cnf_out.nv
            
            cnf_in = CardEnc.equals(lits=in_edges[i], bound=1, top_id=top_var)
            wcnf.extend(cnf_in.clauses)
            top_var = cnf_in.nv

    # 3. Subtour Elimination Cuts
    if subtour_cuts:
        for cut_edges in subtour_cuts:
            clause = []
            for u, v in cut_edges:
                if (u, v) in edge_to_var:
                    clause.append(-edge_to_var[(u, v)])
            if clause:
                wcnf.append(clause)

    return wcnf, edge_to_var
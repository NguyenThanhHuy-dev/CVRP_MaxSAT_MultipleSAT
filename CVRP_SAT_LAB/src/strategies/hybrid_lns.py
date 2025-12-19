import time
import os
import random
import numpy as np
from typing import List, Tuple

from data_loader import Instance
from heuristic import clarke_wright_savings, two_opt, total_distance
from encoders.edge_encoder import encode_edges_as_wcnf
from solver_service import call_openwbo
from utils.decoder import parse_openwbo_model
from utils.logger import log_benchmark
from utils.plot import plot_solution
from config import TIMEOUT_LNS_INNER, MAX_ITERATIONS

class HybridLNSStrategy:
    def __init__(self, instance: Instance):
        self.instance = instance
        self.k_nearest = 10  # Tăng K lên một chút để không gian tìm kiếm rộng hơn
        self.max_inner_iter = 3
        self.stagnation_limit = 20 # Dừng nếu 15 lần liên tiếp không cải thiện

    def _build_restricted_graph(self, current_routes: List[List[int]]) -> List[Tuple[int, int]]:
        edges = set()
        N = self.instance.n
        D = self.instance.dist_matrix
        
        keep_probability = 0.6 
        for r in current_routes:
            for i in range(len(r) - 1):
                if random.random() < keep_probability:
                    edges.add((r[i], r[i+1]))
        
        for i in range(N):
            nearest_indices = np.argsort(D[i])[1:self.k_nearest+1]
            for neighbor in nearest_indices:
                if i != neighbor:
                    edges.add((i, neighbor))
                    edges.add((neighbor, i)) 
        
        depot_nearest = np.argsort(D[0])[1:self.k_nearest+1]
        for i in depot_nearest:
            edges.add((0, i))
            edges.add((i, 0))
            
        for r in current_routes:
            if len(r) > 2:
                first, last = r[1], r[-2] # r[0] và r[-1] là 0
                edges.add((0, first))
                edges.add((first, 0))
                edges.add((0, last))
                edges.add((last, 0))

        return list(edges)

    def _extract_routes_from_edges(self, active_edges: List[Tuple[int, int]]) -> Tuple[List[List[int]], List[List[int]]]:
        adj = {}
        all_active_nodes = set()
        depot_outgoing = []

        for u, v in active_edges:
            adj[u] = v
            all_active_nodes.add(u)
            all_active_nodes.add(v)
            if u == 0:
                depot_outgoing.append(v)

        valid_routes = []
        visited_edges = set()
        
        for start_node in depot_outgoing:
            if (0, start_node) in visited_edges: continue
            
            path = [0]
            curr = start_node
            visited_edges.add((0, start_node))
            
            is_closed = False
            steps = 0
            while steps < self.instance.n * 2: # Phòng ngừa lặp vô hạn
                steps += 1
                path.append(curr)
                if curr == 0: 
                    is_closed = True
                    break
                
                if curr not in adj: break 
                next_node = adj[curr]
                
                edge = (curr, next_node)
                if edge in visited_edges: break 
                
                visited_edges.add(edge)
                curr = next_node
            
            if is_closed:
                valid_routes.append(path)

        subtours = []
        remaining_edges = [e for e in active_edges if e not in visited_edges]
        adj_rem = {u:v for u, v in remaining_edges}
        visited_rem = set()
        
        for u in adj_rem:
            if u in visited_rem: continue
            cycle = []
            curr = u
            steps = 0
            while curr in adj_rem and curr not in visited_rem and steps < self.instance.n:
                visited_rem.add(curr)
                cycle.append(curr)
                curr = adj_rem[curr]
                steps += 1
                if curr == u:
                    cycle.append(curr)
                    subtours.append(cycle)
                    break
            
            if cycle and cycle[0] != cycle[-1]:
                subtours.append(cycle)

        return valid_routes, subtours

    def _greedy_split(self, route: List[int]) -> List[List[int]]:
        split_routes = []
        capacity = self.instance.capacity
        current_segment = [0]
        current_load = 0
        
        customers = route[1:-1]
        for cust in customers:
            d = self.instance.demands[cust]
            if current_load + d <= capacity:
                current_segment.append(cust)
                current_load += d
            else:
                current_segment.append(0)
                split_routes.append(current_segment)
                current_segment = [0, cust]
                current_load = d
        
        current_segment.append(0)
        if len(current_segment) > 2:
            split_routes.append(current_segment)
        
        return split_routes

    def _repair_unvisited_nodes(self, valid_routes: List[List[int]], unvisited_nodes: List[int]) -> List[List[int]]:
        if not unvisited_nodes:
            return valid_routes
        routes = [r[:] for r in valid_routes]
        if not routes: routes.append([0, 0])
        dist = self.instance.dist_matrix

        for node in unvisited_nodes:
            best_cost_increase = float('inf')
            best_route_idx = -1
            best_insert_pos = -1
            
            for r_idx, route in enumerate(routes):
                for i in range(len(route) - 1):
                    u, v = route[i], route[i+1]
                    increase = dist[u][node] + dist[node][v] - dist[u][v]
                    if increase < best_cost_increase:
                        best_cost_increase = increase
                        best_route_idx = r_idx
                        best_insert_pos = i + 1
            
            if best_route_idx != -1:
                routes[best_route_idx].insert(best_insert_pos, node)
            else:
                routes.append([0, node, 0])
        return routes

    def solve(self):
        print(f"🚀 [HybridLNS] Running Robust LNS for {self.instance.name}...")
        start_time = time.time()
        
        initial_routes = clarke_wright_savings(self.instance.dist_matrix, self.instance.demands, self.instance.capacity)
        initial_routes = [two_opt(r, self.instance.dist_matrix) for r in initial_routes]
        
        best_routes = initial_routes
        best_cost = total_distance(best_routes, self.instance.dist_matrix)
        print(f"   [Init] Initial Cost: {best_cost:.2f}")

        stagnation_counter = 0

        # 2. MAIN LOOP
        for it in range(1, MAX_ITERATIONS + 1):
            # Tạo graph có yếu tố ngẫu nhiên
            allowed_edges = self._build_restricted_graph(best_routes)
            current_subtour_cuts = [] 
            candidate_routes = []
            
            # Inner Loop (SAT Solver)
            for inner_it in range(self.max_inner_iter):
                wcnf, edge_map = encode_edges_as_wcnf(
                    range(self.instance.n), 
                    allowed_edges, 
                    self.instance.dist_matrix,
                    subtour_cuts=current_subtour_cuts
                )
                
                # Tên file unique để tránh lỗi I/O
                wcnf_file = f"lns_{self.instance.name}_iter{it}_{inner_it}.wcnf"
                wcnf.to_file(wcnf_file)
                
                out = call_openwbo(wcnf_file, timeout=TIMEOUT_LNS_INNER)
                if os.path.exists(wcnf_file): os.remove(wcnf_file)
                
                vars_true = parse_openwbo_model(out)
                
                if not vars_true:
                    # Nếu Solver fail (UNSAT/Timeout), thử thêm random edges ở vòng sau
                    break 

                chosen_edges = []
                for u, v in edge_map:
                    if edge_map[(u, v)] in vars_true:
                        chosen_edges.append((u, v))
                
                if not chosen_edges: break
                
                valid_routes, subtours = self._extract_routes_from_edges(chosen_edges)
                
                covered_nodes = set()
                for r in valid_routes:
                    for n in r: covered_nodes.add(n)
                missing_nodes = [n for n in range(1, self.instance.n) if n not in covered_nodes]

                if not missing_nodes and not subtours:
                    candidate_routes = valid_routes
                    break
                else:
                    if inner_it < self.max_inner_iter - 1:
                        if subtours:
                            for st in subtours:
                                cut = []
                                for k in range(len(st)-1):
                                    cut.append((st[k], st[k+1]))
                                current_subtour_cuts.append(cut)
                            continue 
                        else:
                             candidate_routes = self._repair_unvisited_nodes(valid_routes, missing_nodes)
                             break
                    else:
                        candidate_routes = self._repair_unvisited_nodes(valid_routes, missing_nodes)
                        break

            # C. Post-Processing & Update
            if candidate_routes:
                feasible_routes = []
                for r in candidate_routes:
                    load = sum(self.instance.demands[n] for n in r)
                    if load > self.instance.capacity:
                        splits = self._greedy_split(r)
                        feasible_routes.extend(splits)
                    else:
                        feasible_routes.append(r)
                
                optimized_routes = [two_opt(r, self.instance.dist_matrix) for r in feasible_routes]
                current_cost = total_distance(optimized_routes, self.instance.dist_matrix)
                
                if current_cost < best_cost - 1e-4:
                    print(f"   ✅ ITER {it}: {best_cost:.2f} -> {current_cost:.2f}")
                    best_cost = current_cost
                    best_routes = optimized_routes
                    stagnation_counter = 0
                else:
                    # LOGGING: In ra ngay cả khi không cải thiện để biết chương trình đang chạy
                    print(f"   .  ITER {it}: {current_cost:.2f} (No improvement)")
                    stagnation_counter += 1
            else:
                print(f"   ⚠️ ITER {it}: Solver failed to find feasible routes.")
                stagnation_counter += 1

            # Kiểm tra điều kiện dừng sớm
            if stagnation_counter >= self.stagnation_limit:
                print(f"   🛑 Stopping early due to stagnation ({self.stagnation_limit} iters without improvement).")
                break

        # 3. FINISH
        total_time = time.time() - start_time
        print("\n" + "="*50)
        print(f"🏆 FINAL RESULT")
        print(f"   Cost: {best_cost:.2f}")
        
        log_benchmark(self.instance, best_routes, 
                      total_distance(initial_routes, self.instance.dist_matrix),
                      best_cost, total_time, it, 0)
        try: plot_solution(self.instance, best_routes, best_cost)
        except: pass
        
        return best_cost, best_routes
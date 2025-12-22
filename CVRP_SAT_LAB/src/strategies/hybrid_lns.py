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
        
        # [DYNAMIC CONFIG] Tự động điều chỉnh tham số dựa trên kích thước bài toán
        # Với bài toán nhỏ (như E-n31), ta cần không gian tìm kiếm dày đặc hơn
        if instance.n < 45:
            self.k_nearest = 20
            self.max_inner_iter = 5  # Thử nhiều lần hơn vì solver chạy nhanh
        else:
            self.k_nearest = 10
            self.max_inner_iter = 3

        self.stagnation_limit = 20 
        
        # [GLS CONFIG] Guided Local Search
        self.penalties = {} 
        self.lambda_factor = 0.1 

    def _build_restricted_graph(self, current_routes: List[List[int]]) -> List[Tuple[int, int]]:
        edges = set()
        N = self.instance.n
        D = self.instance.dist_matrix
        
        # 1. Giữ lại cạnh từ nghiệm hiện tại (Inheritance)
        keep_probability = 0.6 
        for r in current_routes:
            for i in range(len(r) - 1):
                if random.random() < keep_probability:
                    edges.add((r[i], r[i+1]))
        
        # 2. Thêm K láng giềng gần nhất (Spatial Locality)
        # K sẽ thay đổi tùy theo kích thước bài toán (đã set trong __init__)
        for i in range(N):
            nearest_indices = np.argsort(D[i])[1:self.k_nearest+1]
            for neighbor in nearest_indices:
                if i != neighbor:
                    edges.add((i, neighbor))
                    edges.add((neighbor, i)) 
        
        # 3. Đảm bảo kết nối với Depot
        depot_nearest = np.argsort(D[0])[1:self.k_nearest+1]
        for i in depot_nearest:
            edges.add((0, i))
            edges.add((i, 0))
            
        for r in current_routes:
            if len(r) > 2:
                first, last = r[1], r[-2]
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
            while steps < self.instance.n * 2: 
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

    # [NEW] THAY THẾ HOÀN TOÀN _greedy_split BẰNG _split_optimal
    def _split_optimal(self, route: List[int]) -> List[List[int]]:
        """
        Optimal Split Algorithm (Prins 2004)
        Sử dụng Quy hoạch động (Dynamic Programming) trên đồ thị DAG để tìm cách chia tách
        lộ trình thành các chuyến xe con sao cho tổng chi phí là nhỏ nhất và thỏa mãn tải trọng.
        """
        # 1. Trích xuất danh sách khách hàng (bỏ depot đầu/cuối)
        route_cust = [x for x in route if x != 0]
        if not route_cust:
            return []

        n = len(route_cust)
        capacity = self.instance.capacity
        dist = self.instance.dist_matrix
        demands = self.instance.demands
        
        # V[i] là chi phí thấp nhất để phục vụ i khách hàng đầu tiên trong danh sách
        # Khởi tạo V[0] = 0, còn lại là vô cùng
        V = [float('inf')] * (n + 1)
        V[0] = 0
        
        # P[i] lưu điểm "ngắt" trước đó để truy vết (Predecessor)
        P = [0] * (n + 1)
        
        # 2. Quy hoạch động
        for i in range(n): # i là điểm bắt đầu chuyến xe mới (index trong route_cust là i)
            load = 0
            cost = 0
            
            # j là điểm kết thúc chuyến xe
            for j in range(i, n):
                cust_j = route_cust[j]
                load += demands[cust_j]
                
                if load > capacity:
                    break # Nếu quá tải thì dừng mở rộng chuyến này
                
                # Tính chi phí phát sinh cho chuyến xe từ i đến j
                if i == j:
                    # Chuyến chỉ có 1 khách: 0 -> cust -> 0
                    cost = dist[0][cust_j] + dist[cust_j][0]
                else:
                    # Chuyến xe kéo dài từ khách hàng trước đó (prev) đến khách hàng hiện tại (curr)
                    # Cost mới = Cost cũ - cạnh về kho cũ + cạnh nối + cạnh về kho mới
                    prev_cust = route_cust[j-1]
                    cost = cost - dist[prev_cust][0] + dist[prev_cust][cust_j] + dist[cust_j][0]
                
                # Cập nhật V[j+1] nếu tìm thấy phương án chia tốt hơn
                if V[i] + cost < V[j+1]:
                    V[j+1] = V[i] + cost
                    P[j+1] = i
                    
        # 3. Truy vết để tái tạo các chuyến xe (Routes)
        split_routes = []
        curr = n
        while curr > 0:
            prev = P[curr]
            # Route segment từ prev đến curr (trong mảng route_cust)
            # Thêm 0 ở đầu và cuối để thành route hoàn chỉnh
            segment = [0] + route_cust[prev:curr] + [0]
            split_routes.append(segment)
            curr = prev
            
        return split_routes[::-1] # Đảo ngược lại vì ta truy vết từ cuối về đầu

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

    def _update_penalties(self, solution_routes: List[List[int]]):
        edges_to_consider = []
        for r in solution_routes:
            for i in range(len(r) - 1):
                u, v = r[i], r[i+1]
                if u > v: u, v = v, u 
                edges_to_consider.append((u, v))
        
        if not edges_to_consider: return

        max_utility = -1
        candidates = []
        
        for u, v in edges_to_consider:
            dist = self.instance.dist_matrix[u][v]
            penalty = self.penalties.get((u, v), 0)
            utility = dist / (1 + penalty)
            
            if utility > max_utility:
                max_utility = utility
                candidates = [(u, v)]
            elif abs(utility - max_utility) < 1e-6:
                candidates.append((u, v))
        
        for u, v in candidates:
            self.penalties[(u, v)] = self.penalties.get((u, v), 0) + 1

    def solve(self):
        print(f"🚀 [HybridLNS + GLS + OptimalSplit] Running for {self.instance.name}...")
        start_time = time.time()
        
        initial_routes = clarke_wright_savings(self.instance.dist_matrix, self.instance.demands, self.instance.capacity)
        initial_routes = [two_opt(r, self.instance.dist_matrix) for r in initial_routes]
        
        best_routes = initial_routes
        best_cost = total_distance(best_routes, self.instance.dist_matrix)
        print(f"   [Init] Initial Cost: {best_cost:.2f}")

        stagnation_counter = 0
        avg_dist = np.mean(self.instance.dist_matrix)

        for it in range(1, MAX_ITERATIONS + 1):
            
            # GLS Weight Update
            augmented_dist_matrix = np.copy(self.instance.dist_matrix)
            if self.penalties:
                for (u, v), count in self.penalties.items():
                    penalty_value = int(self.lambda_factor * avg_dist * count)
                    augmented_dist_matrix[u][v] += penalty_value
                    augmented_dist_matrix[v][u] += penalty_value

            allowed_edges = self._build_restricted_graph(best_routes)
            current_subtour_cuts = [] 
            candidate_routes = []
            
            for inner_it in range(self.max_inner_iter):
                wcnf, edge_map = encode_edges_as_wcnf(
                    range(self.instance.n), 
                    allowed_edges, 
                    augmented_dist_matrix, 
                    subtour_cuts=current_subtour_cuts
                )
                
                wcnf_file = f"lns_{self.instance.name}_iter{it}_{inner_it}.wcnf"
                wcnf.to_file(wcnf_file)
                
                out = call_openwbo(wcnf_file, timeout=TIMEOUT_LNS_INNER)
                if os.path.exists(wcnf_file): os.remove(wcnf_file)
                
                vars_true = parse_openwbo_model(out)
                
                if not vars_true:
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
                    # LUÔN LUÔN DÙNG OPTIMAL SPLIT (Thay vì chỉ dùng khi quá tải)
                    # Lý do: Optimal Split có thể tối ưu lại cả những route "tưởng là ngon"
                    # nhưng thực ra cắt chưa khéo.
                    if load > self.instance.capacity:
                         splits = self._split_optimal(r) # Sử dụng hàm mới
                         feasible_routes.extend(splits)
                    else:
                         # Bạn cũng có thể thử chạy split_optimal ngay cả khi load <= capacity
                         # để xem nó có tìm được cách sắp xếp tốt hơn không.
                         # Nhưng để an toàn và nhanh, ta chỉ chạy khi quá tải.
                         feasible_routes.append(r)
                
                optimized_routes = [two_opt(r, self.instance.dist_matrix) for r in feasible_routes]
                current_cost = total_distance(optimized_routes, self.instance.dist_matrix)
                
                if current_cost < best_cost - 1e-4:
                    print(f"   ✅ ITER {it}: {best_cost:.2f} -> {current_cost:.2f}")
                    best_cost = current_cost
                    best_routes = optimized_routes
                    stagnation_counter = 0 
                else:
                    stagnation_counter += 1
                    print(f"   .  ITER {it}: {current_cost:.2f} (No improv) - Stag: {stagnation_counter}")
                    
                    if stagnation_counter % 5 == 0:
                        print(f"      🔥 [GLS] Local Optima detected. Updating penalties...")
                        self._update_penalties(best_routes)

            else:
                print(f"   ⚠️ ITER {it}: Solver failed to find feasible routes.")
                stagnation_counter += 1

            if stagnation_counter >= self.stagnation_limit:
                print(f"   🛑 Stopping early due to stagnation ({self.stagnation_limit} iters).")
                break

        total_time = time.time() - start_time
        print("\n" + "="*50)
        print(f"🏆 FINAL RESULT (Enhanced)")
        print(f"   Cost: {best_cost:.2f}")
        
        log_benchmark(self.instance, best_routes, 
                      total_distance(initial_routes, self.instance.dist_matrix),
                      best_cost, total_time, it, 0)
        try: plot_solution(self.instance, best_routes, best_cost)
        except: pass
        
        return best_cost, best_routes
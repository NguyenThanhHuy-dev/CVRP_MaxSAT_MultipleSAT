import time
import os
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
        # K=5: Cân bằng giữa tốc độ và khả năng tìm kiếm
        self.k_nearest = 5       
        # Số lần thử cắt Subtour trước khi bỏ cuộc và dùng Repair thủ công
        self.max_inner_iter = 5 

    def _build_restricted_graph(self, current_routes: List[List[int]]) -> List[Tuple[int, int]]:
        edges = set()
        N = self.instance.n
        D = self.instance.dist_matrix
        
        # 1. Giữ lại cạnh cũ
        for r in current_routes:
            for i in range(len(r) - 1):
                edges.add((r[i], r[i+1]))
        
        # 2. Thêm K-Nearest Neighbors
        for i in range(N):
            nearest_indices = np.argsort(D[i])[1:self.k_nearest+1]
            for neighbor in nearest_indices:
                if i != neighbor:
                    edges.add((i, neighbor))
                    edges.add((neighbor, i)) 
        
        # 3. [SAFETY] Luôn kết nối Depot với tất cả Node
        # Giúp Solver không bị UNSAT và dễ dàng Repair nếu cần
        for i in range(1, N):
            edges.add((0, i))
            edges.add((i, 0))
                    
        return list(edges)

    def _extract_routes_from_edges(self, active_edges: List[Tuple[int, int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Trích xuất tuyến đường một cách an toàn.
        - valid_routes: Các tuyến 0 -> ... -> 0
        - subtours: Các chu trình hoặc đường dẫn bị đứt đoạn không chứa 0
        """
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
        visited_edges = set() # Tránh lặp vô hạn
        
        # BƯỚC 1: Duyệt các tuyến xuất phát từ Depot
        for start_node in depot_outgoing:
            if (0, start_node) in visited_edges: continue
            
            path = [0]
            curr = start_node
            visited_edges.add((0, start_node))
            
            is_closed = False
            while True:
                path.append(curr)
                if curr == 0: # Quay về đích an toàn
                    is_closed = True
                    break
                
                if curr not in adj: break # Cụt đường
                next_node = adj[curr]
                
                edge = (curr, next_node)
                if edge in visited_edges: break # Gặp lại cạnh cũ (Subtour dính vào)
                
                visited_edges.add(edge)
                curr = next_node
            
            if is_closed:
                valid_routes.append(path)
            else:
                # Tuyến bị hở (không về được 0), coi như rác cần Repair
                # Ta không thêm vào valid_routes, để logic Repair tự nhặt lại các node này
                pass 

        # BƯỚC 2: Tìm Subtours (Các cạnh còn lại chưa được duyệt)
        subtours = []
        remaining_edges = [e for e in active_edges if e not in visited_edges]
        
        # Xây dựng lại map cho phần còn lại
        adj_rem = {u:v for u, v in remaining_edges}
        visited_rem = set()
        
        for u in adj_rem:
            if u in visited_rem: continue
            
            # Duyệt chu trình
            cycle = []
            curr = u
            while curr in adj_rem and curr not in visited_rem:
                visited_rem.add(curr)
                cycle.append(curr)
                curr = adj_rem[curr]
                
                # Nếu quay lại điểm đầu -> Subtour kín
                if curr == u:
                    cycle.append(curr) # Đóng vòng
                    subtours.append(cycle)
                    break
            
            # Nếu là đường thẳng rời rạc (1->2->3 nhưng không về 1), cũng gom vào subtour để xử lý
            if cycle and cycle[0] != cycle[-1]:
                subtours.append(cycle)

        return valid_routes, subtours

    def _greedy_split(self, route: List[int]) -> List[List[int]]:
        """Chia tuyến đường quá tải Capacity thành nhiều tuyến nhỏ."""
        split_routes = []
        capacity = self.instance.capacity
        demands = self.instance.demands
        
        current_segment = [0]
        current_load = 0
        
        customers = route[1:-1]
        for cust in customers:
            d = demands[cust]
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
        """
        Chiến thuật 'Cứu hộ': Chèn các node bị bỏ rơi vào vị trí tốt nhất có thể.
        """
        if not unvisited_nodes:
            return valid_routes
            
        routes = [r[:] for r in valid_routes]
        if not routes: routes.append([0, 0]) # Tạo tuyến rỗng nếu cần
        
        dist = self.instance.dist_matrix

        for node in unvisited_nodes:
            best_cost_increase = float('inf')
            best_route_idx = -1
            best_insert_pos = -1
            
            # Tìm vị trí chèn (Best Insertion)
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
        
        # 1. INIT
        initial_routes = clarke_wright_savings(self.instance.dist_matrix, self.instance.demands, self.instance.capacity)
        initial_routes = [two_opt(r, self.instance.dist_matrix) for r in initial_routes]
        
        best_routes = initial_routes
        best_cost = total_distance(best_routes, self.instance.dist_matrix)
        print(f"   [Init] Initial Cost: {best_cost:.2f}")

        # 2. MAIN LOOP
        for it in range(1, MAX_ITERATIONS + 1):
            # A. Build Graph
            allowed_edges = self._build_restricted_graph(best_routes)
            current_subtour_cuts = [] 
            
            candidate_routes = []
            
            # B. Inner Loop (SAT Solver)
            for inner_it in range(self.max_inner_iter):
                wcnf, edge_map = encode_edges_as_wcnf(
                    range(self.instance.n), 
                    allowed_edges, 
                    self.instance.dist_matrix,
                    subtour_cuts=current_subtour_cuts
                )
                
                wcnf_file = f"lns_iter_{it}_{inner_it}.wcnf"
                wcnf.to_file(wcnf_file)
                
                out = call_openwbo(wcnf_file, timeout=TIMEOUT_LNS_INNER)
                if os.path.exists(wcnf_file): os.remove(wcnf_file)
                
                vars_true = parse_openwbo_model(out)
                
                if not vars_true:
                    break # Solver failed

                chosen_edges = []
                for u, v in edge_map:
                    if edge_map[(u, v)] in vars_true:
                        chosen_edges.append((u, v))
                
                if not chosen_edges: break
                
                # Trích xuất tuyến đường
                valid_routes, subtours = self._extract_routes_from_edges(chosen_edges)
                
                # --- LOGIC QUAN TRỌNG: COVERAGE CHECK ---
                # Gom tất cả node hiện có trong valid_routes
                covered_nodes = set()
                for r in valid_routes:
                    for n in r: covered_nodes.add(n)
                
                # Tìm các node bị thiếu (bao gồm cả node trong subtours và node bị rơi)
                missing_nodes = [n for n in range(1, self.instance.n) if n not in covered_nodes]

                # Điều kiện chấp nhận nghiệm:
                # 1. Nếu đủ node và không Subtour -> Perfect
                # 2. Nếu thiếu node hoặc có Subtour -> Kiểm tra xem nên Repair hay Retry?
                
                if not missing_nodes and not subtours:
                    # Nghiệm hoàn hảo
                    candidate_routes = valid_routes
                    break
                
                else:
                    # Nghiệm lỗi (Subtour hoặc thiếu node)
                    if inner_it < self.max_inner_iter - 1:
                        # Nếu còn lượt, ưu tiên thêm Cut để Solver tự sửa (tốt hơn Repair tham lam)
                        if subtours:
                            for st in subtours:
                                cut = []
                                for k in range(len(st)-1):
                                    cut.append((st[k], st[k+1]))
                                current_subtour_cuts.append(cut)
                            continue # Quay lại đầu Inner Loop
                        else:
                             # Không có Subtour nhưng vẫn thiếu node (lỗi đồ thị) -> Buộc phải Repair ngay
                             candidate_routes = self._repair_unvisited_nodes(valid_routes, missing_nodes)
                             break
                    else:
                        # Hết lượt Inner Loop: Chấp nhận sửa lỗi thủ công
                        candidate_routes = self._repair_unvisited_nodes(valid_routes, missing_nodes)
                        break

            # C. Post-Processing
            if candidate_routes:
                # 1. Split (Capacity Check)
                feasible_routes = []
                for r in candidate_routes:
                    load = sum(self.instance.demands[n] for n in r)
                    if load > self.instance.capacity:
                        splits = self._greedy_split(r)
                        feasible_routes.extend(splits)
                    else:
                        feasible_routes.append(r)
                
                # 2. Local Search (2-Opt) - Làm mượt sau khi Repair/Split
                optimized_routes = [two_opt(r, self.instance.dist_matrix) for r in feasible_routes]
                
                # 3. Update Best Solution
                current_cost = total_distance(optimized_routes, self.instance.dist_matrix)
                
                if current_cost < best_cost - 1e-4:
                    print(f"   ✅ ITER {it}: {best_cost:.2f} -> {current_cost:.2f}")
                    best_cost = current_cost
                    best_routes = optimized_routes
                else:
                    # print(f"   . Not improved ({current_cost:.2f})")
                    pass
            else:
                pass # Inner loop failed hoàn toàn, bỏ qua

        # 3. FINISH
        total_time = time.time() - start_time
        print("\n" + "="*50)
        print(f"🏆 FINAL RESULT")
        print(f"   Cost: {best_cost:.2f}")
        
        log_benchmark(self.instance, best_routes, 
                      total_distance(initial_routes, self.instance.dist_matrix),
                      best_cost, total_time, MAX_ITERATIONS, 0)
        try: plot_solution(self.instance, best_routes, best_cost)
        except: pass
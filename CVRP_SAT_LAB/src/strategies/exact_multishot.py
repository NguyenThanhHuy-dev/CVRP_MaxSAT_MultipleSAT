import time
import os
import numpy as np
from typing import List, Tuple

from data_loader import Instance
from encoders.edge_encoder import encode_edges_as_wcnf
from solver_service import call_openwbo
from utils.decoder import parse_openwbo_model
from utils.plot import plot_solution
from heuristic import total_distance
from config import TIMEOUT_SOLVER # Dùng timeout dài hơn (60s - 300s)

class ExactMultiShotStrategy:
    def __init__(self, instance: Instance):
        self.instance = instance
        # Với phương pháp chính xác, ta nên dùng K lớn hoặc Full Graph
        # Tuy nhiên để chạy nổi N=45, ta tạm dùng K=20 (chấp nhận rủi ro nhỏ)
        # Nếu muốn Optimal 100% lý thuyết, phải dùng Full Graph (K = N)
        self.k_nearest = 20  
        
    def _build_graph(self) -> List[Tuple[int, int]]:
        """
        Xây dựng đồ thị. Với phương pháp Exact, ta cần đồ thị dày hơn LNS.
        """
        edges = set()
        N = self.instance.n
        D = self.instance.dist_matrix
        
        # 1. Thêm K-Nearest Neighbors
        for i in range(N):
            # Lấy nhiều hàng xóm hơn LNS để đảm bảo không bỏ sót cạnh tối ưu
            nearest_indices = np.argsort(D[i])[1:self.k_nearest+1]
            for neighbor in nearest_indices:
                if i != neighbor:
                    edges.add((i, neighbor))
                    edges.add((neighbor, i))
        
        # 2. Bắt buộc kết nối Depot với TOÀN BỘ node (để đảm bảo tính đầy đủ)
        for i in range(1, N):
            edges.add((0, i))
            edges.add((i, 0))
            
        return list(edges)

    def _extract_routes(self, active_edges: List[Tuple[int, int]]) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Trích xuất tuyến đường. Logic giống LNS nhưng chặt chẽ hơn.
        """
        adj = {u: v for u, v in active_edges}
        routes = []
        subtours = []
        visited = set()
        
        # Duyệt từ Depot
        depot_outgoing = [v for u, v in active_edges if u == 0]
        for start in depot_outgoing:
            if start in visited: continue
            path = [0]
            curr = start
            visited.add((0, start)) # Đánh dấu cạnh
            
            while True:
                path.append(curr)
                if curr == 0: # Về đích
                    routes.append(path)
                    break
                if curr not in adj: # Cụt đường (Lỗi Solver)
                    routes.append(path) 
                    break
                
                next_node = adj[curr]
                if next_node in path and next_node != 0: # Subtour dính vào
                    routes.append(path) # Coi như tuyến lỗi
                    break
                    
                curr = next_node

        # Duyệt các mảnh còn lại (Subtours rời rạc)
        # ... (Logic tìm subtour tương tự LNS)
        # Để code gọn, ta có thể tái sử dụng hàm check subtour.
        # Ở đây mình viết đơn giản hóa logic kiểm tra:
        
        all_nodes_in_routes = set()
        for r in routes:
            for n in r: all_nodes_in_routes.add(n)
            
        # Nếu thiếu node -> Chắc chắn có subtour rời rạc hoặc node bị cô lập
        missing = [n for n in range(self.instance.n) if n not in all_nodes_in_routes]
        
        return routes, missing

    def solve(self):
        print(f"🔬 [ExactMultiShot] Running Exact Iterative SAT for {self.instance.name}...")
        start_time = time.time()
        
        # 1. Build Graph
        allowed_edges = self._build_graph()
        cuts = [] # Lưu trữ cả Subtour Cuts và Capacity Cuts
        
        best_routes = []
        best_cost = float('inf')
        
        iteration = 0
        while True:
            iteration += 1
            print(f"   Using {len(cuts)} cuts...", end="\r")
            
            # 2. Encode & Solve
            wcnf, edge_map = encode_edges_as_wcnf(
                range(self.instance.n), 
                allowed_edges, 
                self.instance.dist_matrix,
                subtour_cuts=cuts
            )
            
            filename = f"exact_iter_{iteration}.wcnf"
            wcnf.to_file(filename)
            
            # Gọi Solver (Cần timeout cao vì đây là Exact method)
            out = call_openwbo(filename, timeout=300) 
            if os.path.exists(filename): os.remove(filename)
            
            vars_true = parse_openwbo_model(out)
            
            if not vars_true:
                print("\n   ❌ Solver returned UNSAT or Timeout. Stopping.")
                break
                
            # 3. Extract Solution
            chosen_edges = []
            for u, v in edge_map:
                if edge_map[(u, v)] in vars_true:
                    chosen_edges.append((u, v))
            
            # 4. Verification (Kiểm tra tính đúng đắn)
            # A. Check Subtours (Kiểm tra kết nối)
            # Ta dùng logic tìm chu trình đơn giản:
            adj = {u:v for u, v in chosen_edges}
            
            has_error = False
            
            # Tìm tất cả các chu trình (Cycles)
            visited_global = set()
            current_subtours = []
            
            for node in range(self.instance.n):
                if node in visited_global: continue
                if node not in adj: continue # Node bị cô lập (Lỗi)
                
                path = [node]
                curr = adj[node]
                visited_global.add(node)
                
                while curr not in path and curr not in visited_global and curr in adj:
                    visited_global.add(curr)
                    path.append(curr)
                    curr = adj[curr]
                
                if curr in path: # Tìm thấy chu trình
                    # Nếu chu trình không chứa 0 -> Là Subtour -> CẮT
                    if 0 not in path:
                        # Cắt chính xác chu trình này
                        # Logic cắt: sum(x_ij) <= |S| - 1
                        cycle_edges = []
                        idx = path.index(curr)
                        cycle = path[idx:]
                        cycle.append(cycle[0]) # Đóng vòng
                        
                        for k in range(len(cycle)-1):
                            cycle_edges.append((cycle[k], cycle[k+1]))
                        
                        cuts.append(cycle_edges)
                        has_error = True
                        print(f"\n   Detected Subtour: {[int(x) for x in cycle]}. Adding Cut.")
            
            if has_error: 
                continue # Quay lại giải tiếp
            
            # B. Check Capacity (Kiểm tra tải trọng)
            # Đến đây đảm bảo nghiệm là tập hợp các Route hợp lệ đi qua 0
            # Giờ ta kiểm tra từng Route xem có quá tải không
            current_routes = []
            # Trích xuất lại routes tử tế
            depot_starts = [v for u, v in chosen_edges if u == 0]
            for s in depot_starts:
                r = [0, s]
                curr = s
                while curr != 0:
                    curr = adj[curr]
                    r.append(curr)
                current_routes.append(r)
                
            capacity_violated = False
            for r in current_routes:
                load = sum(self.instance.demands[n] for n in r)
                if load > self.instance.capacity:
                    # Tuyến đường bị quá tải!
                    # CẮT: Cấm tập cạnh này xuất hiện đồng thời
                    # Logic: Tuyến r = [0, 1, 2, 0] quá tải
                    # Cut: NOT(x01) OR NOT(x12) OR NOT(x20)
                    bad_edges = []
                    for k in range(len(r)-1):
                        bad_edges.append((r[k], r[k+1]))
                    cuts.append(bad_edges)
                    capacity_violated = True
                    print(f"\n   Route Overloaded (L={load}): {r}. Adding Capacity Cut.")
            
            if capacity_violated:
                continue # Quay lại giải tiếp
            
            # 5. Nếu vượt qua cả Subtour Check và Capacity Check
            # -> ĐÂY LÀ NGHIỆM TỐI ƯU (đối với đồ thị hiện tại)
            best_routes = current_routes
            best_cost = total_distance(best_routes, self.instance.dist_matrix)
            print(f"\n   ✅ OPTIMAL FOUND (on current graph): {best_cost:.2f}")
            break

        # FINISH
        total_time = time.time() - start_time
        print("\n" + "="*50)
        print(f"🏆 EXACT MULTI-SHOT RESULT")
        print(f"   Cost: {best_cost:.2f}")
        print(f"   Iterations: {iteration}")
        
        # Gọi logger (nhớ import log_benchmark)
        # log_benchmark(...)
        try: plot_solution(self.instance, best_routes, best_cost)
        except: pass
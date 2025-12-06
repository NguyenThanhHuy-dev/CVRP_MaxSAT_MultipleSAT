import time
import os
import random
import numpy as np
from typing import List

# --- SỬA CÁC DÒNG IMPORT DƯỚI ĐÂY ---
from data_loader import Instance
from heuristic import clarke_wright_savings, two_opt, total_distance, route_cost
from encoders.route_encoder import encode_routes_as_wcnf, write_wcnf_to_file
from solver_service import call_openwbo
from utils.decoder import parse_openwbo_model, chosen_routes_from_vars
from utils.plot import plot_solution
from utils.logger import log_benchmark
from config import TIMEOUT_SOLVER, MAX_ITERATIONS, MAX_POOL_SIZE
# ------------------------------------

class ColumnGenerationStrategy:
    # ... (Phần code còn lại của Class giữ nguyên như cũ) ...
    # ... (Bạn chỉ cần sửa đoạn import ở trên thôi) ...
    def __init__(self, instance: Instance):
        self.instance = instance
        self.route_pool = {} # Key: tuple(route), Value: cost

    def _generate_mutation(self, current_best_routes: List[List[int]]) -> List[List[int]]:
        """Sinh cột mới: 2-opt và Crossover (Swap)."""
        new_candidates = []
        dist_matrix = self.instance.dist_matrix
        demands = self.instance.demands
        capacity = self.instance.capacity
        
        # 1. 2-opt improvement
        for r in current_best_routes:
            improved = two_opt(r, dist_matrix, max_iter=500)
            if route_cost(improved, dist_matrix) < route_cost(r, dist_matrix) - 1e-5:
                new_candidates.append(improved)

        # 2. Crossover (Lai ghép)
        n_routes = len(current_best_routes)
        if n_routes >= 2:
            num_trials = min(10, n_routes * 2)
            for _ in range(num_trials):
                idx1, idx2 = np.random.choice(n_routes, 2, replace=False)
                r1, r2 = current_best_routes[idx1], current_best_routes[idx2]
                
                if len(r1) > 3 and len(r2) > 3:
                    cut1 = random.randint(1, len(r1) - 2)
                    cut2 = random.randint(1, len(r2) - 2)
                    
                    child1 = r1[:cut1] + r2[cut2:]
                    child2 = r2[:cut2] + r1[cut1:]
                    
                    # Helper check valid
                    def is_valid(route):
                        if len(route) <= 2: return False
                        if route[0] != 0 or route[-1] != 0: return False
                        return sum(demands[n] for n in route) <= capacity

                    if child1[-1] != 0: child1.append(0)
                    if child2[-1] != 0: child2.append(0)

                    if is_valid(child1): new_candidates.append(two_opt(child1, dist_matrix))
                    if is_valid(child2): new_candidates.append(two_opt(child2, dist_matrix))

        return new_candidates

    def _generate_merge(self, current_routes: List[List[int]]) -> List[List[int]]:
        """Gộp tuyến nhỏ vào tuyến lớn để giảm số lượng xe."""
        candidates = []
        dist_matrix = self.instance.dist_matrix
        demands = self.instance.demands
        capacity = self.instance.capacity
        
        # Sắp xếp các tuyến theo độ dài tăng dần (ưu tiên xử lý tuyến ngắn)
        sorted_indices = np.argsort([len(r) for r in current_routes])
        
        for idx_src in sorted_indices:
            src_route = current_routes[idx_src]
            if len(src_route) > 6: continue
            
            customers_to_move = [c for c in src_route if c != 0]
            
            for idx_dest, dest_route in enumerate(current_routes):
                if idx_src == idx_dest: continue 
                
                new_route = list(dest_route)
                possible = True
                
                for cust in customers_to_move:
                    # --- QUAN TRỌNG: Tránh thêm khách đã có (Fix lỗi trùng lặp) ---
                    if cust in new_route: continue 
                    # -------------------------------------------------------------

                    best_pos = -1
                    best_increase = float('inf')
                    
                    # Greedy insertion
                    for i in range(1, len(new_route)):
                        increase = (dist_matrix[new_route[i-1], cust] + 
                                    dist_matrix[cust, new_route[i]] - 
                                    dist_matrix[new_route[i-1], new_route[i]])
                        if increase < best_increase:
                            best_increase = increase
                            best_pos = i
                    
                    if best_pos != -1:
                        new_route.insert(best_pos, cust)
                    else:
                        possible = False
                        break
                
                if possible and sum(demands[c] for c in new_route) <= capacity:
                    optimized_new_route = two_opt(new_route, dist_matrix)
                    candidates.append(optimized_new_route)

        return candidates

    def _clean_solution(self, routes: List[List[int]]) -> List[List[int]]:
        """Hậu xử lý: Loại bỏ khách trùng lặp, giữ lại lần đầu tiên."""
        served = set()
        cleaned_routes = []
        
        for r in routes:
            new_r = [0]
            for node in r[1:-1]:
                if node not in served:
                    served.add(node)
                    new_r.append(node)
            new_r.append(0)
            
            if len(new_r) > 2:
                cleaned_routes.append(new_r)
        
        # Tối ưu lại bằng 2-opt sau khi xóa điểm
        final_routes = [two_opt(r, self.instance.dist_matrix) for r in cleaned_routes]
        return final_routes

    def solve(self):
        print(f"🚀 [ColumnGeneration] Running for {self.instance.name}...")
        start_time = time.time()
        
        # 1. INIT
        initial_routes = clarke_wright_savings(self.instance.dist_matrix, self.instance.demands, self.instance.capacity)
        initial_routes = [two_opt(r, self.instance.dist_matrix) for r in initial_routes]
        
        initial_cost = total_distance(initial_routes, self.instance.dist_matrix)
        print(f"   [Init] Initial Heuristic Cost: {initial_cost:.2f}")
        
        # Initialize Pool
        self.route_pool = {tuple(r): route_cost(r, self.instance.dist_matrix) for r in initial_routes}
        
        best_overall_cost = float('inf')
        best_solution_routes = []
        final_iterations = 0
        
        print(f"   [Init] Pool size: {len(self.route_pool)}")

        # 2. MAIN LOOP
        for it in range(1, MAX_ITERATIONS + 1):
            final_iterations = it
            print(f"\n🔄 ITERATION {it}/{MAX_ITERATIONS}")
            
            pool_list = [list(r) for r in self.route_pool.keys()]
            
            # A. Encode
            wcnf, route_map = encode_routes_as_wcnf(pool_list, self.instance.dist_matrix)
            # Dùng tên file unique để tránh conflict nếu chạy nhiều process
            wcnf_filename = f"iter_{self.instance.name}_{it}.wcnf"
            write_wcnf_to_file(wcnf, wcnf_filename)
            
            # B. Solve
            out = call_openwbo(wcnf_filename, timeout=TIMEOUT_SOLVER)
            # Clean temp file
            if os.path.exists(wcnf_filename): os.remove(wcnf_filename)
            
            # C. Decode
            vars_true = parse_openwbo_model(out)
            chosen_indices = chosen_routes_from_vars(vars_true, route_map)
            
            if not chosen_indices:
                print("   ⚠️ Solver failed or timeout.")
                break
                
            raw_solution = [pool_list[i-1] for i in chosen_indices]
            
            # D. Clean & Update
            current_solution = self._clean_solution(raw_solution)
            current_cost = total_distance(current_solution, self.instance.dist_matrix)
            
            print(f"   🔹 Cost (Valid): {current_cost:.2f}")
            
            if current_cost < best_overall_cost - 1e-4:
                print(f"   ✅ FOUND BETTER: {best_overall_cost:.2f} -> {current_cost:.2f}")
                best_overall_cost = current_cost
                best_solution_routes = current_solution
            else:
                print("   Creating new columns...")

            # E. Mutation (Pricing)
            # Dùng current_solution (đã clean) để sinh cột mới
            new_routes = self._generate_mutation(current_solution)
            merge_routes = self._generate_merge(current_solution)
            new_routes.extend(merge_routes)
            
            added_count = 0
            for nr in new_routes:
                t_nr = tuple(nr)
                if t_nr not in self.route_pool:
                    if len(self.route_pool) < MAX_POOL_SIZE:
                        self.route_pool[t_nr] = route_cost(nr, self.instance.dist_matrix)
                        added_count += 1
            
            print(f"   ✚ Added {added_count} new routes.")
            
            if added_count == 0:
                print("   🛑 Stagnation (No new columns). Stopping.")
                break

        # 3. FINISH
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "="*50)
        print(f"🏆 FINAL RESULT ({self.instance.name})")
        for i, r in enumerate(best_solution_routes, 1):
            c = route_cost(r, self.instance.dist_matrix)
            load = sum(self.instance.demands[n] for n in r)
            print(f"   Route {i}: {r} (Cost: {c:.2f}, Load: {load}/{self.instance.capacity})")
        
        # Log & Plot
        log_benchmark(
            self.instance, 
            best_solution_routes, 
            initial_cost, 
            best_overall_cost, 
            total_time, 
            final_iterations, 
            len(self.route_pool)
        )
        
        try:
            plot_solution(self.instance, best_solution_routes, best_overall_cost)
        except Exception as e:
            print(f"⚠️ Cannot plot: {e}")
"""
main_iterative.py
=================
Triển khai phương pháp MaxSAT-based Column Generation (Method 2).
Tích hợp:
- Ghi log Benchmark CSV.
- Post-processing để loại bỏ khách trùng lặp (Fix lỗi Set Covering).
"""

import os
import time
import random
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# Import các module đã có
from data_loader import from_coords_and_demands, Instance, read_vrplib, compute_distance_matrix
from heuristic import clarke_wright_savings, two_opt, total_distance, route_cost
from encoder import encode_routes_as_wcnf, write_wcnf_to_file
from solver_service import call_openwbo
from decoder import parse_openwbo_model, chosen_routes_from_vars

# --- CẤU HÌNH ---
TIMEOUT_SOLVER = 60   # Giây cho mỗi lần gọi solver
MAX_ITERATIONS = 50   # Số vòng lặp sinh cột tối đa
MAX_POOL_SIZE = 5000  # Giới hạn kích thước bể chứa tuyến đường


def generate_new_routes_mutation(current_best_routes: List[List[int]], 
                                 dist_matrix: np.ndarray, 
                                 demands: List[int], 
                                 capacity: int) -> List[List[int]]:
    """Sinh cột mới: 2-opt và Crossover."""
    new_candidates = []
    
    # 1. 2-opt improvement
    for r in current_best_routes:
        improved = two_opt(r, dist_matrix, max_iter=500)
        if route_cost(improved, dist_matrix) < route_cost(r, dist_matrix) - 1e-5:
            new_candidates.append(improved)

    # 2. Crossover
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
                
                def is_valid(route):
                    if len(route) <= 2: return False
                    if route[0] != 0 or route[-1] != 0: return False
                    return sum(demands[n] for n in route) <= capacity

                if child1[-1] != 0: child1.append(0)
                if child2[-1] != 0: child2.append(0)

                if is_valid(child1): new_candidates.append(two_opt(child1, dist_matrix))
                if is_valid(child2): new_candidates.append(two_opt(child2, dist_matrix))

    return new_candidates


def generate_merge_mutation(current_routes: List[List[int]], 
                            dist_matrix: np.ndarray, 
                            demands: List[int], 
                            capacity: int) -> List[List[int]]:
    """Gộp tuyến để giảm số lượng xe."""
    candidates = []
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
                # --- QUAN TRỌNG: Tránh thêm khách đã có ---
                if cust in new_route: continue 
                # ------------------------------------------

                best_pos = -1
                best_increase = float('inf')
                
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


def clean_solution(routes: List[List[int]], dist_matrix: np.ndarray) -> List[List[int]]:
    """
    Hậu xử lý: Loại bỏ các khách hàng bị trùng lặp (do mô hình Set Covering).
    Giữ lại lần xuất hiện đầu tiên, xóa các lần sau.
    """
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
    
    # Optional: Chạy 2-opt lại cho các tuyến vừa bị xóa bớt điểm để tối ưu lại
    final_routes = [two_opt(r, dist_matrix) for r in cleaned_routes]
    return final_routes


def plot_solution(instance: Instance, routes: List[List[int]], cost: float):
    plt.figure(figsize=(10, 8))
    
    # Vẽ Depot
    depot_x, depot_y = (0, 0)
    if instance.coords and len(instance.coords) > 0:
        depot_x, depot_y = instance.coords[instance.depot]
        
    plt.scatter(depot_x, depot_y, c='red', marker='s', s=150, zorder=10, label='Depot')
    
    # Vẽ Khách (Nếu tọa độ thật, nếu là Explicit Matrix thì tất cả là 0,0 sẽ chồng lên nhau)
    if instance.coords:
        xs = [c[0] for c in instance.coords[1:]]
        ys = [c[1] for c in instance.coords[1:]]
        plt.scatter(xs, ys, c='blue', s=40, zorder=5)
        for i in range(1, instance.n):
            if i < len(instance.coords):
                plt.text(instance.coords[i][0], instance.coords[i][1], str(i), fontsize=9)

    cmap = plt.get_cmap('tab20')
    for idx, r in enumerate(routes):
        route_coords = []
        for node in r:
            if node < len(instance.coords):
                route_coords.append(instance.coords[node])
            else:
                route_coords.append((0,0))
                
        r_xs, r_ys = zip(*route_coords)
        plt.plot(r_xs, r_ys, marker='.', linestyle='-', linewidth=2, color=cmap(idx % 20), label=f'Route {idx+1}')

    plt.title(f"Solution for {instance.name}\nTotal Cost: {cost:.2f} | Vehicles: {len(routes)}")
    plt.legend()
    plt.tight_layout()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(base_dir, 'results', 'plots', f"{instance.name}_solution.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def run_column_generation(instance: Instance):
    start_time = time.time()
    print(f"\n🚀 BẮT ĐẦU GIẢI: {instance.name} (n={instance.n-1}, Q={instance.capacity}, BKS={instance.bks})")
    
    # 1. Init
    initial_routes = clarke_wright_savings(instance.dist_matrix, instance.demands, instance.capacity)
    initial_routes = [two_opt(r, instance.dist_matrix) for r in initial_routes]
    initial_cost = total_distance(initial_routes, instance.dist_matrix)
    print(f"   [Init] Initial Heuristic Cost: {initial_cost:.2f}")
    
    route_pool = {tuple(r): route_cost(r, instance.dist_matrix) for r in initial_routes}
    best_overall_cost = float('inf')
    best_solution_routes = []
    final_iterations = 0

    # 2. Loop
    for it in range(1, MAX_ITERATIONS + 1):
        final_iterations = it
        print(f"\n🔄 ITERATION {it}/{MAX_ITERATIONS}")
        
        pool_list = [list(r) for r in route_pool.keys()]
        
        # Encode & Solve
        wcnf, route_map = encode_routes_as_wcnf(pool_list, instance.dist_matrix)
        wcnf_path = f"iter_{it}.wcnf"
        write_wcnf_to_file(wcnf, wcnf_path)
        
        out = call_openwbo(wcnf_path, timeout=TIMEOUT_SOLVER)
        if os.path.exists(wcnf_path): os.remove(wcnf_path) # Clean up immediately
        
        vars_true = parse_openwbo_model(out)
        chosen_indices = chosen_routes_from_vars(vars_true, route_map)
        
        if not chosen_indices:
            print("   ⚠️ Solver fail.")
            break
            
        raw_solution = [pool_list[i-1] for i in chosen_indices]
        
        # --- CLEAN DUPLICATES ---
        current_solution = clean_solution(raw_solution, instance.dist_matrix)
        # ------------------------
        
        current_cost = total_distance(current_solution, instance.dist_matrix)
        print(f"   🔹 Cost (Valid): {current_cost:.2f}")
        
        if current_cost < best_overall_cost - 1e-4:
            print(f"   ✅ KẾT QUẢ TỐT HƠN! ({best_overall_cost:.2f} -> {current_cost:.2f})")
            best_overall_cost = current_cost
            best_solution_routes = current_solution
        else:
            print("   Creating new columns...")

        # Mutation
        # Lưu ý: Dùng raw_solution (chưa clean) để lai ghép có thể tạo đa dạng tốt hơn
        # nhưng dùng current_solution (đã clean) sẽ an toàn hơn. Ta dùng current_solution.
        new_routes = generate_new_routes_mutation(current_solution, instance.dist_matrix, instance.demands, instance.capacity)
        merge_routes = generate_merge_mutation(current_solution, instance.dist_matrix, instance.demands, instance.capacity)
        new_routes.extend(merge_routes)
        
        added = 0
        for nr in new_routes:
            t_nr = tuple(nr)
            if t_nr not in route_pool:
                if len(route_pool) < MAX_POOL_SIZE:
                    route_pool[t_nr] = route_cost(nr, instance.dist_matrix)
                    added += 1
        print(f"   ✚ Added {added} routes.")
        
        if added == 0:
            print("   🛑 Stagnation.")
            break

    # 3. Finalize
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*50)
    print(f"🏆 KẾT QUẢ CUỐI CÙNG ({instance.name})")
    for i, r in enumerate(best_solution_routes, 1):
        c = route_cost(r, instance.dist_matrix)
        load = sum(instance.demands[n] for n in r)
        print(f"   Route {i}: {r} (Cost: {c:.2f}, Load: {load}/{instance.capacity})")
    
    # Metrics
    imp_percent = 0.0
    if initial_cost > 0: imp_percent = ((initial_cost - best_overall_cost)/initial_cost)*100
    
    gap_percent = 0.0
    if instance.bks > 0: gap_percent = ((best_overall_cost - instance.bks)/instance.bks)*100
    else: gap_percent = -1.0

    print("-" * 50)
    print(f"📊 BENCHMARK METRICS:")
    print(f"   BKS: {instance.bks} | Final: {best_overall_cost:.2f}")
    print(f"   Improvement: {imp_percent:.2f}%")
    print(f"   Gap: {gap_percent:.2f}%")
    print(f"   Time: {total_time:.2f}s")
    print("="*50)

    # CSV Log
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(base_dir, "results", "benchmark_log.csv")
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not os.path.isfile(csv_file):
            writer.writerow(["Instance", "n", "k", "Q", "BKS", "Init", "Final", "Imp%", "Gap%", "Time", "Iter", "Pool"])
        writer.writerow([instance.name, instance.n - 1, len(best_solution_routes), instance.capacity, instance.bks,
                         f"{initial_cost:.2f}", f"{best_overall_cost:.2f}", f"{imp_percent:.2f}", f"{gap_percent:.2f}",
                         f"{total_time:.2f}", final_iterations, len(route_pool)])
    
    try: plot_solution(instance, best_solution_routes, best_overall_cost)
    except: pass

if __name__ == "__main__":
    if len(sys.argv) > 1:
        vrp_file = sys.argv[1]
        if os.path.exists(vrp_file):
            run_column_generation(read_vrplib(vrp_file))
        else: print("File not found.")
    else:
        print("Usage: python main_iterative.py <file.vrp>")
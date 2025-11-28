"""
main_iterative.py
=================
Triển khai phương pháp MaxSAT-based Column Generation (Method 2).
Vòng lặp:
1. Khởi tạo Pool tuyến đường (Clarke-Wright).
2. Lặp:
   a. Giải MaxSAT (Master Problem) để chọn bộ tuyến tốt nhất hiện tại.
   b. Phân tích nghiệm, tìm cơ hội cải tiến.
   c. Sinh tuyến mới (Pricing/Mutation) thêm vào Pool.
   d. Nếu không cải thiện được nữa -> Dừng.
"""

import os
import time
import random
import sys
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
TIMEOUT_SOLVER = 30   # Giây cho mỗi lần gọi solver
MAX_ITERATIONS = 20   # Số vòng lặp sinh cột tối đa (Tăng lên để tìm kiếm sâu hơn)
MAX_POOL_SIZE = 2000  # Giới hạn kích thước bể chứa tuyến đường


def generate_new_routes_mutation(current_best_routes: List[List[int]], 
                                 dist_matrix: np.ndarray, 
                                 demands: List[int], 
                                 capacity: int) -> List[List[int]]:
    """
    Sinh cột mới bằng cách 'đột biến' các tuyến đường tốt nhất hiện tại.
    Chiến lược: Lấy 2 tuyến, thử tráo đổi khách hàng (Swap) hoặc gộp.
    """
    new_candidates = []
    
    # Chiến lược 1: Thử chạy 2-opt kỹ hơn (nếu chưa tối ưu)
    for r in current_best_routes:
        improved = two_opt(r, dist_matrix, max_iter=500)
        # Chỉ thêm nếu thực sự cải thiện đáng kể để tránh trùng lặp
        if route_cost(improved, dist_matrix) < route_cost(r, dist_matrix) - 1e-5:
            new_candidates.append(improved)

    # Chiến lược 2: Destroy & Repair đơn giản (Lai ghép 2 tuyến)
    # Lấy ngẫu nhiên các cặp tuyến để lai ghép
    n_routes = len(current_best_routes)
    if n_routes >= 2:
        # Số lần thử lai ghép tùy thuộc vào số lượng tuyến đang có
        num_trials = min(10, n_routes * 2)
        
        for _ in range(num_trials):
            idx1, idx2 = np.random.choice(n_routes, 2, replace=False)
            r1, r2 = current_best_routes[idx1], current_best_routes[idx2]
            
            # Cắt đôi tuyến r1 và r2 tại điểm ngẫu nhiên (trừ điểm đầu/cuối là depot)
            if len(r1) > 3 and len(r2) > 3:
                cut1 = random.randint(1, len(r1) - 2)
                cut2 = random.randint(1, len(r2) - 2)
                
                # Tạo tuyến con mới: Đầu r1 + Đuôi r2
                child1 = r1[:cut1] + r2[cut2:]
                # Đầu r2 + Đuôi r1
                child2 = r2[:cut2] + r1[cut1:]
                
                # Hàm check tải trọng nội bộ
                def is_valid(route):
                    # Route phải có ít nhất 1 khách (len > 2 vì có 2 depot)
                    if len(route) <= 2: return False
                    # Phải bắt đầu và kết thúc bằng 0
                    if route[0] != 0 or route[-1] != 0: return False
                    
                    load = sum(demands[n] for n in route)
                    return load <= capacity

                # Đảm bảo format đúng (kết thúc bằng 0)
                if child1[-1] != 0: child1.append(0)
                if child2[-1] != 0: child2.append(0)

                # Nếu hợp lệ thì tối ưu hóa ngay bằng 2-opt trước khi thêm
                if is_valid(child1): new_candidates.append(two_opt(child1, dist_matrix))
                if is_valid(child2): new_candidates.append(two_opt(child2, dist_matrix))

    return new_candidates


def plot_solution(instance: Instance, routes: List[List[int]], cost: float):
    """
    Vẽ biểu đồ kết quả và lưu vào file ảnh.
    """
    plt.figure(figsize=(10, 8))
    
    # 1. Vẽ Depot
    if instance.coords:
        depot_x, depot_y = instance.coords[instance.depot]
        plt.scatter(depot_x, depot_y, c='red', marker='s', s=150, zorder=10, label='Depot')
        
        # 2. Vẽ Khách hàng
        coords = instance.coords
        # Khách hàng từ index 1 trở đi
        xs = [c[0] for c in coords[1:]]
        ys = [c[1] for c in coords[1:]]
        plt.scatter(xs, ys, c='blue', s=40, zorder=5)
        
        # Đánh số thứ tự khách hàng
        for i in range(1, instance.n):
            plt.text(coords[i][0], coords[i][1] + 0.5, str(i), fontsize=9, ha='center')
            
        # 3. Vẽ Tuyến đường
        # Dùng colormap để mỗi tuyến 1 màu
        cmap = plt.get_cmap('tab20')
        
        for idx, r in enumerate(routes):
            route_coords = [coords[node] for node in r]
            r_xs, r_ys = zip(*route_coords)
            
            # Vẽ đường nối
            plt.plot(r_xs, r_ys, marker='.', linestyle='-', linewidth=2, 
                     color=cmap(idx % 20), label=f'Route {idx+1}', alpha=0.7)
            
            # Vẽ mũi tên chỉ hướng (tùy chọn, vẽ ở giữa tuyến)
            mid = len(r) // 2
            if mid < len(r) - 1:
                p1 = coords[r[mid]]
                p2 = coords[r[mid+1]]
                plt.arrow(p1[0], p1[1], (p2[0]-p1[0])*0.5, (p2[1]-p1[1])*0.5, 
                          head_width=0.5, color=cmap(idx % 20))

    plt.title(f"Solution for {instance.name}\nTotal Cost: {cost:.2f} | Vehicles: {len(routes)}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Lưu ảnh
    # Tạo thư mục nếu chưa có
    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'plots')
    os.makedirs(result_dir, exist_ok=True)
    
    save_path = os.path.join(result_dir, f"{instance.name}_solution.png")
    plt.savefig(save_path)
    print(f"\n📊 Đã lưu biểu đồ trực quan tại: {save_path}")
    plt.close() # Đóng figure để giải phóng bộ nhớ


def run_column_generation(instance: Instance):
    print(f"\n🚀 BẮT ĐẦU GIẢI: {instance.name} (n={instance.n-1}, Q={instance.capacity})")
    
    # 1. KHỞI TẠO (Initialization)
    # Dùng heuristic để tạo tập cột ban đầu
    initial_routes = clarke_wright_savings(instance.dist_matrix, instance.demands, instance.capacity)
    initial_routes = [two_opt(r, instance.dist_matrix) for r in initial_routes]
    
    # Pool chứa tất cả các tuyến đường duy nhất đã tìm thấy (chuyển sang tuple để hash)
    # Key: Tuple tuyến đường, Value: Cost
    route_pool = {tuple(r): route_cost(r, instance.dist_matrix) for r in initial_routes}
    
    best_overall_cost = float('inf')
    best_solution_routes = []

    print(f"   [Init] Pool size: {len(route_pool)}")

    # 2. VÒNG LẶP (Iteration Loop)
    for it in range(1, MAX_ITERATIONS + 1):
        print(f"\n🔄 ITERATION {it}/{MAX_ITERATIONS}")
        
        # Chuyển pool thành list để encode
        pool_list = [list(r) for r in route_pool.keys()]
        
        # a. Encode sang MaxSAT (Master Problem)
        wcnf, route_map = encode_routes_as_wcnf(pool_list, instance.dist_matrix)
        wcnf_filename = f"iter_{it}.wcnf"
        wcnf_path = os.path.join(os.getcwd(), wcnf_filename)
        write_wcnf_to_file(wcnf, wcnf_path)
        
        # b. Gọi Solver
        out = call_openwbo(wcnf_path, timeout=TIMEOUT_SOLVER)
        
        # c. Giải mã kết quả
        vars_true = parse_openwbo_model(out)
        chosen_indices = chosen_routes_from_vars(vars_true, route_map)
        
        if not chosen_indices:
            print("   ⚠️ Solver không tìm thấy nghiệm (hoặc timeout).")
            # Nếu timeout, có thể do bài toán quá lớn, ta giữ lại kết quả tốt nhất trước đó
            break
            
        current_solution = [pool_list[i-1] for i in chosen_indices]
        current_cost = total_distance(current_solution, instance.dist_matrix)
        
        print(f"   🔹 Cost vòng này: {current_cost:.2f}")
        
        # Cập nhật kết quả tốt nhất (Best so far)
        # Lưu ý: Do MaxSAT tính xấp xỉ số nguyên nên ta cho phép sai số nhỏ float
        if current_cost < best_overall_cost - 1e-4:
            print(f"   ✅ TÌM THẤY KẾT QUẢ TỐT HƠN! ({best_overall_cost:.2f} -> {current_cost:.2f})")
            best_overall_cost = current_cost
            best_solution_routes = current_solution
        else:
            print("   Creating new columns (routes) to improve...")

        # d. Sinh cột mới (Column Generation / Pricing)
        new_routes = generate_new_routes_mutation(current_solution, 
                                                  instance.dist_matrix, 
                                                  instance.demands, 
                                                  instance.capacity)
        
        # Thêm vào Pool
        added_count = 0
        for nr in new_routes:
            t_nr = tuple(nr)
            if t_nr not in route_pool:
                # Kiểm tra giới hạn Pool để tránh tràn RAM
                if len(route_pool) < MAX_POOL_SIZE:
                    route_pool[t_nr] = route_cost(nr, instance.dist_matrix)
                    added_count += 1
        
        print(f"   ✚ Đã thêm {added_count} tuyến đường mới vào Pool.")
        
        # Dọn dẹp file tạm
        if os.path.exists(wcnf_path):
            os.remove(wcnf_path)
            
        # Điều kiện dừng sớm: Nếu không sinh được gì mới
        if added_count == 0:
            print("   🛑 Không sinh thêm được tuyến mới nào. Dừng thuật toán.")
            break

    # 3. KẾT THÚC
    print("\n" + "="*50)
    print(f"🏆 KẾT QUẢ CUỐI CÙNG ({instance.name})")
    print(f"   Tổng chi phí: {best_overall_cost:.4f}")
    print("   Các tuyến đường:")
    for i, r in enumerate(best_solution_routes, 1):
        c = route_cost(r, instance.dist_matrix)
        load = sum(instance.demands[n] for n in r)
        print(f"     Route {i}: {r} (Cost: {c:.2f}, Load: {load}/{instance.capacity})")
    print("="*50)

    # 4. VẼ BIỂU ĐỒ
    try:
        plot_solution(instance, best_solution_routes, best_overall_cost)
    except Exception as e:
        print(f"⚠️ Không thể vẽ biểu đồ: {e}")


if __name__ == "__main__":
    # HỖ TRỢ CHẠY TỪ DÒNG LỆNH
    # Cách dùng: python main_iterative.py ../data/A/A-n32-k5.vrp
    
    if len(sys.argv) > 1:
        vrp_file = sys.argv[1]
        if not os.path.exists(vrp_file):
            print(f"❌ File không tồn tại: {vrp_file}")
            sys.exit(1)
            
        print(f"📂 Đang đọc file: {vrp_file}")
        try:
            # Đọc instance từ file .vrp
            instance = read_vrplib(vrp_file)
            run_column_generation(instance)
        except Exception as e:
            print(f"❌ Lỗi khi chạy thực nghiệm: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        print("⚠️ Không có file input. Chạy chế độ DEMO với dữ liệu giả lập...")
        print("💡 Gợi ý: python main_iterative.py <path_to_vrp_file>")
        
        # DỮ LIỆU DEMO
        coords = [(0,0), (10,0), (0,10), (5,5), (2,8), (8,2), (10,10), (1,1), (9,9), (3,3), (7,7)]
        demands = [0, 2, 3, 1, 5, 2, 4, 1, 3, 2, 4] 
        capacity = 10 
        
        instance = Instance(
            name="demo_iterative",
            n=len(coords),
            depot=0,
            coords=coords,
            demands=demands,
            capacity=capacity,
            dist_matrix=None 
        )
        instance.dist_matrix = compute_distance_matrix(coords)

        run_column_generation(instance)
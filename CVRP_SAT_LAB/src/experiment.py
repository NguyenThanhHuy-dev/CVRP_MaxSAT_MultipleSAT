"""
experiment.py
==============
Chạy toàn bộ pipeline CVRP-SAT đơn giản:
1. Load dữ liệu (tạo ví dụ nhỏ)
2. Sinh nghiệm bằng Clarke–Wright + 2-opt
3. Mã hóa thành WCNF
4. Gọi solver (Open-WBO)
5. Giải mã nghiệm và tính chi phí cuối cùng
"""

import os
import numpy as np

from data_loader import from_coords_and_demands
from heuristic import clarke_wright_savings, two_opt, total_distance
from encoder import encode_routes_as_wcnf, write_wcnf_to_file
from solver_service import call_openwbo
from decoder import parse_openwbo_model, chosen_routes_from_vars


def run_demo():
    print("=" * 50)
    print("🚚  DEMO: CVRP + MaxSAT Solver Integration")
    print("=" * 50)

    # --- 1️⃣ DỮ LIỆU VÍ DỤ ---
    coords = [
        (0.0, 0.0),  # depot
        (1.0, 2.0),
        (2.0, 1.0),
        (2.5, 4.0),
        (0.5, 3.0),
    ]
    demands = [0, 5, 8, 4, 6]
    capacity = 15
    inst = from_coords_and_demands("demo", coords, demands, capacity)

    # --- 2️⃣ CHẠY HEURISTIC ---
    routes = clarke_wright_savings(inst.dist_matrix, inst.demands, inst.capacity)
    routes = [two_opt(r, inst.dist_matrix) for r in routes]

    print("\n✅ Candidate routes (after CW + 2-opt):")
    for r in routes:
        print(" ", r)
    print("Initial total cost:", total_distance(routes, inst.dist_matrix))

    # --- 3️⃣ ENCODE WCNF ---
    wcnf, route_map = encode_routes_as_wcnf(routes, inst.dist_matrix)
    wcnf_path = os.path.join(os.getcwd(), "temp.wcnf")
    write_wcnf_to_file(wcnf, wcnf_path)
    print(f"\n🧩 WCNF file written to: {wcnf_path}")

    # --- 4️⃣ GỌI SOLVER ---
    print("\n⚙️  Running Open-WBO solver...")
    out = call_openwbo(wcnf_path, timeout=30)
    print("\n--- Solver Output (truncated) ---")
    print("\n".join(out.splitlines()[:20]))

    # --- 5️⃣ GIẢI MÃ NGHIỆM ---
    vars_true = parse_openwbo_model(out)
    chosen = chosen_routes_from_vars(vars_true, route_map)
    chosen_routes = [routes[i - 1] for i in chosen]

    print("\n✅ Chosen route indices:", chosen)
    print("Chosen routes:")
    for r in chosen_routes:
        print(" ", r)
    print("Final cost:", total_distance(chosen_routes, inst.dist_matrix))

    print("\n🎉 Done.")


if __name__ == "__main__":
    run_demo()

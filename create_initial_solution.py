import numpy as np
from verypy.classic_heuristics.parallel_savings import parallel_savings_init
from verypy.local_search.intra_route_operators import do_2opt_move
from verypy.util import totald

# --- 1. DỮ LIỆU ---
D = np.array([
    [0, 10, 12, 11, 8],
    [10, 0, 5, 13, 10],
    [12, 5, 0, 7, 9],
    [11, 13, 7, 0, 6],
    [8, 10, 9, 6, 0],
])
d = [0, 5, 8, 4, 6]   # list
C = 15

print("Đã nạp dữ liệu: 1 Kho, 4 Khách hàng, Sức chứa xe =", C)
print("-" * 30)

try:
    # Clarke–Wright Savings
    cw_routes = parallel_savings_init(D, d, C, minimize_K=True)
    cw_cost = totald(cw_routes, D)

    print("Giải pháp Clarke-Wright (CW) ban đầu:")
    print("Các tuyến đường:", cw_routes)
    print("Chi phí (quãng đường):", cw_cost)
    print("-" * 30)

    # 2-Opt cải thiện
    improved_routes = []
    for r in cw_routes:
        improved_r, _, _ = do_2opt_move(r, D, max_iterations=100)
        improved_routes.append(improved_r)

    final_cost = totald(improved_routes, D)

    print("Giải pháp sau khi cải thiện 2-Opt:")
    print("Các tuyến đường:", improved_routes)
    print("Chi phí cuối cùng:", final_cost)
    print("-" * 30)

except ValueError as e:
    print(f"Lỗi: {e}")

import time
import random
from pysat.solvers import Glucose3
n_vars = 200
n_clauses = 10000
seed = 42
random.seed(seed)


print(f"Đang tạo {n_clauses} mệnh đề ngẫu nhiên với {n_vars} biến...")
base_clauses = []
for _ in range(n_clauses):
    # Tạo ngẫu nhiên một mệnh đề gồm 3 biến (3-SAT)
    clause = [random.randint(1, n_vars) * random.choice([-1, 1]) for _ in range(3)]
    base_clauses.append(clause)

# Số lượng bài toán con cần giải
iterations = 50 

print("-" * 50)

# --- CÁCH 1: NON-INCREMENTAL (CÁCH THÔNG THƯỜNG) ---
print(">>> Bắt đầu chạy Non-Incremental...")
start_time = time.time()

for i in range(iterations):
    solver = Glucose3()
    
    solver.append_formula(base_clauses)
    
    solver.add_clause([1, -2, (i % n_vars) + 1])
    
    solver.solve()
    solver.delete()
    
end_time = time.time()
non_incremental_time = end_time - start_time
print(f"Thời gian Non-Incremental: {non_incremental_time:.4f} giây")

print("-" * 50)

# --- CÁCH 2: INCREMENTAL (CÁCH TỐI ƯU) ---
print(">>> Bắt đầu chạy Incremental...")
start_time = time.time()

# 1. Khởi tạo bộ giải MỘT LẦN DUY NHẤT
solver = Glucose3()
solver.append_formula(base_clauses) # Nạp kiến thức nền một lần

for i in range(iterations):
    # 2. Giải với GIẢ ĐỊNH (Assumptions)
    # Assumptions là các biến tạm thời coi là True/False chỉ trong lần giải này.
    # Sau khi giải xong, bộ giải tự động quay về trạng thái cũ, không cần reset.
    assumptions = [1, -2, (i % n_vars) + 1]
    
    solver.solve(assumptions=assumptions)
    
    # Bộ giải vẫn sống và giữ lại các 'learned clauses' (kinh nghiệm) từ các vòng lặp trước!

solver.delete()
end_time = time.time()
inc_time = end_time - start_time
print(f"Thời gian chạy Incremental:     {inc_time:.4f} giây")

# --- KẾT QUẢ ---
print("-" * 50)
if inc_time > 0:
    speedup = non_incremental_time / inc_time
    print(f"Kết luận: Incremental nhanh hơn gấp {speedup:.1f} lần!")
else:
    print("Thời gian quá ngắn để so sánh.")
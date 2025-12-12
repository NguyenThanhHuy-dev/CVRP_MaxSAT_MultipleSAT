import csv
import os
import datetime

def validate_solution(instance, routes):
    """Kiểm tra tính hợp lệ của nghiệm: Capacity và Coverage."""
    # 1. Check Capacity
    for i, r in enumerate(routes):
        load = sum(instance.demands[n] for n in r)
        if load > instance.capacity:
            return False, f"Route {i+1} overloaded: {load}/{instance.capacity}"
            
    # 2. Check Coverage (All customers visited exactly once)
    visited = set()
    for r in routes:
        for n in r:
            if n != 0: 
                if n in visited:
                    return False, f"Customer {n} visited multiple times"
                visited.add(n)
            
    if len(visited) != instance.n - 1: # Trừ depot (node 0)
        return False, f"Missing customers. Visited: {len(visited)}/{instance.n - 1}"
        
    return True, "Valid"

def log_benchmark(instance, best_routes, initial_cost, final_cost, time_taken, iterations, pool_size, method_name="HybridLNS"):
    # Validate trước khi log
    is_valid, message = validate_solution(instance, best_routes)
    
    if not is_valid:
        print(f"❌ CRITICAL ERROR: Solution Invalid - {message}")
        # Đánh dấu phạt nặng để khi tính trung bình sẽ nhận ra ngay
        final_cost = float('inf')
        gap_percent = 999.99 
    else:
        # Tính Gap
        if instance.bks > 0:
            gap_percent = ((final_cost - instance.bks) / instance.bks) * 100
        else:
            gap_percent = 0.0

    imp_percent = 0.0
    if initial_cost > 0:
        imp_percent = ((initial_cost - final_cost) / initial_cost) * 100

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Tạo nội dung Log
    log_content = []
    log_content.append("=" * 50)
    log_content.append(f"   BENCHMARK METRICS ({instance.name}) - {timestamp}")
    log_content.append(f"   Method:        {method_name}")
    log_content.append(f"   Validity:      {'✅ VALID' if is_valid else '❌ INVALID'}")
    log_content.append(f"   BKS (Optimal): {instance.bks}")
    log_content.append(f"   Initial Cost:  {initial_cost:.2f}")
    log_content.append(f"   Final Cost:    {final_cost:.2f}")
    log_content.append(f"   Improvement:   {imp_percent:.2f}%")
    log_content.append(f"   Gap to BKS:    {gap_percent:.2f}%")
    log_content.append(f"   Time (s):      {time_taken:.2f}s")
    log_content.append("-" * 50)
    
    # In ra màn hình
    print("\n".join(log_content))

    # Định nghĩa đường dẫn
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, "results")
    benchmark_file = os.path.join(results_dir, "benchmark_log.csv")
    
    os.makedirs(results_dir, exist_ok=True)

    # Ghi vào CSV tổng hợp
    file_exists = os.path.isfile(benchmark_file)
    try:
        with open(benchmark_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Instance", "Method", "Valid", "n", "k_vehicles", "Q", "BKS", 
                                 "Final_Cost", "Gap_%", "Time_s", "Iterations"])
            
            writer.writerow([
                timestamp,
                instance.name, 
                method_name,
                is_valid,
                instance.n - 1, 
                len(best_routes), 
                instance.capacity, 
                instance.bks, 
                f"{final_cost:.2f}", 
                f"{gap_percent:.2f}", 
                f"{time_taken:.2f}",
                iterations
            ])
    except Exception as e:
        print(f"⚠️ Lỗi ghi benchmark csv: {e}")
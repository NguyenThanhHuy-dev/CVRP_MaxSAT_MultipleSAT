import csv
import os

def log_benchmark(instance, best_routes, initial_cost, final_cost, time_taken, iterations, pool_size):
    """
    Tính toán Gap, Improvement và ghi log vào file CSV chung.
    """
    # 1. Tính Metrics
    imp_percent = 0.0
    if initial_cost > 0:
        imp_percent = ((initial_cost - final_cost) / initial_cost) * 100
    
    gap_percent = 0.0
    if instance.bks > 0:
        gap_percent = ((final_cost - instance.bks) / instance.bks) * 100
    else:
        gap_percent = -1.0 # Không có BKS

    # 2. In ra màn hình console
    print("-" * 50)
    print(f"📊 BENCHMARK METRICS ({instance.name}):")
    print(f"   BKS (Optimal): {instance.bks}")
    print(f"   Initial Cost:  {initial_cost:.2f}")
    print(f"   Final Cost:    {final_cost:.2f}")
    print(f"   Improvement:   {imp_percent:.2f}%")
    print(f"   Gap to BKS:    {gap_percent:.2f}%")
    print(f"   Time (s):      {time_taken:.2f}s")
    print(f"   Iterations:    {iterations}")
    print(f"   Pool Size:     {pool_size}")
    print("=" * 50)

    # 3. Ghi vào file CSV
    # Đường dẫn: CVRP_SAT_LAB/results/benchmark_log.csv
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_file = os.path.join(base_dir, "results", "benchmark_log.csv")
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    file_exists = os.path.isfile(csv_file)
    
    try:
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            # Ghi Header nếu file mới tạo
            if not file_exists:
                writer.writerow(["Instance", "n", "k_vehicles", "Q", "BKS", 
                                 "Init_Cost", "Final_Cost", "Improvement_%", "Gap_%", 
                                 "Time_s", "Iterations", "Pool_Size", "Method"])
            
            # Ghi Data Row
            writer.writerow([
                instance.name, 
                instance.n - 1, 
                len(best_routes), 
                instance.capacity, 
                instance.bks, 
                f"{initial_cost:.2f}", 
                f"{final_cost:.2f}", 
                f"{imp_percent:.2f}", 
                f"{gap_percent:.2f}", 
                f"{time_taken:.2f}",
                iterations,
                pool_size,
                "ColumnGeneration" # Đánh dấu phương pháp
            ])
        print(f"📝 Đã ghi kết quả so sánh vào: {csv_file}")
    except Exception as e:
        print(f"⚠️ Lỗi khi ghi log CSV: {e}")
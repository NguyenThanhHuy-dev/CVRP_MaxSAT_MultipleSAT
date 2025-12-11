import csv
import os
import datetime

def log_benchmark(instance, best_routes, initial_cost, final_cost, time_taken, iterations, pool_size, method_name="HybridLNS"):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    results_dir = os.path.join(base_dir, "results")
    benchmark_file = os.path.join(results_dir, "benchmark_log.csv")
    
    csv_detail_dir = os.path.join(results_dir, "csv")
    log_txt_dir = os.path.join(results_dir, "log")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(csv_detail_dir, exist_ok=True)
    os.makedirs(log_txt_dir, exist_ok=True)

    imp_percent = 0.0
    if initial_cost > 0:
        imp_percent = ((initial_cost - final_cost) / initial_cost) * 100
    
    gap_percent = 0.0
    if instance.bks > 0:
        gap_percent = ((final_cost - instance.bks) / instance.bks) * 100
    else:
        gap_percent = -1.0

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_content = []
    log_content.append("=" * 50)
    log_content.append(f"   BENCHMARK METRICS ({instance.name}) - {timestamp}")
    log_content.append(f"   Method:        {method_name}")
    log_content.append(f"   BKS (Optimal): {instance.bks}")
    log_content.append(f"   Initial Cost:  {initial_cost:.2f}")
    log_content.append(f"   Final Cost:    {final_cost:.2f}")
    log_content.append(f"   Improvement:   {imp_percent:.2f}%")
    log_content.append(f"   Gap to BKS:    {gap_percent:.2f}%")
    log_content.append(f"   Time (s):      {time_taken:.2f}s")
    log_content.append(f"   Iterations:    {iterations}")
    log_content.append(f"   Pool Size:     {pool_size}")
    log_content.append("-" * 50)
    log_content.append("    DETAILED ROUTES:")
    
    for i, route in enumerate(best_routes, 1):
        load = sum(instance.demands[n] for n in route)
        route_str = f"   Route {i} (Load {load}/{instance.capacity}): {route}"
        log_content.append(route_str)
    log_content.append("=" * 50)

    print("\n".join(log_content))

    file_exists = os.path.isfile(benchmark_file)
    try:
        with open(benchmark_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Instance", "Method", "n", "k_vehicles", "Q", "BKS", 
                                 "Init_Cost", "Final_Cost", "Improvement_%", "Gap_%", 
                                 "Time_s", "Iterations"])
            
            writer.writerow([
                timestamp,
                instance.name, 
                method_name,
                instance.n - 1, 
                len(best_routes), 
                instance.capacity, 
                instance.bks, 
                f"{initial_cost:.2f}", 
                f"{final_cost:.2f}", 
                f"{imp_percent:.2f}", 
                f"{gap_percent:.2f}", 
                f"{time_taken:.2f}",
                iterations
            ])
        print(f"📝 Đã ghi kết quả tổng hợp vào: {benchmark_file}")
    except Exception as e:
        print(f"⚠️ Lỗi ghi benchmark csv: {e}")

    detail_csv_path = os.path.join(csv_detail_dir, f"{instance.name}_solution.csv")
    try:
        with open(detail_csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Route_ID", "Load", "Capacity", "Path_Sequence"])
            for i, route in enumerate(best_routes, 1):
                load = sum(instance.demands[n] for n in route)
                # Chuyển list [0, 5, 2, 0] thành chuỗi "0-5-2-0"
                path_str = "-".join(map(str, route))
                writer.writerow([i, load, instance.capacity, path_str])
        print(f"📝 Đã lưu chi tiết lộ trình vào: {detail_csv_path}")
    except Exception as e:
        print(f"⚠️ Lỗi ghi detail csv: {e}")


    log_txt_path = os.path.join(log_txt_dir, f"{instance.name}.log")
    try:
        with open(log_txt_path, mode='w', encoding='utf-8') as f:
            f.write("\n".join(log_content))
        print(f"📝 Đã lưu log chạy vào: {log_txt_path}")
    except Exception as e:
        print(f"⚠️ Lỗi ghi log text: {e}")
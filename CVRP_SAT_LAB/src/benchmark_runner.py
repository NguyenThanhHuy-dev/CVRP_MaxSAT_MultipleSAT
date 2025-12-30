import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

from data_loader import read_vrplib
from strategies.hybrid_lns import HybridLNSStrategy
# from strategies.exact_multishot import ExactMultiShotStrategy # Nếu muốn test cả cái này

def run_stability_test(instance_path, runs=5):
    """Chạy thuật toán nhiều lần để đánh giá độ ổn định."""
    
    instance = read_vrplib(instance_path)
    print(f"🔬 STARTING STABILITY TEST: {instance.name} (Runs: {runs})")
    print(f"   BKS: {instance.bks}")
    
    costs = []
    times = []
    gaps = []
    
    for i in range(runs):
        print(f"\n--- Run {i+1}/{runs} ---")
        start_time = time.time()
        
        solver = HybridLNSStrategy(instance)
        
        cost, _ = solver.solve() 
        
        elapsed = time.time() - start_time
        
        gap = 0.0
        if instance.bks > 0:
            gap = ((cost - instance.bks) / instance.bks) * 100
            
        costs.append(cost)
        times.append(elapsed)
        gaps.append(gap)
        
    avg_cost = np.mean(costs)
    best_cost = np.min(costs)
    std_cost = np.std(costs)
    avg_gap = np.mean(gaps)
    best_gap = np.min(gaps)
    avg_time = np.mean(times)
    
    print("\n" + "="*50)
    print(f"📊 SUMMARY REPORT FOR {instance.name}")
    print(f"   Runs: {runs}")
    print(f"   Best Cost: {best_cost:.2f} (Gap: {best_gap:.2f}%)")
    print(f"   Avg Cost:  {avg_cost:.2f} (Gap: {avg_gap:.2f}%)")
    print(f"   Std Dev:   {std_cost:.2f}")
    print(f"   Avg Time:  {avg_time:.2f}s")
    print("="*50)
    
    summary = {
        "Instance": instance.name,
        "Runs": runs,
        "BKS": instance.bks,
        "Best_Cost": best_cost,
        "Avg_Cost": avg_cost,
        "Best_Gap": best_gap,
        "Avg_Gap": avg_gap,
        "Std_Dev": std_cost,
        "Avg_Time": avg_time
    }
    
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CVRP Stability Benchmark Runner")
    parser.add_argument("instances", nargs='+', help="Path to .vrp files")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per instance")
    
    args = parser.parse_args()
    
    results = []
    for path in args.instances:
        res = run_stability_test(path, runs=args.runs)
        results.append(res)
        
    df = pd.DataFrame(results)
    df.to_csv("../results/stability_report.csv", index=False)
    print("\n   Saved stability report to results/stability_report.csv")
    print(df)
import argparse
import os
import sys

from data_loader import read_vrplib
from strategies.column_generation import ColumnGenerationStrategy
from strategies.hybrid_lns import HybridLNSStrategy
from strategies.exact_multishot import ExactMultiShotStrategy

def main():
    parser = argparse.ArgumentParser(description="CVRP Solver using SAT")
    parser.add_argument("instance_path", type=str, help="Path to VRPLIB file")
    # Thêm 'exact' vào choices
    parser.add_argument("--method", type=str, choices=["lns", "colgen", "exact"], default="lns") 
    args = parser.parse_args()

    instance = read_vrplib(args.instance_path)

    if args.method == "lns":
        solver = HybridLNSStrategy(instance)
        solver.solve()
    elif args.method == "colgen":
        print("Running Column Generation (Placeholder)...")
        # solver = ColumnGenerationStrategy(instance)
        # solver.solve()
    elif args.method == "exact":  # <--- Logic mới
        solver = ExactMultiShotStrategy(instance)
        solver.solve()

if __name__ == "__main__":
    main()
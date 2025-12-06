import argparse
import os
import sys

from data_loader import read_vrplib
from strategies.column_generation import ColumnGenerationStrategy
from strategies.hybrid_lns import HybridLNSStrategy

def main():
    parser = argparse.ArgumentParser(description="CVRP MaxSAT Solver Framework")
    parser.add_argument("file_path", type=str, help="Path to .vrp instance file")
    parser.add_argument("--method", type=str, default="cg", choices=["cg", "lns"], 
                        help="Choose solver strategy: 'cg' (Column Gen) or 'lns' (Hybrid LNS)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"❌ Error: File not found: {args.file_path}")
        sys.exit(1)
        
    # Load Data
    print(f"📂 Loading instance: {args.file_path}")
    instance = read_vrplib(args.file_path)
    
    # Select Strategy
    if args.method == "cg":
        solver = ColumnGenerationStrategy(instance)
    elif args.method == "lns":
        solver = HybridLNSStrategy(instance)
    
    # Run
    solver.solve()

if __name__ == "__main__":
    main()
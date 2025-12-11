from typing import List
import re


def parse_openwbo_model(output: str) -> List[int]:

    vars_true = []
    for line in output.splitlines():
        line = line.strip()
        if line.startswith('v') or line.startswith('V'):
            parts = line[1:].strip().split()
            for p in parts:
                iv = int(p)
                if iv > 0:
                    vars_true.append(iv)
    return vars_true


def chosen_routes_from_vars(vars_true: List[int], route_map: dict) -> List[int]:
    chosen = [ridx for ridx, var in route_map.items() if var in vars_true]
    return chosen
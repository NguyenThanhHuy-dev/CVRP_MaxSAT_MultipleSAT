"""
Decoder to extract chosen route variables from solver textual output.
Supports Open-WBO output (lines starting with 'v').
"""
from typing import List
import re


def parse_openwbo_model(output: str) -> List[int]:
    """
    Parse model from open-wbo output, return list of true variable ids.
    """
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
    """
    Given list of true vars and mapping route_idx -> var, return list of route indices chosen.
    """
    chosen = [ridx for ridx, var in route_map.items() if var in vars_true]
    return chosen
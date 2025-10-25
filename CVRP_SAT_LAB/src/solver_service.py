"""
Wrapper to call external solvers (open-wbo, clasp) and parse output.
"""
import subprocess
from typing import Tuple
import os


def call_openwbo(wcnf_path: str, timeout: int = 60) -> str:
    OPENWBO = os.environ.get('OPENWBO_PATH', 'open-wbo')
    cmd = [OPENWBO, wcnf_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.stdout + '\n' + proc.stderr
    except subprocess.TimeoutExpired:
        return 'TIMEOUT'


def call_clasp(cnf_path: str, timeout: int = 60) -> str:
    CLASP = os.environ.get('CLASP_PATH', 'clasp')
    cmd = [CLASP, cnf_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.stdout + '\n' + proc.stderr
    except subprocess.TimeoutExpired:
        return 'TIMEOUT'
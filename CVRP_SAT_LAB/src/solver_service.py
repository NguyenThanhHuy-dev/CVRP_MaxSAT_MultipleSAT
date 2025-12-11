"""
Wrapper to call external solvers (open-wbo, clasp) and parse output.
"""
import subprocess
import os

def call_openwbo(wcnf_path: str, timeout: int = 60) -> str:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    OPENWBO_EXEC = os.path.join(BASE_DIR, 'solvers', 'open-wbo_bin')
    
    if not os.path.isfile(OPENWBO_EXEC):
        return f"ERROR: Solver binary not found at {OPENWBO_EXEC}"


    cmd = [OPENWBO_EXEC, wcnf_path, f"-cpu-lim={timeout}"]
    
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        
        
        return proc.stdout 
        
    except subprocess.TimeoutExpired:
        print(f"   ⚠️ Python killed solver process (Hard Timeout after {timeout+5}s).")
        return 'TIMEOUT'
    except Exception as e:
        return f"ERROR: Execution failed - {e}"


def call_clasp(cnf_path: str, timeout: int = 60) -> str:
    CLASP = os.environ.get('CLASP_PATH', 'clasp')
    cmd = [CLASP, cnf_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.stdout + '\n' + proc.stderr
    except subprocess.TimeoutExpired:
        return 'TIMEOUT'
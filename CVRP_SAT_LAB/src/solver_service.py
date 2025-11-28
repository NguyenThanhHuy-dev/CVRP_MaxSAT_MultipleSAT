"""
Wrapper to call external solvers (open-wbo, clasp) and parse output.
"""
import subprocess
from typing import Tuple
import os


def call_openwbo(wcnf_path: str, timeout: int = 60) -> str:
    # Lấy đường dẫn tuyệt đối của thư mục chứa file này (src)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Trỏ đến file open-wbo_bin nằm trong folder solvers
    OPENWBO_EXEC = os.path.join(BASE_DIR, 'solvers', 'open-wbo_bin')
    
    # Kiểm tra xem file có tồn tại không để debug dễ hơn
    if not os.path.isfile(OPENWBO_EXEC):
        return f"ERROR: Solver binary not found at {OPENWBO_EXEC}"

    cmd = [OPENWBO_EXEC, wcnf_path]
    
    # --- PHẦN BỊ THIẾU TRONG CODE CỦA BẠN ---
    try:
        # Gọi lệnh hệ thống để chạy solver
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        # Trả về kết quả (cả stdout và stderr)
        return proc.stdout + '\n' + proc.stderr
        
    except subprocess.TimeoutExpired:
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
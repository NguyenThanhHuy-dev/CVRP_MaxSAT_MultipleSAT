"""
Wrapper to call external solvers (open-wbo, clasp) and parse output.
"""
import subprocess
import os

def call_openwbo(wcnf_path: str, timeout: int = 60) -> str:
    # Lấy đường dẫn tuyệt đối của thư mục chứa file này (src)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Trỏ đến file open-wbo_bin nằm trong folder solvers
    OPENWBO_EXEC = os.path.join(BASE_DIR, 'solvers', 'open-wbo_bin')
    
    # Kiểm tra xem file có tồn tại không để debug dễ hơn
    if not os.path.isfile(OPENWBO_EXEC):
        return f"ERROR: Solver binary not found at {OPENWBO_EXEC}"

    # --- CẬP NHẬT: Thêm tham số -cpu-lim để Solver tự quản lý thời gian ---
    # Solver sẽ cố gắng dừng lại và in kết quả tốt nhất khi hết giờ
    cmd = [OPENWBO_EXEC, wcnf_path, f"-cpu-lim={timeout}"]
    
    try:
        # Tăng timeout của Python lên một chút (ví dụ +5s) để Solver kịp in kết quả trước khi bị kill
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
        
        # Trả về kết quả stdout (chứa dòng v ... và s ...)
        # Không cần in stderr ra màn hình để tránh rác log, trừ khi cần debug sâu
        return proc.stdout 
        
    except subprocess.TimeoutExpired:
        # Chỉ khi Solver bị treo cứng quá lâu (quá timeout+5s) mới vào đây
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
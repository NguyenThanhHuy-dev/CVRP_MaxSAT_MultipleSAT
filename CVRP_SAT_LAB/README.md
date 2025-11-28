# CVRP Solver using MaxSAT-based Column Generation

Dự án nghiên cứu giải quyết bài toán **Định tuyến xe có ràng buộc tải trọng (CVRP)** bằng phương pháp lai ghép mới: **MaxSAT-based Column Generation** (Sinh cột dựa trên MaxSAT).

Hệ thống sử dụng **Open-WBO** (Weighted MaxSAT Solver) làm động cơ tối ưu hóa chính, kết hợp với các thuật toán Heuristic (Clarke-Wright, 2-opt) để sinh các tuyến đường ứng viên.

---

## 🚀 Tính năng nổi bật

* **Iterative Solving:** Cơ chế vòng lặp sinh cột (Column Generation) giúp cải thiện chất lượng nghiệm theo thời gian.
* **MaxSAT Encoding:** Mô hình hóa bài toán chọn tuyến đường dưới dạng Weighted Partial MaxSAT (WCNF).
* **Visualization:** Tự động vẽ và lưu biểu đồ lộ trình tối ưu sau khi chạy.
* **Standard Benchmark:** Hỗ trợ đọc định dạng chuẩn `.vrp` từ CVRPLIB.

---

## 🛠️ Yêu cầu hệ thống

* **Hệ điều hành:** Linux (Ubuntu 20.04/22.04) hoặc WSL2 trên Windows.
* **Ngôn ngữ:** Python 3.10+.
* **Solver:** Open-WBO (Yêu cầu biên dịch từ C++).

---

## 📦 Hướng dẫn Cài đặt

### 1. Clone dự án và thiết lập môi trường
Chúng tôi khuyến nghị sử dụng **Conda** để quản lý môi trường.

```bash
# 1. Di chuyển vào thư mục dự án
cd CVRP_SAT_LAB

# 2. Tạo môi trường Conda (nếu chưa có)
conda create -n cvrp_env python=3.10 -y

# 3. Kích hoạt môi trường
conda activate cvrp_env

# 4. Cài đặt các thư viện Python cần thiết
pip install -r requirements.txt

```

### 2. Thiết lập Bộ giải (Solver)

Dự án yêu cầu file thực thi của **Open-WBO**.

1.  Đảm bảo file thực thi `open-wbo_bin` nằm trong thư mục `solvers/`.
    
2.  Cấp quyền thực thi cho file:
    

Bash

```
chmod +x solvers/open-wbo_bin

```

_(Lưu ý: Nếu bạn chưa có file này, vui lòng biên dịch từ nguồn `solvers/open-wbo` bằng lệnh `make r` và copy file `open-wbo_release` ra ngoài thành `open-wbo_bin`)._

----------

## 🏃 Hướng dẫn chạy

### 1. Chạy chế độ Demo (Dữ liệu giả lập)

Để kiểm tra hệ thống hoạt động đúng hay không:

Bash

```
cd src
python main_iterative.py

```

### 2. Chạy với dữ liệu Benchmark (.vrp)

Để giải các bài toán chuẩn (ví dụ `A-n32-k5.vrp`):

Bash

```
# Cấu trúc: python main_iterative.py <đường_dẫn_file_vrp>
cd src
python main_iterative.py ../data/A-n32-k5.vrp

```

Kết quả sẽ hiển thị trên màn hình console (Cost qua từng vòng lặp) và biểu đồ kết quả sẽ được lưu tại `results/plots/`.

----------

## 📂 Cấu trúc thư mục

Plaintext

```
CVRP_SAT_LAB/
├── data/                  # Chứa các file dữ liệu đầu vào (.vrp, .csv)
├── results/               # Kết quả đầu ra
│   ├── plots/             # Ảnh biểu đồ lộ trình (.png)
│   └── logs/              # (Tùy chọn) File log chạy
├── scripts/               # Các script tiện ích (run_experiment.sh,...)
├── solvers/               # Chứa bộ giải MaxSAT
│   ├── open-wbo/          # Mã nguồn Open-WBO
│   └── open-wbo_bin       # [QUAN TRỌNG] File thực thi của Solver
├── src/                   # Mã nguồn chính (Python)
│   ├── main_iterative.py  # Chương trình chính (Vòng lặp sinh cột)
│   ├── encoder.py         # Mã hóa bài toán sang WCNF
│   ├── decoder.py         # Giải mã kết quả từ Solver
│   ├── heuristic.py       # Thuật toán Clarke-Wright & 2-opt
│   ├── data_loader.py     # Đọc file .vrp
│   └── solver_service.py  # Wrapper gọi Open-WBO
├── environment.yml        # File cấu hình Conda
├── requirements.txt       # Các thư viện Python phụ thuộc
└── README.md              # File hướng dẫn này

```

----------

## 🧠 Nguyên lý hoạt động

Thuật toán hoạt động theo quy trình lặp (**Iterative Loop**):

1.  **Khởi tạo (Initialization):** Sử dụng thuật toán tham lam **Clarke-Wright Savings** để tạo ra tập hợp các tuyến đường ban đầu (Initial Pool).
    
2.  **Mã hóa (Encoding):** Chuyển bài toán chọn tập tuyến đường tối ưu từ Pool hiện tại thành công thức logic **Weighted MaxSAT**.
    
    -   _Hard Clauses:_ Mỗi khách hàng được phục vụ ít nhất 1 lần.
        
    -   _Soft Clauses:_ Chi phí chọn tuyến đường (Cost) được đưa vào trọng số phạt.
        
3.  **Giải (Solving):** Gọi **Open-WBO** để tìm tập hợp tuyến đường có tổng chi phí nhỏ nhất.
    
4.  **Sinh cột (Column Generation/Mutation):**
    
    -   Phân tích nghiệm hiện tại.
        
    -   Sử dụng các toán tử lai ghép (Merge/Swap/2-opt) để sinh ra các tuyến đường mới tiềm năng.
        
    -   Thêm tuyến đường mới vào Pool.
        
5.  **Lặp lại:** Quay lại bước 2 cho đến khi không tìm thấy cải thiện hoặc đạt số vòng lặp tối đa.
    

----------

## 📝 Kết quả thực nghiệm (Sơ bộ)

**Dataset**

**Best Cost (Current)**

**Optimal**

**Gap**

Demo (10 nodes)

90.47

-

-

A-n32-k5

~1259.58

784

~60%

_(Lưu ý: Kết quả đang trong giai đoạn tối ưu hóa Heuristic sinh cột để giảm Gap)_
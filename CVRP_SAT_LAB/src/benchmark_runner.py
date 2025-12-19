#!/usr/bin/env bash
# Script chạy thực nghiệm tự động cho Hybrid LNS
# Cách dùng: 
# 1. Cấp quyền: chmod +x scripts/run_experiment.sh
# 2. Chạy: ./scripts/run_experiment.sh

# Đường dẫn đến file main
MAIN_SCRIPT="src/main.py"

# Danh sách 13 bộ dữ liệu cần chạy (Đảm bảo bạn đã tải file .vrp về thư mục data/)
FILES=(
    # --- Nhóm Đã có ---
    "data/A-n32-k5.vrp"
    "data/E-n31-k7.vrp"
    "data/F-n45-k4.vrp"

    # --- Nhóm Nhỏ (Chạy nhanh) ---
    "data/P-n19-k2.vrp"
    "data/P-n22-k2.vrp"
    "data/A-n33-k5.vrp"
    "data/A-n37-k6.vrp"

    # --- Nhóm Trung bình (Test hiệu năng LNS) ---
    "data/B-n39-k5.vrp"
    "data/E-n51-k5.vrp"
    "data/B-n45-k5.vrp"
    "data/P-n55-k7.vrp"
    
    # --- Nhóm Lớn (Optional - Nếu máy khỏe) ---
    "data/A-n60-k9.vrp"
)

echo "🚀 Bắt đầu chạy thực nghiệm hàng loạt (Hybrid LNS)..."
echo "----------------------------------------------------"

for FILE in "${FILES[@]}"; do
    if [ -f "$FILE" ]; then
        echo "▶️  Đang chạy: $FILE"
        
        # Gọi python với method là 'lns'
        # Dùng 'timeout' của Linux để tự động ngắt nếu treo quá 10 phút (600s)
        # Để tránh việc 1 bài bị lỗi làm treo cả máy qua đêm.
        timeout 600s python3 "$MAIN_SCRIPT" "$FILE" --method lns
        
        EXIT_STATUS=$?
        if [ $EXIT_STATUS -eq 124 ]; then
            echo "⚠️  TIMEOUT: Bài toán $FILE chạy quá 600s và bị ngắt."
        fi
        
        echo "✅ Xong $FILE"
        echo "------------------------------------------------"
        
        # Nghỉ 2 giây để máy tản nhiệt tí
        sleep 2
    else
        echo "❌ Lỗi: Không tìm thấy file $FILE (Bạn đã tải về chưa?)"
    fi
done

echo "🎉 Đã hoàn tất toàn bộ danh sách thực nghiệm!"
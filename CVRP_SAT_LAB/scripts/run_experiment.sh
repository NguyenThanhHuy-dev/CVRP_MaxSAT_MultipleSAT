#!/usr/bin/env bash
# Script chạy thực nghiệm tự động
# Cách dùng: ./scripts/run_experiment.sh

# Kích hoạt conda nếu cần (tùy môi trường của bạn)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate cvrp_env

# Đường dẫn đến file main
MAIN_SCRIPT="src/main.py"

# Danh sách các file dữ liệu muốn chạy test
FILES=(
    "data/A-n32-k5.vrp"
    "data/E-n31-k7.vrp"
    # Thêm các file khác vào đây
)

echo "🚀 Bắt đầu chạy thực nghiệm hàng loạt..."

for FILE in "${FILES[@]}"; do
    if [ -f "$FILE" ]; then
        echo "------------------------------------------------"
        echo "▶️  Running: $FILE"
        python3 "$MAIN_SCRIPT" "$FILE" --method cg
    else
        echo "⚠️  File not found: $FILE"
    fi
done

echo "✅ Hoàn tất thực nghiệm!"
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_solution(instance, routes, cost):
    plt.figure(figsize=(10, 8))

    # Vẽ Depot
    depot_x, depot_y = (0, 0)
    if instance.coords and len(instance.coords) > 0:
        depot_x, depot_y = instance.coords[instance.depot]

    plt.scatter(depot_x, depot_y, c="red", marker="s", s=150, zorder=10, label="Depot")

    # Vẽ Khách hàng
    if instance.coords:
        xs = [c[0] for c in instance.coords[1:]]
        ys = [c[1] for c in instance.coords[1:]]
        plt.scatter(xs, ys, c="blue", s=40, zorder=5)

        for i in range(1, instance.n):
            if i < len(instance.coords):
                plt.text(
                    instance.coords[i][0],
                    instance.coords[i][1],
                    str(i),
                    fontsize=9,
                    ha="center",
                )

    # Vẽ Tuyến đường
    cmap = plt.get_cmap("tab20")
    for idx, r in enumerate(routes):
        route_coords = []
        for node in r:
            if node < len(instance.coords):
                route_coords.append(instance.coords[node])
            else:
                route_coords.append((0, 0))

        r_xs, r_ys = zip(*route_coords)

        # --- CẬP NHẬT: Tính tải trọng để hiển thị trên Legend ---
        load = sum(instance.demands[n] for n in r)
        label_str = f"R{idx+1} (Load: {load}/{instance.capacity})"
        # -------------------------------------------------------

        plt.plot(
            r_xs,
            r_ys,
            marker=".",
            linestyle="-",
            linewidth=2,
            color=cmap(idx % 20),
            label=label_str,
            alpha=0.7,
        )

        # Vẽ mũi tên hướng đi
        mid = len(r) // 2
        if mid < len(r) - 1:
            p1 = coords_at(instance, r[mid])
            p2 = coords_at(instance, r[mid + 1])
            plt.arrow(
                p1[0],
                p1[1],
                (p2[0] - p1[0]) * 0.5,
                (p2[1] - p1[1]) * 0.5,
                head_width=0.5,
                color=cmap(idx % 20),
            )

    # Tiêu đề chuẩn Benchmark
    gap = ((cost - instance.bks) / instance.bks * 100) if instance.bks > 0 else 0
    plt.title(
        f"Solution for {instance.name}\n"
        f"Cost: {cost:.2f} (Gap: {gap:.2f}%) | Vehicles: {len(routes)} | BKS: {instance.bks}"
    )
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    
    # Di chuyển Legend ra ngoài để không che bản đồ
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Lưu file
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    result_dir = os.path.join(base_dir, "results", "plots")
    os.makedirs(result_dir, exist_ok=True)

    save_path = os.path.join(result_dir, f"{instance.name}_solution.png")
    plt.savefig(save_path, dpi=300) # Tăng DPI cho ảnh nét hơn
    print(f"\n📊 Đã lưu biểu đồ trực quan tại: {save_path}")
    plt.close()

def coords_at(instance, node_idx):
    if node_idx < len(instance.coords):
        return instance.coords[node_idx]
    return (0, 0)
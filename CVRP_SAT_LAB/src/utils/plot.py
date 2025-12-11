import os
import matplotlib.pyplot as plt
import numpy as np

def plot_solution(instance, routes, cost):

    plt.figure(figsize=(10, 8))
    
    depot_x, depot_y = (0, 0)
    if instance.coords and len(instance.coords) > 0:
        depot_x, depot_y = instance.coords[instance.depot]
        
    plt.scatter(depot_x, depot_y, c='red', marker='s', s=150, zorder=10, label='Depot')
    

    if instance.coords:
        xs = [c[0] for c in instance.coords[1:]]
        ys = [c[1] for c in instance.coords[1:]]
        plt.scatter(xs, ys, c='blue', s=40, zorder=5)
        
        for i in range(1, instance.n):
            if i < len(instance.coords):
                plt.text(instance.coords[i][0], instance.coords[i][1], str(i), fontsize=9, ha='center')

    cmap = plt.get_cmap('tab20')
    for idx, r in enumerate(routes):
        route_coords = []
        for node in r:
            if node < len(instance.coords):
                route_coords.append(instance.coords[node])
            else:
                route_coords.append((0,0))
                
        r_xs, r_ys = zip(*route_coords)
        
        plt.plot(r_xs, r_ys, marker='.', linestyle='-', linewidth=2, 
                 color=cmap(idx % 20), label=f'Route {idx+1}', alpha=0.7)
        
        mid = len(r) // 2
        if mid < len(r) - 1:
            p1 = coords_at(instance, r[mid])
            p2 = coords_at(instance, r[mid+1])
            plt.arrow(p1[0], p1[1], (p2[0]-p1[0])*0.5, (p2[1]-p1[1])*0.5, 
                      head_width=0.5, color=cmap(idx % 20))

    plt.title(f"Solution for {instance.name}\nTotal Cost: {cost:.2f} | Vehicles: {len(routes)}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    result_dir = os.path.join(base_dir, 'results', 'plots')
    os.makedirs(result_dir, exist_ok=True)
    
    save_path = os.path.join(result_dir, f"{instance.name}_solution.png")
    plt.savefig(save_path)
    print(f"\n📊 Đã lưu biểu đồ trực quan tại: {save_path}")
    plt.close()

def coords_at(instance, node_idx):
    if node_idx < len(instance.coords):
        return instance.coords[node_idx]
    return (0,0)
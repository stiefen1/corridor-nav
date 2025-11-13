"""
Script for Optimal Decision Making (ODM) based on corridors.
"""
from decision_making.planner import Planner
from corridor_opt.corridor import Corridor
import os, pathlib, datetime as dt, matplotlib.pyplot as plt, matplotlib.cm as cm, numpy as np
from seacharts.enc import ENC
from corridor_opt.extract_shoreline import get_obstacles_in_window
import networkx as nx
from typing import List
import datetime as dt
from traffic.ais import snapshot_records, get_token, bbox_to_polygon
from datetime import datetime, timezone
from pyproj import Transformer
"""
TODO: Add method to WeatherClient to plot weather for debugging purpose
TODO: Add methods to display sub terms of the risk model -> prob of collision, prob of powered exit due to bad tracking, etc..
TODO: Integrate the risk value along the corridor instead of taking a single value.
TODO: Solve the corridor direction issues: When we compute force acting on the ship, we assume the ship is sailing in a specific direction in the corridor. But this information will only be available after we solve it.
----> Potential solution: We build a directed graph, where each edge of the original graph is double, one for each direction. 

"""

def create_corridor_colormap(corridors, costs, colormap='viridis'):
    """
    Create colors for corridors based on their costs using matplotlib colormap.
    
    Args:
        corridors: List of Corridor objects
        costs: List of float values (costs) matching corridors
        colormap: Matplotlib colormap name ('viridis', 'plasma', 'coolwarm', etc.)
    
    Returns:
        colors: List of colors (RGBA tuples) for each corridor
        colorbar_info: Dictionary with colormap and normalization for colorbar
    """
    costs = np.array(costs)
    
    # Normalize costs to [0, 1] range
    norm = plt.Normalize(vmin=costs.min(), vmax=costs.max())
    
    # Get colormap
    cmap = cm.get_cmap(colormap)
    
    # Generate colors for each corridor
    colors = [cmap(norm(cost)) for cost in costs]
    
    # Return colorbar info for adding a colorbar to plots
    colorbar_info = {
        'cmap': cmap,
        'norm': norm,
        'costs': costs
    }
    
    return colors, colorbar_info



# Load corridors
path_to_corridors = os.path.join(pathlib.Path(__file__).parent, 'output', 'corridors_best')
corridors = Corridor.load_all_corridors_in_folder(path_to_corridors)
print(f"Found {len(corridors)} corridors")
corridors[0].area

# Visualize nodes
# _, ax = plt.subplots()
# for corridor in corridors:
#     corridor.plot(ax=ax)
#     ax.text(*corridor.centroid, s=f"{corridor.prev_main_node}, {corridor.next_main_node}")
# plt.show()
# plt.close()


# Load obstacles from enc
enc_config = os.path.join('config', 'kristiansund.yaml')
enc = ENC(enc_config)
xlim = enc.bbox[0], enc.bbox[2]
ylim = enc.bbox[1], enc.bbox[3]
obstacles = get_obstacles_in_window(enc, depth=10)
target_node = 40696
u_des = 3 # m/s

# 2) AIS and Traffic helpers
TO_WGS = Transformer.from_crs(32633, 4326, always_xy=True).transform     # x/y (UTM33N) -> lon/lat
x_min, y_min, x_max, y_max = enc.bbox
lon_min, lat_min = TO_WGS(x_min, y_min)
lon_max, lat_max = TO_WGS(x_max, y_max)
aoi = bbox_to_polygon((lat_min, lon_min, lat_max, lon_max), "latlon")
T_ISO = "2025-10-31T12:00:00Z"
time_shot = datetime.fromisoformat(T_ISO.replace("Z", "+00:00")).astimezone(timezone.utc)
records = snapshot_records(aoi, time_shot)

# === Coordinate conversion WGS84 → UTM33N ===
transformer = Transformer.from_crs(4326, 32633, always_xy=True)
def wgs84_to_utm33n(lat, lon):
    x, y = transformer.transform(lon, lat)
    return x, y





planner = Planner(
    corridors,
    target_node,
    ais_client=records,
    mu=1e-5
)

node = 1343
t = dt.datetime(2025, 11, 11)
while node != target_node:
    nodes, distance, corridors_total = planner.get_optimal_corridor_sequence(
        start_node=node,
        u=u_des,
        when_utc=dt.datetime.now(dt.UTC),
        ship_nominal_maneuverability=1e3,
        ship_nominal_tracking_accuracy=5,
        disable_wave=True,
        weight='total'
    )

    # Extract all the corridors until the next intersection
    corridors_until_next_node: List[Corridor] = planner.graph_of_corridors[node][nodes[1]][corridors_total[0].edge_id]['corridors']
    
    # Compute the time it will take to reach next main node, to simulate actual mission
    total_length = 0
    for corridor in corridors_until_next_node:
        total_length += corridor.backbone.length
    travel_time = dt.timedelta(seconds=total_length/u_des)
    print(f"Travel time: {travel_time}")
    t = t + travel_time

    # Plot corridors with colors
    fig, ax = plt.subplots(figsize=(10, 8))

    for corridor in corridors_total:
        corridor.fill(ax=ax, c='orange', alpha=0.7)

    if records:
        ship_xy = np.array([wgs84_to_utm33n(r["latitude"], r["longitude"]) for r in records])
        ax.scatter(ship_xy[:,0], ship_xy[:,1], s=18, color='red', label='Ships')

    # Plot obstacles
    for obs in obstacles:
        obs.fill(ax=ax, c='forestgreen')

    ax.set_title(f"T = {t}")
    ax.set_aspect('equal')

    node = nodes[1]

    plt.show()
    plt.close()
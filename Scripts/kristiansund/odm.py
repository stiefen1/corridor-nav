"""
Script for Optimal Decision Making (ODM) based on corridors.
"""
from decision_making.planner import Planner
from corridor_opt.corridor import Corridor
import os, pathlib, datetime as dt, matplotlib.pyplot as plt, matplotlib.cm as cm, numpy as np
from seacharts.enc import ENC
from corridor_opt.extract_shoreline import get_obstacles_in_window
import networkx as nx
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


# Load obstacles from enc
enc_config = os.path.join('config', 'kristiansund.yaml')
enc = ENC(enc_config)
xlim = enc.bbox[0], enc.bbox[2]
ylim = enc.bbox[1], enc.bbox[3]
obstacles = get_obstacles_in_window(enc, depth=10)





# # 2) AIS and Traffic helpers
# TO_WGS = Transformer.from_crs(32633, 4326, always_xy=True).transform     # x/y (UTM33N) -> lon/lat
# x_min, y_min, x_max, y_max = enc.bbox
# lon_min, lat_min = TO_WGS(x_min, y_min)
# lon_max, lat_max = TO_WGS(x_max, y_max)
# aoi = bbox_to_polygon((lat_min, lon_min, lat_max, lon_max), "latlon")
# T_ISO = "2025-10-31T12:00:00Z"
# time_shot = datetime.fromisoformat(T_ISO.replace("Z", "+00:00")).astimezone(timezone.utc)
# records = snapshot_records(aoi, time_shot)






planner = Planner(
    corridors,
    # records,
    40696,
    mu=1e-5
)

planner.graph_of_corridors

u_des = 3 # m/s
corridors_total = planner.get_optimal_corridor_sequence(
    start_node=1343,
    u=u_des,
    when_utc=dt.datetime.now(dt.UTC),
    ship_nominal_maneuverability=1e3,
    ship_nominal_tracking_accuracy=5,
    disable_wave=True,
    weight='total'
)

corridors_risk = planner.get_optimal_corridor_sequence(
    start_node=1343,
    u=u_des,
    when_utc=dt.datetime.now(dt.UTC),
    ship_nominal_maneuverability=1e3,
    ship_nominal_tracking_accuracy=5,
    disable_wave=True,
    weight='risk'
)

corridors_energy = planner.get_optimal_corridor_sequence(
    start_node=1343,
    u=u_des,
    when_utc=dt.datetime.now(dt.UTC),
    ship_nominal_maneuverability=1e3,
    ship_nominal_tracking_accuracy=5,
    disable_wave=True,
    weight='energy'
)



# Total cost
# colors_tot, cb_info_tot = create_corridor_colormap(corridors, costs, colormap='viridis')
# colors_energy, cb_info_energy = create_corridor_colormap(corridors, info['energy_consumption'], colormap='viridis')
# colors_travel_time, cb_travel_time = create_corridor_colormap(corridors, info['expected_travel_time'], colormap='viridis')

# Plot corridors with colors
fig, axs = plt.subplots(1, 3, figsize=(10, 8))

for corridor in corridors_total:
    corridor.fill(ax=axs[0], c='orange', alpha=0.7)
for corridor in corridors_energy:
    corridor.fill(ax=axs[1], c='orange', alpha=0.7)
for corridor in corridors_risk:
    corridor.fill(ax=axs[2], c='orange', alpha=0.7)

# for corridor in corridors:
#     for ax in axs:
#         corridor.fill(ax=ax, c='orange', alpha=0.7)

# for corridor, color_tot, color_energy, color_travel_time in zip(corridors, colors_tot, colors_energy, colors_travel_time):
#     corridor.fill(ax=axs[0], c=color_tot, alpha=0.7)
#     corridor.fill(ax=axs[1], c=color_energy, alpha=0.7)
#     corridor.fill(ax=axs[2], c=color_travel_time, alpha=0.7)

# Plot obstacles
for obs in obstacles:
    for ax in axs:
        obs.fill(ax=ax, c='green')

# for weather_sample in info['weather_samples']:
#     for ax in axs:
#         weather_sample.quiver_wind(ax=ax, color='red')
#         weather_sample.quiver_current(ax=ax, color='blue')
#         weather_sample.quiver_wave(ax=ax, color='purple')

# Add colorbar
# sm_tot = plt.cm.ScalarMappable(cmap=cb_info_tot['cmap'], norm=cb_info_tot['norm'])
# sm_energy = plt.cm.ScalarMappable(cmap=cb_info_energy['cmap'], norm=cb_info_energy['norm'])
# sm_travel_time = plt.cm.ScalarMappable(cmap=cb_travel_time['cmap'], norm=cb_travel_time['norm'])

# sm_tot.set_array([])
# sm_energy.set_array([])
# sm_travel_time.set_array([])

# cbar_tot = plt.colorbar(sm_tot, ax=axs[0])
# cbar_energy = plt.colorbar(sm_energy, ax=axs[1])
# cbar_travel_time = plt.colorbar(sm_travel_time, ax=axs[2])

# cbar_tot.set_label('Cost')
# cbar_energy.set_label('Cost')
# cbar_travel_time.set_label('Cost')

axs[0].set_title(f"Total cost")
axs[1].set_title(f"Energy consumption")
axs[2].set_title(f"Expected travel time")

for ax in axs:
    ax.set_aspect('equal')

plt.show()
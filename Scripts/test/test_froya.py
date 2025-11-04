import os
from corridor_opt.extract_shoreline import get_obstacles_in_window
from seacharts.enc import ENC
import matplotlib.pyplot as plt


depth = 0 # meters

# SeaCharts
config_path = os.path.join('config', 'froya.yaml')
enc = ENC(config_path)

# Extract obstacles at a given depth
obstacles = get_obstacles_in_window(enc, depth=depth)

# Show obstacles
ax = None
for obs in obstacles:
    ax = obs.plot(ax=ax)
    obs.fill(ax=ax, c='green')
if ax is not None:
    ax.set_aspect('equal')
    plt.show()
else:
    print(f"obstacles seems to be empty for depth={depth} m: len(obstacles)={len(obstacles)}")
import os
import numpy as np
import matplotlib.pyplot as plt
from corridor_opt.extract_shoreline import get_obstacles_in_window
from seacharts.enc import ENC
from shapely.geometry import Polygon
from pyproj import Transformer

# === AIS snapshot ===
from traffic.ais import snapshot_records, get_token, bbox_to_polygon
from datetime import datetime, timezone

# === ENC map ===
depth = 5
config_path = os.path.join('config', 'kristiansund.yaml')
enc = ENC(config_path)
obstacles = get_obstacles_in_window(enc, depth=depth)

fig, ax = plt.subplots(figsize=(10, 8))
for obs in obstacles:
    ax = obs.plot(ax=ax)
    obs.fill(ax=ax, c='green')

# AOI bounding box
BBOX = (62.88, 7.61, 63.17, 8.13)
BBOX_ORDER = "latlon"
aoi = bbox_to_polygon(BBOX, BBOX_ORDER)

# Time
T_ISO = "2025-10-29T12:00:00Z"
DELTA_MIN = 15

token = get_token()
T = datetime.fromisoformat(T_ISO.replace("Z","+00:00")).astimezone(timezone.utc)
records = snapshot_records(token, aoi, T, DELTA_MIN)
print(f"{len(records)} ships retrieved.")

# === Coordinate conversion WGS84 → UTM33N ===
transformer = Transformer.from_crs(4326, 32633, always_xy=True)
def wgs84_to_utm33n(lat, lon):
    x, y = transformer.transform(lon, lat)
    return x, y

if records:
    ship_xy = np.array([wgs84_to_utm33n(r["latitude"], r["longitude"]) for r in records])
    ax.scatter(ship_xy[:,0], ship_xy[:,1], s=18, color='red', label='Ships')

ax.legend()
ax.set_aspect('equal')
plt.title("Kristiansund ENC with AIS ships (snapshot)")
plt.show()

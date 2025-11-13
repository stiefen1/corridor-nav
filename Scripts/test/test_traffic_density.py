from __future__ import annotations
"""
Optimized AIS traffic density plotting script
- Loads ENC and obstacles once
- Fetches AIS snapshot once
- Plots obstacles, ships, and corridors cleanly
- Computes density per corridor with one-line API

Requirements:
  - seacharts.enc.ENC
  - corridor_opt.extract_shoreline.get_obstacles_in_window
  - corridor_opt.corridor.Corridor
  - traffic.ais: snapshot_records, get_token, bbox_to_polygon
  - traffic.traffic_density_eval.TrafficDensityCalculator
"""

import os
from datetime import datetime, timezone
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib import colormaps
from pyproj import Transformer

from seacharts.enc import ENC
from corridor_opt.extract_shoreline import get_obstacles_in_window
from corridor_opt.corridor import Corridor

from traffic.ais import snapshot_records, get_token, bbox_to_polygon
from traffic.traffic_density_eval import TrafficDensityCalculator

# ----------------------------
# Config & tuning knobs
# ----------------------------
DEPTH_M = 5
CONFIG_PATH = os.path.join('config', 'kristiansund.yaml')
TIME_ISO = "2025-10-31T12:00:00Z"  # UTC ISO
DELTA_MIN = 15

# Traffic density knobs
BUFFER_M = 500.0
D_MIN = 10.0
D_MAX = 1000.0
AREA_SHAPE_FACTOR = 1.0
OVERLAP_F = 1.0
IMPUTE_AREA_M2 = 1500.0   # set to None to skip ships with unknown area
INCLUDE_BOUNDARY = True

# Paths
CORRIDOR_DIR = os.path.join('Scripts', 'kristiansund', 'output', 'corridors_best')

# CRS transformers
TO_METRIC = Transformer.from_crs(4326, 32633, always_xy=True).transform  # lon/lat -> x/y (UTM33N)
TO_WGS = Transformer.from_crs(32633, 4326, always_xy=True).transform     # x/y (UTM33N) -> lon/lat

# ----------------------------
# Helpers
# ----------------------------
def _color_for_pair(pair: Tuple[int, int], cache: Dict[Tuple[int, int], Tuple[float, float, float]], cmap_name: str = 'tab20'):
    """Deterministic color from a corridor (prev_main_node, next_main_node) pair."""
    if pair in cache:
        return cache[pair]
    cmap = colormaps.get_cmap(cmap_name)
    # Use a stable hash to index the colormap
    idx = (hash(pair) % 20) / 20.0
    color = cmap(idx)[:3]
    cache[pair] = color
    return color


def _plot_obstacles(ax, obstacles):
    for obs in obstacles:
        # Fill then outline for a clean map
        obs.fill(ax=ax, c='lightgreen', alpha=0.45)
        obs.plot(ax=ax, c='green', lw=0.8)


def _plot_ships(ax, records):
    if not records:
        return None
    # Convert to metric once for plotting
    xs, ys = [], []
    for r in records:
        lon = r.get('longitude'); lat = r.get('latitude')
        if lon is None or lat is None:
            continue
        x, y = TO_METRIC(lon, lat)
        xs.append(x); ys.append(y)
    if not xs:
        return None
    sc = ax.scatter(np.array(xs), np.array(ys), s=12, marker='o', alpha=0.85, label='AIS ships', zorder=3)
    return sc


def _plot_corridor(ax, corridor, color):
    # Polygon fill
    xy = np.array(corridor.get_xy_as_list())
    patch = MplPolygon(xy, closed=True, facecolor=color, edgecolor=color, alpha=0.25, lw=1.0, label='Corridor')
    ax.add_patch(patch)
    # Backbone
    ts = np.linspace(0, 1, 60)
    bb = np.array([corridor.backbone.interpolate(t, normalized=True).xy for t in ts])
    ax.plot(bb[:, 0], bb[:, 1], '--', c=color, lw=1.6, label='Backbone')
    # Centroid
    cx, cy = corridor.centroid
    ax.scatter([cx], [cy], marker='x', c=[color], s=30, lw=1.2, label='Centroid')


# ----------------------------
# Main
# ----------------------------
def main():
    # 1) ENC & obstacles
    enc = ENC(CONFIG_PATH)
    obstacles = get_obstacles_in_window(enc, depth=DEPTH_M)

    # 2) AIS snapshot once (BBOX from ENC)
    x_min, y_min, x_max, y_max = enc.bbox
    lon_min, lat_min = TO_WGS(x_min, y_min)
    lon_max, lat_max = TO_WGS(x_max, y_max)
    aoi = bbox_to_polygon((lat_min, lon_min, lat_max, lon_max), "latlon")

    #token = get_token()
    T = datetime.fromisoformat(TIME_ISO.replace("Z", "+00:00")).astimezone(timezone.utc)
    records = snapshot_records(aoi, T)

    # 3) Load all corridors
    corridors = Corridor.load_all_corridors_in_folder(CORRIDOR_DIR)


    # 4) Figure & base map
    fig, ax = plt.subplots(figsize=(11, 9))
    _plot_obstacles(ax, obstacles)

    # Plot ships once (not per corridor)
    _plot_ships(ax, records)

    # 5) Loop corridors: compute density and draw corridor/backbone/centroid
    color_cache: Dict[Tuple[int, int], Tuple[float, float, float]] = {}

    print("=== Traffic density per corridor ===")
    for corridor in corridors:
        pair = (corridor.prev_main_node, corridor.next_main_node)
        color = _color_for_pair(pair, color_cache)

        # One-line evaluation (uses WGS84 records; transformer defined above)
        res = TrafficDensityCalculator.evaluate_density_for_corridor(
            corridor_obj=corridor,
            ais_records=records,
            buffer_m=BUFFER_M,
            d_min=D_MIN,
            D_max=D_MAX,
            impute_area_m2=IMPUTE_AREA_M2
        )

        print(f"id={corridor.id}  density={res['density']}  used={res['ships_used_in_sum']:>3}  total={res['ships_total_considered']:>3}")

        # Draw corridor glyphs
        _plot_corridor(ax, corridor, color)

    # 6) Aesthetics
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('AIS Traffic Density — Kristiansund Corridors', fontsize=14, pad=10)
    ax.grid(True, linestyle=':', alpha=0.6)

    # Legend: de-duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc='lower left', frameon=True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

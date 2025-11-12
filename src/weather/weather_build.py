import numpy as np
from pyproj import Transformer

from weather.weather_build_helpers import WeatherClient

import datetime as dt


def latlon_grid_from_bbox(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float,
    spacing_km: float = 5.0,
    utm_epsg: int = 32633,   # UTM zone 33N (common for Norway). Change if needed.
    include_edges: bool = False
):
    """
    Build a grid of (lat, lon) points inside a bounding box.
    - spacing_km: approximate grid spacing in kilometers
    - include_edges: if True, include points exactly on bbox edges; otherwise use cell centers strictly inside.
    Returns: list[(lat, lon)]
    """
    # 1) Build forward/backward transformers (lon,lat) <-> (E,N)
    to_utm = Transformer.from_crs(4326, utm_epsg, always_xy=True)  # lon/lat -> E/N
    to_wgs = Transformer.from_crs(utm_epsg, 4326, always_xy=True)  # E/N -> lon/lat

    # 2) Project bbox corners to UTM
    e_min, n_min = to_utm.transform(min_lon, min_lat)
    e_max, n_max = to_utm.transform(max_lon, max_lat)

    # 3) Ensure proper ordering
    if e_min > e_max: e_min, e_max = e_max, e_min
    if n_min > n_max: n_min, n_max = n_max, n_min

    dx = spacing_km * 1000.0
    dy = spacing_km * 1000.0

    if include_edges:
        e_vals = np.arange(e_min, e_max + 1e-6, dx)
        n_vals = np.arange(n_min, n_max + 1e-6, dy)
    else:
        # use cell centers (strictly inside bbox)
        e_vals = np.arange(e_min + dx/2.0, e_max - dx/2.0 + 1e-6, dx)
        n_vals = np.arange(n_min + dy/2.0, n_max - dy/2.0 + 1e-6, dy)

    if len(e_vals) == 0 or len(n_vals) == 0:
        return []

    E, N = np.meshgrid(e_vals, n_vals)
    E_flat = E.ravel()
    N_flat = N.ravel()

    # 4) Convert back to lon/lat, then return as (lat, lon)
    lons, lats = to_wgs.transform(E_flat, N_flat)
    return list(zip(lats.tolist(), lons.tolist()))


if __name__ == '__main__':
    # Your bbox (lat/lon):
    bbox = (62.88, 7.61, 63.17, 8.13)  # (min_lat, min_lon, max_lat, max_lon)
    points = latlon_grid_from_bbox(*bbox, spacing_km=0.5, utm_epsg=32633, include_edges=False)

    print(len(points))

    # Run this hourly to snapshot the whole box
    now = dt.datetime.now(dt.UTC)
    wc = WeatherClient(
        user_agent="AutoShipWeatherRecorder/1.0 you@example.com",
        mode="met",
        source="live",
        archive_csv="src/weather/data/kristiansund_weather.csv"
    )

    for lat, lon in points:
        wc.get(lat=lat, lon=lon, when_utc=now)


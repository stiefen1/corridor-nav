from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from shapely import Polygon as ShapelyPolygon
from corridor_opt.geometry import GeometryWrapper

from pyproj import Transformer



XY = List[Tuple[float, float]]

TO_METRIC = Transformer.from_crs(4326, 32633, always_xy=True).transform  # lon/lat -> x/y (UTM33N)
TO_WGS = Transformer.from_crs(32633, 4326, always_xy=True).transform     # x/y (UTM33N) -> lon/lat

class TrafficDensityCalculator:
    """
    Traffic density calculator with a one-call API for corridor evaluation.

    Steps:
    1) Build & expand corridor polygon
    2) Collect AIS ships inside expanded corridor (WGS84 → metric CRS)
    3) Set centerline from the Corridor object
    4) Compute density with Option-B (quadratic clamp) distance weights

    Use `evaluate_density_for_corridor(...)` to run all steps in one line.
    """

    def __init__(self, corridor_vertices_xy: XY):
        self._base: GeometryWrapper = self._ensure_polygon(corridor_vertices_xy)
        self._expanded: Optional[GeometryWrapper] = None
        self.params = {"buffer_m": None, "d_min": None, "D_max": None}
        # CRS transformers (optional): set via set_transformers()
        self._to_metric = None       # callable: (lon, lat) -> (x, y) in metric CRS
        self._to_wgs84 = None        # callable: (x, y) -> (lon, lat)
        # storage for ships inside expanded corridor (metric coords)
        self._ships_inside = []      # list of dicts with x, y, and provided fields
        # centerline (shapely LineString expected)
        self._centerline = None

    # ---------- Step 1: corridor expansion ----------
    def expand_corridor(self, buffer_m: float, join_style: int = 1, mitre_limit: float = 2.0, resolution: int = 16) -> None:
        if buffer_m <= 0:
            raise ValueError("buffer_m must be positive to expand the corridor.")
        self._expanded = self._base.buffer(
            distance=buffer_m,
            join_style=join_style,
            mitre_limit=mitre_limit,
            resolution=resolution,
        )
        self.params["buffer_m"] = buffer_m

    # ---------- Step 2: AIS collection into expanded corridor ----------
    def set_transformers(self, to_metric_callable=None, to_wgs84_callable=None) -> None:
        self._to_metric = to_metric_callable
        self._to_wgs84 = to_wgs84_callable

    def clear_ships(self) -> None:
        self._ships_inside = []

    def collect_ships_in_expanded(self, ais_records: list, *, area_shape_factor: float = 0.85, include_boundary: bool = True) -> int:
        """
        Filter AIS records to those whose centroid lies within the expanded corridor.
        Safely handles None/strings in numeric fields.

        include_boundary=True uses polygon.covers(point) so ships on the boundary count.
        """
        if self._expanded is None:
            raise RuntimeError("Call expand_corridor(buffer_m) before collecting ships.")

        def _to_float(val, default=0.0):
            try:
                if val is None:
                    return default
                if isinstance(val, str):
                    s = val.strip()
                    if s == "" or s.lower() in {"nan", "none", "null"}:
                        return default
                    return float(s)
                return float(val)
            except (TypeError, ValueError):
                return default

        self._ships_inside = []
        exp_poly = self._expanded._geometry  # underlying shapely geometry
        from shapely.geometry import Point

        for r in ais_records:
            # Coordinates (metric)
            if ("x" in r and "y" in r):
                x = _to_float(r["x"], default=None)
                y = _to_float(r["y"], default=None)
            else:
                if self._to_metric is None:
                    raise RuntimeError("No metric transformer set. Provide to_metric via set_transformers().")
                lon = _to_float(r.get("longitude"), default=None)
                lat = _to_float(r.get("latitude"), default=None)
                if lon is None or lat is None:
                    continue
                x, y = self._to_metric(lon, lat)

            if x is None or y is None:
                continue

            pt = Point(x, y)
            inside = exp_poly.covers(pt) if include_boundary else exp_poly.contains(pt)
            if not inside:
                continue

            # Area preference with robust fallback
            area_m2 = _to_float(r.get("area_rect_m2"), default=None)
            if area_m2 is None or area_m2 <= 0:
                area_m2 = _to_float(r.get("area_ellip_m2"), default=None)
            if area_m2 is None or area_m2 <= 0:
                L = _to_float(r.get("shipLength"), default=0.0)
                W = _to_float(r.get("shipWidth"), default=0.0)
                area_m2 = area_shape_factor * L * W if (L > 0 and W > 0) else 0.0

            stored = dict(r)  # preserve original fields
            stored.update({"x": x, "y": y, "area_m2": area_m2})
            self._ships_inside.append(stored)

        return len(self._ships_inside)

    @property
    def ships_inside(self) -> list:
        return list(self._ships_inside)

    # ---------- Step 3: set centerline & compute density ----------
    def set_centerline_from_corridor(self, corridor_obj) -> None:
        """Store the corridor backbone (shapely LineString) as centerline."""
        try:
            self._centerline = corridor_obj.backbone
        except Exception as e:
            raise ValueError("corridor_obj.backbone not available/valid") from e

    def compute_density(self,
                        d_min: float,
                        D_max: float,
                        *,
                        overlap_fraction: float = 1.0,
                        impute_area_m2: Optional[float] = None,
                        debug: bool = False) -> dict:
        """
        Compute availability and density using Option-B quadratic distance weights.
        - overlap_fraction: constant f_i (keep 1.0 for the simple model)
        - impute_area_m2: if provided, use this area when a ship has area_m2 <= 0; if None, skip such ships.
        """
        if self._expanded is None:
            raise RuntimeError("expand_corridor() must be called before compute_density().")
        if self._centerline is None:
            raise RuntimeError("set_centerline_from_corridor() must be called before compute_density().")

        A_base = float(self.base_area)
        if A_base <= 0:
            raise ValueError("Base corridor area is non-positive.")

        from shapely.geometry import Point
        line = self._centerline

        A_occ = 0.0
        n_total = 0
        n_used = 0
        dbg_rows = []

        for s in self._ships_inside:
            n_total += 1
            area = float(s.get("area_m2", 0.0) or 0.0)
            if area <= 0.0:
                if impute_area_m2 is None:
                    if debug:
                        dbg_rows.append((s.get('mmsi'), 'area<=0 skip', None, None, 0.0))
                    continue
                area = float(impute_area_m2)

            p = Point(float(s["x"]), float(s["y"]))
            d = float(line.distance(p))  # meters
            w = self.weight_quadratic_clamp(d, d_min, D_max)
            if w <= 0.0:
                if debug:
                    dbg_rows.append((s.get('mmsi'), 'w=0 skip', d, w, area))
                continue

            occ = area * overlap_fraction * w
            A_occ += occ
            n_used += 1
            if debug:
                dbg_rows.append((s.get('mmsi'), 'used', d, w, area))

        A_avail = max(0.0, A_base - A_occ)
        availability = max(0.0, min(1.0, A_avail / A_base))
        density = max(1e-10, min(1.0, 1.0 - availability))#Fix the 0 part later

        out = {
            "base_area": A_base,
            "occupied_effective_area": A_occ,
            "available_area": A_avail,
            "availability": availability,
            "density": density,
            "ships_total_considered": n_total,
            "ships_used_in_sum": n_used,
        }
        if debug:
            out["debug_rows"] = dbg_rows
        return out

    # ---------- One-line convenience API ----------
    @staticmethod
    def evaluate_density_for_corridor(
        corridor_obj,
        ais_records: list,
        *,
        to_metric=TO_METRIC,               # callable(lon, lat) -> (x, y)
        to_wgs84=TO_WGS,           # optional callable(x, y) -> (lon, lat)
        buffer_m: float = 500.0,
        d_min: float = 10.0,
        D_max: float = 1000.0,
        area_shape_factor: float = 1.0,
        overlap_fraction: float = 1.0,
        impute_area_m2: Optional[float] = 1500.0,
        include_boundary: bool = True,
        debug: bool = False,
    ) -> dict:
        """
        One-call evaluation used inside your main loop over corridors.

        Example:
            res = TrafficDensityCalculator.evaluate_density_for_corridor(
                corridor_obj=corridor,          # has .get_xy_as_list() and .backbone
                ais_records=records,            # list of AIS dicts
                to_metric=_to_metric,           # lon/lat -> x/y in metric CRS
                buffer_m=500,
                d_min=10,
                D_max=1000,
                area_shape_factor=0.85,
                overlap_fraction=1.0,
                impute_area_m2=1500.0,          # or None to skip unknown-area ships
                include_boundary=True,
                debug=False,
            )
        """
        calc = TrafficDensityCalculator(corridor_obj.get_xy_as_list())
        calc.set_transformers(to_metric_callable=to_metric, to_wgs84_callable=to_wgs84)
        calc.expand_corridor(buffer_m=buffer_m)
        calc.collect_ships_in_expanded(
            ais_records,
            area_shape_factor=area_shape_factor,
            include_boundary=include_boundary,
        )
        calc.set_centerline_from_corridor(corridor_obj)
        return calc.compute_density(
            d_min=d_min,
            D_max=D_max,
            overlap_fraction=overlap_fraction,
            impute_area_m2=impute_area_m2,
            debug=debug,
        )

    # ---------- Accessors ----------
    @property
    def base_vertices(self) -> XY:
        return self._base.get_xy_as_list()

    @property
    def base_area(self) -> float:
        return self._base.area

    @property
    def expanded_vertices(self) -> XY:
        if self._expanded is None:
            raise RuntimeError("Call expand_corridor(buffer_m) first.")
        return self._expanded.get_xy_as_list()

    @property
    def expanded_area(self) -> float:
        if self._expanded is None:
            raise RuntimeError("Call expand_corridor(buffer_m) first.")
        return self._expanded.area

    # ---------- Math helpers ----------
    @staticmethod
    def weight_quadratic_clamp(d: float, d_min: float, D_max: float) -> float:
        """
        Option B — quadratic clamp:
          w=1 for d<=d_min;
          w=((1 - (d-d_min)/(D_max-d_min))**2) for d_min<d<D_max;
          w=0 for d>=D_max
        """
        if D_max <= d_min:
            raise ValueError("Require D_max > d_min.")
        if d <= d_min:
            return 1.0
        if d >= D_max:
            return 0.0
        t = 1.0 - (d - d_min) / (D_max - d_min)
        return float(t * t)

    @staticmethod
    def _ensure_polygon(vertices: XY) -> GeometryWrapper:
        if len(vertices) < 3:
            raise ValueError("Corridor needs at least 3 vertices.")
        # Drop duplicate closing point if present
        if np.allclose(vertices[0], vertices[-1]):
            vertices = vertices[:-1]
        poly = ShapelyPolygon(vertices)
        if not poly.is_valid:
            poly = poly.buffer(0.0)
        if not poly.is_valid:
            raise ValueError("Corridor polygon is invalid (self-intersections?).")
        return GeometryWrapper(polygon=poly, geometry_type=ShapelyPolygon)


# Optional tiny helper for plotting
def _color_by_weight(ws: np.ndarray) -> np.ndarray:
    """Clamp weights to [0,1] for use as alphas or colors."""
    ws = np.asarray(ws, dtype=float)
    return np.clip(ws, 0.0, 1.0)




if __name__ == "__main__":
    """
    Steps 1–3 demo:
    - Load corridor[0]
    - Expand by 500 m
    - Pull AIS snapshot in ENC BBOX, filter to expanded corridor
    - Set centerline from corridor.backbone
    - Compute density (Option-B weights) and plot ships colored by weight
    """
    from corridor_opt.corridor import Corridor
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    from seacharts.enc import ENC
    from corridor_opt.extract_shoreline import get_obstacles_in_window
    from pyproj import Transformer
    from datetime import datetime, timezone

    # Your AIS helpers
    from traffic.ais import snapshot_records, get_token, bbox_to_polygon

    # ---------------- Load corridor ----------------
    path_to_corridors = os.path.join('Scripts', 'kristiansund', 'output', 'corridors_best')
    corridors = Corridor.load_all_corridors_in_folder(path_to_corridors)
    corridor = corridors[33]
    corridor_vertices = corridor.get_xy_as_list()

    calc = TrafficDensityCalculator(corridor_vertices)
    calc.expand_corridor(buffer_m=500.0)

    # ---------------- Prepare ENC & transformers (UTM33N <-> WGS84) ----------------
    depth = 5
    config_path = os.path.join('config', 'kristiansund.yaml')
    enc = ENC(config_path)
    obstacles = get_obstacles_in_window(enc, depth=depth)

    # UTM33N (EPSG:32633) <-> WGS84 (EPSG:4326)
    to_wgs = Transformer.from_crs(32633, 4326, always_xy=True)
    to_metric = Transformer.from_crs(4326, 32633, always_xy=True)

    def _to_metric(lon, lat):
        x, y = to_metric.transform(lon, lat)
        return x, y

    def _to_wgs(x, y):
        lon, lat = to_wgs.transform(x, y)
        return lon, lat

    calc.set_transformers(to_metric_callable=_to_metric, to_wgs84_callable=_to_wgs)

    # ---------------- Get AIS snapshot in ENC bbox ----------------
    x_min, y_min, x_max, y_max = enc.bbox
    lon_min, lat_min = to_wgs.transform(x_min, y_min)
    lon_max, lat_max = to_wgs.transform(x_max, y_max)

    T_ISO = "2025-10-31T12:00:00Z"
    DELTA_MIN = 15
    T = datetime.fromisoformat(T_ISO.replace("Z","+00:00")).astimezone(timezone.utc)

    token = get_token()
    aoi = bbox_to_polygon((lat_min, lon_min, lat_max, lon_max), "latlon")
    records = snapshot_records(aoi, T, DELTA_MIN)

    # ---------------- Filter ships to expanded corridor ----------------
    calc.collect_ships_in_expanded(records, area_shape_factor=1)

    # ---------------- Centerline & density ----------------
    calc.set_centerline_from_corridor(corridor)
    d_min = 10.0
    D_max = 1000.0
    results = calc.compute_density(d_min=d_min, D_max=D_max, overlap_fraction=1.0, impute_area_m2=1500.0)

    print("=== Density results ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    # ---------------- Plot ----------------
    fig, ax = plt.subplots(figsize=(9, 7))

    # Obstacles for context
    for obs in obstacles:
        obs.fill(ax=ax, c='lightgreen', alpha=0.5)
        obs.plot(ax=ax, c='green')

    # Base and expanded corridor
    base_xy = np.array(calc.base_vertices)
    exp_xy = np.array(calc.expanded_vertices)
    base_patch = mpatches.Polygon(base_xy, closed=True, facecolor='tab:red', alpha=0.20, edgecolor='tab:red', linewidth=1.0, label='Base corridor')
    exp_patch  = mpatches.Polygon(exp_xy,  closed=True, facecolor='none', edgecolor='tab:blue', linewidth=2.0, label=f'Expanded (+{int(calc.params["buffer_m"])} m)')
    ax.add_patch(exp_patch)
    ax.add_patch(base_patch)

    # Backbone (if available)
    try:
        backbone = np.array([corridor.backbone.interpolate(prog, normalized=True).xy for prog in np.linspace(0, 1, 100).tolist()])
        ax.plot(backbone[:, 0], backbone[:, 1], '--', c='gold', linewidth=1.5, label='Backbone')
    except Exception:
        pass

    # Ships colored by weight
    if calc.ships_inside:
        xs = np.array([s['x'] for s in calc.ships_inside])
        ys = np.array([s['y'] for s in calc.ships_inside])
        # compute weights for color
        from shapely.geometry import Point
        ws = []
        for s in calc.ships_inside:
            d = corridor.backbone.distance(Point(s['x'], s['y']))
            print(d)
            ws.append(TrafficDensityCalculator.weight_quadratic_clamp(d, d_min, D_max))
        ws = np.array(ws)
        sc = ax.scatter(xs, ys, s=18, c=ws, cmap='viridis', label=f'Ships (color=weight)')
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label('Distance weight w(d)')

    ax.set_aspect('equal', 'box')
    ax.set_title('Steps 1–3: Density from AIS in Expanded Corridor')
    ax.legend(loc='best')
    ax.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()

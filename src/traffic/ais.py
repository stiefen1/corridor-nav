"""
Goal: return a traffic density value given a position in Norway. 
"""


#!/usr/bin/env python3
"""
ais_history_snapshot_basic.py
----------------------------------
Fetch AIS snapshot for an AOI at time T with ±Δ minutes,
and plot vessels on a simple lon/lat scatter (no basemap).

Outputs:
- A matplotlib scatter of ship positions
- Optional CSV with full ship records
"""
import os, csv, concurrent.futures, requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

# ==================== CONFIG ====================
CLIENT_ID     = os.getenv("BW_CLIENT_ID", "adetunjiaduragbemi1@gmail.com:Aduragbemi")
CLIENT_SECRET = os.getenv("BW_CLIENT_SECRET", "UTCecdis123456")
TOKEN_URL     = "https://id.barentswatch.no/connect/token"

HIST_MMSI_URL   = "https://historic.ais.barentswatch.no/v1/historic/mmsiinarea"
HIST_TRACKS_URL = "https://historic.ais.barentswatch.no/v1/historic/tracks/{mmsi}/{from_iso}/{to_iso}"

# AOI (bbox or polygon)
BBOX       = (62.88, 7.61, 63.17, 8.13)  # (minLat, minLon, maxLat, maxLon) 63.50, 9.41, 63.71, 10.13
BBOX_ORDER = "latlon"
POLY_LONLAT: Optional[List[Tuple[float,float]]] = None

T_ISO     = "2025-11-03T12:00:00Z"
DELTA_MIN = 15

SAVE_CSV = False #Turn true to save ship data
CSV_PATH = "ais_snapshot_records.csv"
# ================================================

def get_token() -> str:
    r = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "scope": "ais",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]

def iso(dt_obj: datetime) -> str:
    return dt_obj.replace(tzinfo=timezone.utc).isoformat().replace("+00:00","Z")

def bbox_to_polygon(bbox, order="latlon"):
    if order == "latlon":
        minLat, minLon, maxLat, maxLon = bbox
    else:
        minLon, minLat, maxLon, maxLat = bbox
    ring = [(minLon, minLat),(maxLon, minLat),(maxLon, maxLat),(minLon, maxLat),(minLon, minLat)]
    return Polygon(ring)

def nearest_to_T(points: List[dict], T: datetime):
    best = None; best_dt = None
    for p in points:
        try:
            t = datetime.fromisoformat(p["msgtime"].replace("Z","+00:00"))
        except Exception:
            continue
        d = abs((t - T).total_seconds())
        if best is None or d < best_dt:
            best, best_dt = p, d
    return best

def mmsi_in_area_at_time(token, aoi_geojson, T, delta_min):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "msgtimefrom": iso(T - timedelta(minutes=delta_min)),
        "msgtimeto":   iso(T + timedelta(minutes=delta_min)),
        "polygon": aoi_geojson,
    }
    r = requests.post(HIST_MMSI_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return [str(m) for m in r.json()]

def fetch_track_segment(token, mmsi, T, delta_min):
    headers = {"Authorization": f"Bearer {token}"}
    url = HIST_TRACKS_URL.format(
        mmsi=mmsi,
        from_iso=iso(T - timedelta(minutes=delta_min)),
        to_iso=iso(T + timedelta(minutes=delta_min)),
    )
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()

def snapshot_records(token, aoi_poly, T, delta_min):
    aoi_geojson = {"type": "Polygon", "coordinates": [list(aoi_poly.exterior.coords)]}
    mmsis = mmsi_in_area_at_time(token, aoi_geojson, T, delta_min)
    aoi_prepared = prep(aoi_poly)
    records = []
    if not mmsis:
        return records

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(fetch_track_segment, token, m, T, delta_min): m for m in mmsis}
        for fut in concurrent.futures.as_completed(futures):
            seg = fut.result()
            p = nearest_to_T(seg, T)
            if not p:
                continue
            lat, lon = p.get("latitude"), p.get("longitude")
            if lat is None or lon is None:
                continue
            if not aoi_prepared.contains(Point(float(lon), float(lat))):
                continue
            records.append(p)
    print(records)
    return records

def plot_positions(records: List[dict]):
    if not records:
        print("[info] No ships to plot.")
        return
    lats = [float(r["latitude"]) for r in records]
    lons = [float(r["longitude"]) for r in records]
    plt.figure(figsize=(8,6))
    plt.scatter(lons, lats, s=16, alpha=0.8)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("AIS ship positions (±Δ window)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_records_csv(records: List[dict], path: str):
    if not records:
        print("[info] No records to save.")
        return
    keys = set()
    for r in records: keys.update(r.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(keys))
        writer.writeheader(); [writer.writerow(r) for r in records]
    print(f"[ok] Saved {len(records)} records to {path}")

def main():
    if POLY_LONLAT and len(POLY_LONLAT) >= 3:
        aoi = Polygon(POLY_LONLAT)
    else:
        aoi = bbox_to_polygon(BBOX, BBOX_ORDER)

    token = get_token()
    T = datetime.fromisoformat(T_ISO.replace("Z","+00:00")).astimezone(timezone.utc)

    records = snapshot_records(token, aoi, T, DELTA_MIN)
    print(records)
    print(f"[info] Snapshot records: {len(records)} vessels")

    plot_positions(records)
    if SAVE_CSV:
        save_records_csv(records, CSV_PATH)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

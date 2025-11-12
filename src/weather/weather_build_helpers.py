"""
weather_helpers.py
------------------
Lightweight helper module for getting weather from MET Norway:
- locationforecast (wind, temp, pressure)
- oceanforecast (waves: Hs, Tp)
- currentforecast (ocean currents: speed, direction)
- tiny cache on a rounded (lat,lon,time) grid to avoid API spam
- quiver updater for Matplotlib arrows
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import requests
from matplotlib.quiver import Quiver
from pyproj import Transformer

import os, csv



@dataclass
class WeatherSample:
    # When/where
    lat: float
    lon: float
    requested_time_utc: dt.datetime
    matched_time_utc: Optional[dt.datetime] = None  # the timestamp picked from MET

    # Atmosphere (locationforecast)
    wind_speed: Optional[float] = None          # m/s
    wind_dir_from: Optional[float] = None       # deg (meteorological FROM)
    wind_dir_to: Optional[float] = None         # deg (meteorological TO)
    air_temp: Optional[float] = None            # °C
    pressure: Optional[float] = None            # hPa

    # Ocean (oceanforecast)
    Hs: Optional[float] = None                  # m (significant wave height)
    wave_dir_from: Optional[float] = None       # deg FROM
    wave_dir_to: Optional[float] = None         # deg TO
    sea_temp: Optional[float] = None            # °C

    current_speed: Optional[float] = None       # m/s
    current_dir_to: Optional[float] = None      # deg TO
    current_dir_from: Optional[float] = None    # deg FROM

    def __post_init__(self):
        """
        Convert directions FROM --> TO
        """
        if self.wind_dir_from is not None:
            self.wind_dir_to = (self.wind_dir_from + 180) % 360

        if self.wave_dir_from is not None:
            self.wave_dir_to = (self.wave_dir_from + 180) % 360

        if self.current_dir_from is not None and self.current_dir_to is None:
            self.current_dir_to = (self.current_dir_from + 180) % 360

    def as_dict(self) -> dict:
        return self.__dict__.copy()



# ------------------------------ WeatherClient --------------------------------

@dataclass
class WeatherClient:
    user_agent: str
    mode: str = "met"              # "none" | "met" | "met+ocean" | "met+current" | "met+ocean+current"
    grid_deg: float = 0.05         # ~5.5 km N-S grid
    time_bucket_min: int = 60      # 60 -> hourly
    timeout_s: int = 20
    quiver_alpha: float = 0.8


    # NEW: simple on-disk archive of fetched points
    archive_csv: Optional[str] = None
    archive_write_immediately: bool = True

    # how to fetch: "auto" (try archive, then live), "archive" (csv only), or "live" (API only)
    source: str = "auto"

    # internal cache: (lat_r, lon_r, hour_iso, mode) -> dict
    _cache: Dict[Tuple[float, float, str, str], Dict] = None
    _quiver: Optional[Quiver] = None

    def __post_init__(self):
        if self._cache is None:
            self._cache = {}


    def _ensure_archive(self):
        """Create CSV with a header if it does not exist."""
        if not self.archive_csv:
            return
        if not os.path.exists(self.archive_csv):
            with open(self.archive_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "lat_r","lon_r","time_iso","mode",
                    "wind_speed","wind_dir_from","air_temp","pressure",
                    "Hs","wave_dir_from","sea_temp",
                    "current_speed","current_dir_to","current_dir_from"
                ])

    def _archive_append(self, lat_r, lon_r, hour_iso, mode, payload: Dict):
        """Append one merged weather record to CSV archive."""
        if not self.archive_csv:
            return
        row = [
            lat_r, lon_r, hour_iso, mode,
            payload.get("wind_speed"), payload.get("wind_dir_from"),
            payload.get("air_temp"), payload.get("pressure"),
            payload.get("Hs"), payload.get("wave_dir_from"), payload.get("sea_temp"),
            payload.get("current_speed"), payload.get("current_dir_to"), payload.get("current_dir_from"),
        ]
        with open(self.archive_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def get_from_archive(self, lat: float, lon: float, when_utc: dt.datetime, max_age_hours: float = 1e9) -> Optional[Dict]:
        """
        Return nearest archived record for (lat, lon) time-bucket.
        Uses same rounding/bucketing as live get(). None if not found.
        """
        if not self.archive_csv or not os.path.exists(self.archive_csv):
            return None

        lat_r = self._round(lat, self.grid_deg)
        lon_r = self._round(lon, self.grid_deg)
        tu = when_utc.astimezone(dt.timezone.utc)

        best = None
        best_dt = None
        with open(self.archive_csv, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if float(row["lat_r"]) != lat_r or float(row["lon_r"]) != lon_r:
                    continue
                try:
                    ti = dt.datetime.fromisoformat(row["time_iso"])
                except Exception:
                    continue
                d = abs((ti - tu).total_seconds())
                if best is None or d < best_dt:
                    best, best_dt = row, d

        if best is None or best_dt is None or best_dt > max_age_hours * 3600.0:
            return None

        def _f(x):
            return None if x in (None, "", "None") else float(x)

        return {
            "wind_speed": _f(best["wind_speed"]),
            "wind_dir_from": _f(best["wind_dir_from"]),
            "air_temp": _f(best["air_temp"]),
            "pressure": _f(best["pressure"]),
            "Hs": _f(best["Hs"]),
            "wave_dir_from": _f(best["wave_dir_from"]),
            "sea_temp": _f(best["sea_temp"]),
            "current_speed": _f(best["current_speed"]),
            "current_dir_to": _f(best["current_dir_to"]),
            "current_dir_from": _f(best["current_dir_from"]),
        }



    # --------------------------- Public API ----------------------------------
    
    def getOld(self, when_utc: dt.datetime, north: Optional[float] = None, east: Optional[float] = None, lat: Optional[float] = None, lon: Optional[float] = None) -> WeatherSample:
        """
        Return a WeatherSample with merged MET (and ocean) data for time and (lat, lon) or (north, east).
        Uses small spatial/temporal bucketing to cache results.
        """
        if lat is not None and lon is not None:
            pass
        elif north is not None and east is not None:
            transformer_to_wgs = Transformer.from_crs(32633, 4326, always_xy=True)
            lon, lat = transformer_to_wgs.transform(east, north)
        else:
            raise ValueError(f"either north, east or lat, lon must be specified.")


        if self.mode == "none":
            return WeatherSample(lat=lat, lon=lon, requested_time_utc=when_utc)

        lat_r = self._round(lat, self.grid_deg)
        lon_r = self._round(lon, self.grid_deg)
        hour  = self._time_bucket(when_utc, self.time_bucket_min).isoformat()
        key = (lat_r, lon_r, hour, self.mode)

        # Cache hit?
        if key in self._cache:
            cached = self._cache[key]
            # return as typed object for convenience
            return WeatherSample(lat=lat_r, lon=lon_r, requested_time_utc=when_utc, **cached)

        # Otherwise fetch
        out: Dict[str, Optional[float]] = {}
        matched_time: Optional[dt.datetime] = None

        # Locationforecast (atmosphere)
        if self.mode.startswith("met"):
            met = self._met_locationforecast(lat_r, lon_r, when_utc)
            out.update({
                "wind_speed": met.get("wind_speed"),
                "wind_dir_from": met.get("wind_dir"),
                "air_temp": met.get("air_temp"),
                "pressure": met.get("pressure"),
            })
            # We'll use the same nearest-time logic as ocean; if you want the exact timestamp
            # from locationforecast too, you can extend _met_locationforecast to return it.


        # Oceanforecast (waves + currents + sea temp)
        # if "ocean" in self.mode:
            oc = self._met_oceanforecast(lat_r, lon_r, when_utc)
            out.update({
                    "Hs" : oc.get("Hs"),
                    "wave_dir_from" : oc.get("wave_dir_from"),
                    "sea_temp" : oc.get("sea_temp"),
                    "current_speed" : oc.get("current_speed"),
                    "current_dir_to" : oc.get("current_dir_to"),
                    "current_dir_from" : oc.get("current_dir_from"),
            })

        # Store a lightweight dict in cache; typed wrapper is built on return
        self._cache[key] = out

        sample = WeatherSample(
            lat=lat_r, lon=lon_r,
            requested_time_utc=when_utc,
            matched_time_utc=None,  # you can populate this if you modify _met_pick_nearest to also return the chosen time
            **out
        )
        return sample
    

    def getOld2(self, when_utc: dt.datetime, north: Optional[float] = None, east: Optional[float] = None, lat: Optional[float] = None, lon: Optional[float] = None) -> WeatherSample:
        """
        Return a WeatherSample with merged MET (and ocean) data for time and (lat, lon) or (north, east).
        Uses small spatial/temporal bucketing to cache results.
        Also supports optional CSV archiving and archive playback.
        """
        # --- position handling (existing) ---
        if lat is not None and lon is not None:
            pass
        elif north is not None and east is not None:
            transformer_to_wgs = Transformer.from_crs(32633, 4326, always_xy=True)
            lon, lat = transformer_to_wgs.transform(east, north)
        else:
            raise ValueError(f"either north, east or lat, lon must be specified.")

        if self.mode == "none":
            return WeatherSample(lat=lat, lon=lon, requested_time_utc=when_utc)

        # --- bucketing / cache key ---
        lat_r = self._round(lat, self.grid_deg)
        lon_r = self._round(lon, self.grid_deg)
        hour_dt = self._time_bucket(when_utc, self.time_bucket_min)
        hour_iso = hour_dt.isoformat()
        key = (lat_r, lon_r, hour_iso, self.mode)

        # cache hit?
        if key in self._cache:
            cached = self._cache[key]
            return WeatherSample(lat=lat_r, lon=lon_r, requested_time_utc=when_utc, **cached)

        # if requesting past data, try archive first
        now_utc = dt.datetime.now(dt.timezone.utc)
        if hour_dt < now_utc:
            archived = self.get_from_archive(lat, lon, when_utc)
            if archived is not None:
                self._cache[key] = archived
                return WeatherSample(lat=lat_r, lon=lon_r, requested_time_utc=when_utc, **archived)

        # live fetch
        out: Dict[str, Optional[float]] = {}

        # Atmosphere
        if self.mode.startswith("met"):
            met = self._met_locationforecast(lat_r, lon_r, when_utc)
            out.update({
                "wind_speed": met.get("wind_speed"),
                "wind_dir_from": met.get("wind_dir"),
                "air_temp": met.get("air_temp"),
                "pressure": met.get("pressure"),
            })

        # Ocean (waves + currents + sea temp)
        # if "ocean" in self.mode:
            oc = self._met_oceanforecast(lat_r, lon_r, when_utc)
            out.update({
                "Hs": oc.get("Hs"),
                "wave_dir_from": oc.get("wave_dir_from"),
                "sea_temp": oc.get("sea_temp"),
                "current_speed": oc.get("current_speed"),
                "current_dir_to": oc.get("current_dir_to"),
                "current_dir_from": oc.get("current_dir_from"),
            })

        # cache and archive
        self._cache[key] = out
        if self.archive_csv:
            self._ensure_archive()
            if self.archive_write_immediately:
                self._archive_append(lat_r, lon_r, hour_iso, self.mode, out)

        return WeatherSample(
            lat=lat_r, lon=lon_r,
            requested_time_utc=when_utc,
            matched_time_utc=None,
            **out
        )
    
    def get(
        self,
        when_utc: dt.datetime,
        north: Optional[float] = None,
        east: Optional[float] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None
    ) -> WeatherSample:
        """
        Return a WeatherSample (from CSV, API, or both) depending on `source`:
        - "archive": read CSV only
        - "live":    hit APIs only
        - "auto":    try CSV first; if missing, hit APIs
        """
        # --- position handling ---
        if lat is not None and lon is not None:
            pass
        elif north is not None and east is not None:
            transformer_to_wgs = Transformer.from_crs(32633, 4326, always_xy=True)
            lon, lat = transformer_to_wgs.transform(east, north)
        else:
            raise ValueError("either north, east or lat, lon must be specified.")

        # --- bucketing / cache key ---
        lat_r = self._round(lat, self.grid_deg)
        lon_r = self._round(lon, self.grid_deg)
        hour_dt = self._time_bucket(when_utc, self.time_bucket_min)
        hour_iso = hour_dt.isoformat()
        key = (lat_r, lon_r, hour_iso, self.mode)

        # cache hit?
        if key in self._cache:
            cached = self._cache[key]
            return WeatherSample(lat=lat_r, lon=lon_r, requested_time_utc=when_utc, **cached)

        # --------------------- ARCHIVE PATH ---------------------
        # if source is "archive" or "auto", try CSV first (for any time)
        if self.source in ("archive", "auto"):
            archived = self.get_from_archive(lat, lon, when_utc)
            if archived is not None:
                self._cache[key] = archived
                return WeatherSample(lat=lat_r, lon=lon_r, requested_time_utc=when_utc, **archived)
            if self.source == "archive":
                # strictly archive-only, but nothing found: return empty sample
                return WeatherSample(lat=lat_r, lon=lon_r, requested_time_utc=when_utc)

        # ----------------------- LIVE PATH ----------------------
        # if we are here, we either requested "live" explicitly or "auto" didn't find CSV

        # if mode == "none", the user explicitly disabled live calls
        if self.mode == "none":
            # in "live" this means: cannot fetch; return empty
            # in "auto" we already tried archive; also return empty
            return WeatherSample(lat=lat_r, lon=lon_r, requested_time_utc=when_utc)

        out: Dict[str, Optional[float]] = {}

        # Atmosphere
        if self.mode.startswith("met"):
            met = self._met_locationforecast(lat_r, lon_r, when_utc)
            out.update({
                "wind_speed": met.get("wind_speed"),
                "wind_dir_from": met.get("wind_dir"),
                "air_temp": met.get("air_temp"),
                "pressure": met.get("pressure"),
            })

        # Ocean (waves + currents + sea temp) — proper guard (was commented/mis-indented before)
        # if "ocean" in self.mode:
            oc = self._met_oceanforecast(lat_r, lon_r, when_utc)
            out.update({
                "Hs": oc.get("Hs"),
                "wave_dir_from": oc.get("wave_dir_from"),
                "sea_temp": oc.get("sea_temp"),
                "current_speed": oc.get("current_speed"),
                "current_dir_to": oc.get("current_dir_to"),
                "current_dir_from": oc.get("current_dir_from"),
            })

        # cache and (optionally) archive
        self._cache[key] = out
        if self.archive_csv:
            self._ensure_archive()
            if self.archive_write_immediately:
                self._archive_append(lat_r, lon_r, hour_iso, self.mode, out)

        return WeatherSample(
            lat=lat_r, lon=lon_r,
            requested_time_utc=when_utc,
            matched_time_utc=None,
            **out
        )




    # --------------------------- Internals -----------------------------------

    @staticmethod
    def _round(val: float, step: float) -> float:
        return round(val / step) * step

    @staticmethod
    def _time_bucket(ts: dt.datetime, mins: int) -> dt.datetime:
        t = ts.astimezone(dt.timezone.utc).replace(second=0, microsecond=0)
        return t - dt.timedelta(minutes=t.minute % mins)

    def _met_pick_nearest(self, timeseries: list, target_utc: dt.datetime) -> dict:
        best = None; best_dt = None
        tu = target_utc.astimezone(dt.timezone.utc)
        for item in timeseries:
            ti = dt.datetime.fromisoformat(item["time"].replace("Z", "+00:00"))
            d = abs((ti - tu).total_seconds())
            if best is None or d < best_dt:
                best, best_dt = item, d
        return best
    
    def _request_json(self, url: str, params: dict) -> dict:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json"
        }
        r = requests.get(url, params=params, headers=headers, timeout=self.timeout_s)
        # If not OK, return {} so callers can skip plotting for that layer
        if r.status_code != 200:
            # Optional: print a short diagnostic (comment out if too chatty)
            print(f"[MET] {url} -> HTTP {r.status_code}: {r.text[:120]!r}")
            return {}
        try:
            return r.json()
        except Exception as e:
            print(f"[MET] JSON decode failed for {url}: {e}")
            return {}

    def _met_locationforecast(self, lat: float, lon: float, when_utc: dt.datetime) -> Dict:
        headers = {"User-Agent": self.user_agent}
        url = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
        r = requests.get(url, params={"lat": lat, "lon": lon}, headers=headers, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()["properties"]["timeseries"]
        item = self._met_pick_nearest(data, when_utc)
        details = item["data"]["instant"]["details"]
        return {
            "wind_speed": details.get("wind_speed"),                # m/s
            "wind_dir": details.get("wind_from_direction"),         # deg FROM
            "air_temp": details.get("air_temperature"),
            "pressure": details.get("air_pressure_at_sea_level")
        }


    
    def _met_oceanforecast(self, lat: float, lon: float, when_utc: dt.datetime) -> Dict:
        """
        Waves (Hs, direction), currents (speed, dir), and sea temperature
        from oceanforecast/2.0/complete at the nearest timestamp.
        """
        url = "https://api.met.no/weatherapi/oceanforecast/2.0/complete"
        data = self._request_json(url, {"lat": lat, "lon": lon})
        if not data:
            return {}

        ts = data.get("properties", {}).get("timeseries", [])
        if not ts:
            return {}

        item = self._met_pick_nearest(ts, when_utc)
        det = item.get("data", {}).get("instant", {}).get("details", {})

        out = {
            # Waves
            "Hs": det.get("sea_surface_wave_height"),                   # m
            "wave_dir_from": det.get("sea_surface_wave_from_direction"),# deg (FROM)
            # Temperature
            "sea_temp": det.get("sea_water_temperature"),               # °C
        }

        # Currents (API gives TO-direction; also provide FROM for convenience)
        cur_speed = det.get("sea_water_speed")
        cur_to = det.get("sea_water_to_direction")

        if cur_speed is not None:
            out["current_speed"] = cur_speed                            # m/s
        if cur_to is not None:
            out["current_dir_to"] = cur_to                              # deg (TO)
            out["current_dir_from"] = (float(cur_to) + 180.0) % 360.0   # deg (FROM)

        return out
    


if __name__ == "__main__":
    wc = WeatherClient(
        user_agent="YourApp/1.0 you@example.com",
        mode="met",
        archive_csv="met_stream.csv"   # turn on archiving
    )
    print(wc.get(dt.datetime.now(dt.UTC), lat=60.10, lon=5.00))


# # In your sim loop:
# # t -> current sim UTC time, eta -> has lat/lon or UTM north/east
# sample = wc.get(when_utc=t, lat=lat, lon=lon)        # or north=east= if you use UTM
# # use 'sample' (WeatherSample) in your models

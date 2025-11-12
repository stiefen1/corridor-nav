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
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from pyproj import Transformer


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

        if self.lat is not None and self.lon is not None:
            transformer_to_utm = Transformer.from_crs(4326, 32633, always_xy=True)
            self.east, self.north =  transformer_to_utm.transform(self.lon, self.lat)
            

    def as_dict(self) -> dict:
        return self.__dict__.copy()
    
    def quiver_wind(self, *args, ax: Optional[Axes] = None, **kwargs) -> Axes:
        if ax is None:
            _, ax = plt.subplots()

        if self.wind_dir_to is not None and self.wind_speed is not None:
            wind_vector = self.wind_speed * np.array([
                np.sin(np.deg2rad(self.wind_dir_to)),
                np.cos(np.deg2rad(self.wind_dir_to))
            ])
            ax.quiver(self.east, self.north, wind_vector[0], wind_vector[1], *args, **kwargs)
        return ax
    
    def quiver_current(self, *args, ax: Optional[Axes] = None, **kwargs) -> Axes:
        if ax is None:
            _, ax = plt.subplots()

        if self.current_dir_to is not None and self.current_speed is not None:
            current_vector = self.wind_speed * np.array([
                np.sin(np.deg2rad(self.current_dir_to)),
                np.cos(np.deg2rad(self.current_dir_to))
            ])
            ax.quiver(self.east, self.north, current_vector[0], current_vector[1], *args, **kwargs)
        return ax
    
    def quiver_wave(self, *args, ax: Optional[Axes] = None, **kwargs) -> Axes:
        if ax is None:
            _, ax = plt.subplots()

        if self.wave_dir_to is not None and self.Hs is not None:
            wave_vector = self.Hs * np.array([
                np.sin(np.deg2rad(self.wave_dir_to)),
                np.cos(np.deg2rad(self.wave_dir_to))
            ])
            ax.quiver(self.east, self.north, wave_vector[0], wave_vector[1], *args, **kwargs)
        return ax


# ------------------------------ WeatherClient --------------------------------

@dataclass
class WeatherClient:
    user_agent: str
    mode: str = "met"              # "none" | "met" | "met+ocean" | "met+current" | "met+ocean+current"
    grid_deg: float = 0.05         # ~5.5 km N-S grid
    time_bucket_min: int = 60      # 60 -> hourly
    timeout_s: int = 20
    quiver_alpha: float = 0.8

    # internal cache: (lat_r, lon_r, hour_iso, mode) -> dict
    _cache: Dict[Tuple[float, float, str, str], Dict] = None
    _quiver: Optional[Quiver] = None

    def __post_init__(self):
        if self._cache is None:
            self._cache = {}

    # --------------------------- Public API ----------------------------------
    
    def get(self, when_utc: dt.datetime, north: Optional[float] = None, east: Optional[float] = None, lat: Optional[float] = None, lon: Optional[float] = None) -> WeatherSample:
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
    wc = WeatherClient(user_agent="ecdisAPP/1.0 ecdis@example.com", mode="met")
    print(wc.get(dt.datetime.now(dt.UTC), lat=60.10, lon=5.00))
    # print(wc.get(60.10, 5.00, dt.datetime.utcnow()))


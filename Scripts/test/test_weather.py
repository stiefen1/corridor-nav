"""
Goal: return a current vector given a position in Norway. 

    # Atmosphere (locationforecast)
    wind_speed: Optional[float] = None          # m/s
    wind_dir_from: Optional[float] = None       # deg (meteorological FROM)
    air_temp: Optional[float] = None            # °C
    pressure: Optional[float] = None            # hPa

    # Ocean (oceanforecast)
    Hs: Optional[float] = None                  # m (significant wave height)
    wave_dir_from: Optional[float] = None       # deg FROM
    sea_temp: Optional[float] = None            # °C

    current_speed: Optional[float] = None       # m/s
    current_dir_to: Optional[float] = None      # deg TO
    current_dir_from: Optional[float] = None    # deg FROM
    
"""


import datetime as dt
from weather.weather_helpers import WeatherClient

wc = WeatherClient(user_agent="ecdisAPP/1.0 ecdis@example.com", mode="met")

lat, lon = 60.10, 5.00  # Trondheim
when = dt.datetime.now(dt.UTC)

sample = wc.get(when, lat=lat, lon=lon)
print(sample)
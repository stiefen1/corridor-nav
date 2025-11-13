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

from weather.weather_build_helpers import WeatherClient as wc


lat, lon = 63.15, 7.75  # Alesund

csv_read = True



if csv_read:
    ####### To use the CSV instead #####
    when = dt.datetime(2025, 11, 12, 0, 0, 0)  # YYYY, MM, DD, HH, MM, SS
    wc = wc(
        user_agent="Replay/1.0",
        mode="none",                    # stop hitting live APIs
        source="archive",
        archive_csv="src/weather/data/kristiansund_weather.csv"
    )
    sample = wc.get(when_utc=when, lat=lat, lon=lon)

else:
    when = dt.datetime.now(dt.UTC)
    wc = wc(
        user_agent="ecdisAPP/1.0 ecdis@example.com",
        mode="met",
        source="live",
        archive_csv="src/weather/data/kristiansund_weather.csv"
        )
    sample = wc.get(when, lat=lat, lon=lon)

print(sample)







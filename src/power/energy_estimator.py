from corridor_opt.corridor import Corridor
from weather.weather_helpers import WeatherSample
from power.total_force_estimator import ForceEstimator
from typing import Optional

class EnergyEstimator:
    def __init__(
            self,
            force_estimator: Optional[ForceEstimator] = None
    ):
        self.force_estimator = force_estimator or ForceEstimator()

    def get(self, corridor: Corridor, weather: WeatherSample) -> float:
        return 0
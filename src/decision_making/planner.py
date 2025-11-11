"""
Goal: Provide a corridor-based route planner that compute the optimal sequence of corridors based on weather, traffic density, os states, etc..  
"""

from typing import List, Any, Tuple, Optional, Dict
from corridor_opt.corridor import Corridor
from power.energy_estimator import EnergyEstimator
from power.total_force_estimator import ForceEstimator
from weather.weather_helpers import WeatherSample, WeatherClient
from traffic.ships import TargetShip
from risk.model import RiskModel
import datetime as dt


class Planner:
    def __init__(
        self,
        corridors: List[Corridor],
        goal: Tuple, # One of the main nodes
        energy_estimator: Optional[EnergyEstimator] = None,
        force_estimator: Optional[ForceEstimator] = None,
        weather_client: Optional[WeatherClient] = None,
        ais_client: Optional[Any] = None,
        risk_model: Optional[RiskModel] = None,
        mu: float = 1 # cost function weight: cost = risk + mu * energy_cons
    ):
        self.corridors = corridors
        self.goal = goal
        self.energy_estimator = energy_estimator or EnergyEstimator()
        self.force_estimator = force_estimator or ForceEstimator()
        self.weather_client = weather_client or WeatherClient(user_agent="ecdisAPP/1.0 ecdis@example.com", mode="met")
        self.ais_client = ais_client or ... # TODO: Create AIS object to query data from
        self.risk_model = risk_model or RiskModel()
        self.mu = mu

    def get_optimal_corridor_sequence(
            self,
            u: float, # Surge speed
            when_utc: dt.datetime, # current time
            ship_nominal_maneuverability: float, # Maneuverability without any external forces
            ship_nominal_tracking_accuracy: float, # Tracking accuracy without any external forces
            disable_wave: bool = False
        ) -> List[Corridor]:
        """
        Returns the optimal sequence of corridor to reach self.goal according to a risk model that accounts for target ships, weather and own ship states.
        """
        return []

    def get_costs(
            self,
            u: float, # Surge speed
            when_utc: dt.datetime, # current time
            ship_nominal_maneuverability: float, # Maneuverability without any external forces
            ship_nominal_tracking_accuracy: float, # Tracking accuracy without any external forces
            disable_wave: bool = False
        ) -> Tuple[List[float], Dict]:
        """
        Returns the optimal sequence of corridor to reach self.goal according to a risk model that accounts for target ships, weather and own ship states.
        """

        costs = []
        etts = [] # expected travel times
        ecs = [] # energy consumptions
        # Compute risk for each corridor
        for corridor in self.corridors:
            # Get corridor's coordinates
            east, north = corridor.centroid

            # Get average orientation in the corridor
            psi = corridor.average_orientation()

            # Travel time
            travel_time = corridor.backbone.length / u # distance / speed

            # Get weather sample at corridor's centroid
            weather_sample = self.weather_client.get(when_utc, east=east, north=north)

            # Environmental forces
            forces = self.force_estimator.get(u, psi, weather_sample, degrees=False, disable_wave=disable_wave)

            # Energy consumption
            energy_cons = self.energy_estimator.get(corridor, u, forces, travel_time)

            # Traffic
            traffic_density = 1e-3 # TODO: Evaluate traffic density

            # Risk = Expected travel time
            expected_travel_time = self.risk_model.get(
                travel_time,
                traffic_density,
                forces,
                corridor.width,
                ship_nominal_maneuverability,
                ship_nominal_tracking_accuracy
            )

            cost = expected_travel_time + self.mu * energy_cons
            costs.append(cost)
            etts.append(expected_travel_time)
            ecs.append(self.mu * energy_cons)
            # print(f"Expected travel time: ", expected_travel_time, "nominal travel time: ", travel_time)
        info = {
            'expected_travel_time': etts,
            'energy_consumption': ecs
            }
        return costs, info
    

if __name__ == "__main__":
    pass

"""
Goal: Provide a corridor-based route planner that compute the optimal sequence of corridors based on weather, traffic density, os states, etc..  
"""

from typing import List, Any, Tuple, Optional, Dict, Literal
from corridor_opt.corridor import Corridor
from corridor_opt.corridors_graph import GraphOfCorridors
from power.energy_estimator import EnergyEstimator
from power.total_force_estimator import ForceEstimator
from weather.weather_helpers import WeatherSample, WeatherClient
from traffic.ships import TargetShip
from risk.model import RiskModel
import datetime as dt, networkx as nx


class Planner:
    def __init__(
        self,
        corridors: List[Corridor],
        target_node: int, # One of the main nodes
        energy_estimator: Optional[EnergyEstimator] = None,
        force_estimator: Optional[ForceEstimator] = None,
        weather_client: Optional[WeatherClient] = None,
        ais_client: Optional[Any] = None,
        risk_model: Optional[RiskModel] = None,
        mu: float = 1 # cost function weight: cost = risk + mu * energy_cons
    ):
        self.graph_of_corridors = GraphOfCorridors(corridors)
        self.target_node = target_node
        self.energy_estimator = energy_estimator or EnergyEstimator()
        self.force_estimator = force_estimator or ForceEstimator()
        self.weather_client = weather_client or WeatherClient(user_agent="ecdisAPP/1.0 ecdis@example.com", mode="met", grid_deg=0.01) # TODO: Replace with new version
        # self.ais_client = ais_client or ... # TODO: Create AIS object to query data from
        # res = TrafficDensityCalculator.evaluate_density_for_corridor(
        #     corridor_obj=corridor,
        #     ais_records=records,
        #     to_metric=TO_METRIC,
        #     to_wgs84=TO_WGS,            # optional
        #     buffer_m=BUFFER_M,
        #     d_min=D_MIN,
        #     D_max=D_MAX,
        #     area_shape_factor=AREA_SHAPE_FACTOR,
        #     overlap_fraction=OVERLAP_F,
        #     impute_area_m2=IMPUTE_AREA_M2,
        #     include_boundary=INCLUDE_BOUNDARY,
        #     debug=False,
        # )
        self.risk_model = risk_model or RiskModel()
        self.mu = mu

    def get_optimal_corridor_sequence(
            self,
            start_node: int,
            u: float, # Surge speed
            when_utc: dt.datetime, # current time
            ship_nominal_maneuverability: float, # Maneuverability without any external forces
            ship_nominal_tracking_accuracy: float, # Tracking accuracy without any external forces
            disable_wave: bool = False,
            weight: Literal['risk', 'energy', 'total'] = 'total'
        ) -> Tuple[List, float, List[Corridor]]:
        """
        Returns the optimal sequence of corridor to reach self.goal according to a risk model that accounts for target ships, weather and own ship states.
        """

        self.set_costs(u, when_utc, ship_nominal_maneuverability, ship_nominal_tracking_accuracy, disable_wave=disable_wave)
        path_nodes, total_distance, corridors_used = self.graph_of_corridors.find_shortest_path(start_node, self.target_node, weight=weight)
        return path_nodes, total_distance, corridors_used

    def set_costs(
            self,
            u: float, # Surge speed
            when_utc: dt.datetime, # current time
            ship_nominal_maneuverability: float, # Maneuverability without any external forces
            ship_nominal_tracking_accuracy: float, # Tracking accuracy without any external forces
            disable_wave: bool = False
        ) -> None:
        """
        Returns the optimal sequence of corridor to reach self.goal according to a risk model that accounts for target ships, weather and own ship states.
        """

        # Compute risk for each corridor
        corridors: List[Corridor]
        costs, expected_travel_times, energy_consumption = {}, {}, {}
        edges_corridors = nx.get_edge_attributes(self.graph_of_corridors, 'corridors')
        for edge, corridors in zip(edges_corridors.keys(), edges_corridors.values()):
            # cost_edge = 0
            # expected_travel_time_edge = 0
            # energy_cons_edge = 0
            for corridor in corridors:
                # Get corridor's coordinates
                east, north = corridor.centroid

                # Get average orientation in the corridor
                psi = corridor.average_orientation()

                # Travel time
                travel_time = corridor.backbone.length / u # distance / speed

                # Get weather sample at corridor's centroid
                weather_sample = self.weather_client.get(when_utc, east=east, north=north) # TODO: Replace with new version

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
                costs.update({
                    corridor: cost
                })
                expected_travel_times.update({
                    corridor: expected_travel_time
                })
                energy_consumption.update({
                    corridor: energy_cons
                })

                # cost_edge += cost
                # expected_travel_time_edge += expected_travel_time
                # energy_cons_edge += energy_cons

        self.graph_of_corridors.update_multiple_corridor_weights({
            'total': costs,
            'risk': expected_travel_times,
            'energy': energy_consumption
        })
    

if __name__ == "__main__":
    pass

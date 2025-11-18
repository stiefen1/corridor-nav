"""
Goal: Provide a corridor-based route planner that compute the optimal sequence of corridors based on weather, traffic density, os states, etc..  
"""

from typing import List, Any, Tuple, Optional, Dict, Literal
from corridor_opt.corridor import Corridor
from corridor_opt.corridors_graph import GraphOfCorridors
from power.energy_estimator import EnergyEstimator
from power.total_force_estimator import ForceEstimator
from weather.weather_build_helpers import WeatherClient
from traffic.ships import TargetShip
from risk.model import RiskModel
import datetime as dt, networkx as nx, numpy as np
from traffic.traffic_density_eval import TrafficDensityCalculator


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
        # mu: float = 1, # cost function weight: cost = risk + mu * energy_cons
        operational_cost_per_hour: float = 1e1 * 3600, # $ / hour
        cost_of_energy: float = 1e2, # $ / kWh
        weather_path: str = 'src/weather/data/kristiansund_weather.csv'
    ):
        self.graph_of_corridors = GraphOfCorridors(corridors)
        self.records = ais_client
        self.target_node = target_node
        self.energy_estimator = energy_estimator or EnergyEstimator()
        self.force_estimator = force_estimator or ForceEstimator()
        self.weather_client = weather_client or WeatherClient(user_agent="Replay/1.0", mode="none", source="archive", archive_csv=weather_path) # mode = met, source = live for live data
        self.risk_model = risk_model or RiskModel()
        # self.mu = mu
        self.operational_cost_per_hour = operational_cost_per_hour
        self.cost_of_energy = cost_of_energy


    def print_statistics(
            self,
            u: float, # Surge speed
            when_utc: dt.datetime, # current time
            disable_wave: bool = False,
    ) -> None:
        values = {
            "torque": [],
            "sway_force": [],
            "traffic": [],
            "width": []
        }
        for corridor in self.graph_of_corridors.corridors:
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


            # Traffic
            traffic_density = TrafficDensityCalculator.evaluate_density_for_corridor(corridor_obj=corridor, ais_records=self.records)['density']
            values["torque"].append(abs(forces[2]))
            values["sway_force"].append(abs(forces[1]))
            values["traffic"].append(traffic_density)
            values["width"].append(corridor.width)

        for key, val in zip(values.keys(), values.values()):
            print(f"{key}:\t Mean: {np.mean(val)}\t Mean-std: {np.mean(val)-np.std(val)}\t Mean+std: {np.mean(val)+np.std(val)}\t min: {np.min(val)}\t max: {np.max(val)}")



    def get_optimal_corridor_sequence(
            self,
            start_node: int,
            u: float, # Surge speed
            when_utc: dt.datetime, # current time
            disable_wave: bool = False,
            weight: Literal['risk', 'energy', 'total'] = 'total'
        ) -> Tuple[List, float, List[Corridor]]:
        """
        Returns the optimal sequence of corridor to reach self.goal according to a risk model that accounts for target ships, weather and own ship states.
        """

        self.set_costs(u, when_utc, disable_wave=disable_wave)
        path_nodes, total_distance, corridors_used = self.graph_of_corridors.find_shortest_path(start_node, self.target_node, weight=weight)
        return path_nodes, total_distance, corridors_used

    def set_costs(
            self,
            u: float, # Surge speed
            when_utc: dt.datetime, 
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
                weather_sample = self.weather_client.get(when_utc, east=east, north=north)

                # Environmental forces
                forces = self.force_estimator.get(u, psi, weather_sample, degrees=False, disable_wave=disable_wave)

                # Energy consumption
                energy_cons = self.energy_estimator.get(corridor, u, forces, travel_time)

                # Traffic
                traffic_density = TrafficDensityCalculator.evaluate_density_for_corridor(corridor_obj=corridor, ais_records=self.records)['density']

                # Risk = Expected travel time
                # expected_travel_time = self.risk_model.get(
                #     travel_time,
                #     traffic_density,
                #     forces,
                #     corridor.width
                # )
                risk_cost, prob = self.risk_model.get(
                    travel_time,
                    traffic_density,
                    forces,
                    corridor.width
                )

                # cost = expected_travel_time + self.mu * energy_cons
                expected_op_cost = (1-prob) * travel_time * (self.operational_cost_per_hour / 3600)
                expected_energy_cost = (1-prob) * self.cost_of_energy * (energy_cons / 3.6e6)
                cost = risk_cost + expected_op_cost + expected_energy_cost

                # print(
                #     risk_cost, prob,
                #     expected_energy_cost, self.cost_of_energy,
                #     expected_op_cost, self.operational_cost_per_second, travel_time
                # )

                costs.update({
                    corridor: cost
                })
                expected_travel_times.update({
                    corridor: risk_cost
                })
                energy_consumption.update({
                    corridor: expected_energy_cost
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

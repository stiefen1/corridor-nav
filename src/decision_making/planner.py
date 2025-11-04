"""
Goal: Provide a corridor-based route planner that compute the optimal sequence of corridors based on weather, traffic density, os states, etc..  
"""

from typing import List, Any, Tuple
from corridor_opt.obstacle import Obstacle

class Planner:
    def __init__(
        self,
        corridors: List[Obstacle],
        goal: Tuple # One of the main nodes
    ):
        self.corridors = corridors
        self.goal = goal

    def get_optimal_corridor_sequence(self, os: Any, traffic_density: Any, weather: Any) -> List[Obstacle]:
        """
        Returns the optimal sequence of corridor to reach self.goal according to a risk model that accounts for target ships, weather and own ship states.
        """
        return self.corridors

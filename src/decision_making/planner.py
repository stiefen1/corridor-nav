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

    def get_optimal_corridor_sequence(self, os: Any, target_ships: List, weather: List) -> List[Obstacle]:
        """
        Returns the optimal sequence of corridor to reach self.goal according to a risk model that accounts for target ships, weather and own ship states.
        """
        return self.corridors

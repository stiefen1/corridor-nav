from pso.cost import CostBase
from shapely import LineString
import numpy as np
from typing import List, Tuple
from corridor_opt.obstacle import Obstacle
from corridor_opt.corridor_utils import get_rectangle_and_bend_from_progression_and_width
from pso import PSO

DEFAULT_LBX = (0.01, -0.99) # progression, width change 
DEFAULT_UBX = (1, 0.0)
class CorridorCostProgAndLimitedWidth(CostBase):
    def __init__(
            self,
            edge: LineString,
            max_width: float,
            obstacles: List[Obstacle],
            lbx_for_rescaling: np.ndarray,
            ubx_for_rescaling: np.ndarray,
            *args,
            edge_prev: LineString | None = None,
            l_offset: float = 0.1,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.edge = edge
        self.edge_prev = edge_prev
        self.max_width = max_width
        self.obstacles = obstacles
        self.l_offset = l_offset
        self.lbx_for_rescaling = lbx_for_rescaling
        self.ubx_for_rescaling = ubx_for_rescaling

    def eval(self, progression:float, width_change:float, *args, normalized:bool=True, **kwargs) -> float:
        """
        angle, length and width are within [0, 1] to avoid issues with different units among dimensions.
        
        """
        if normalized:
            normalized_progression, normalized_width_change = progression, width_change
            normalized_particle = np.array([progression, width_change])
            scaled_particle = normalized_particle * (self.ubx_for_rescaling - self.lbx_for_rescaling) + self.lbx_for_rescaling # Does not change anything for progression since it is in [0, 1]
            progression, width_change = scaled_particle.tolist()
        else:
            particle = np.array([progression, width_change])
            normalized_particle = (particle - self.lbx_for_rescaling) / (self.ubx_for_rescaling - self.lbx_for_rescaling)
            normalized_progression, normalized_width_change = normalized_particle.tolist()

        width = self.max_width * (1 + width_change)
        # print(width, progression)
        out = get_rectangle_and_bend_from_progression_and_width(self.edge, progression, width, edge_prev=self.edge_prev, length_margin=self.l_offset)     
        
        if out is None: # Invalid geometry
            return 10000
        else:
            _, bend = out
        
        # Penalty for intersection with obstacles
        for obs in self.obstacles:
            if obs.intersects(bend):
                cost = 1000*obs.intersection(bend).area + 2000 # I need discontinuity to really make collision worst than collision-free
                return cost

        # Reward for path progression and width
        cost = 10*np.exp(-(normalized_progression**2 + 2 * normalized_width_change)) 
        
        return cost 

class CorridorPSOProgAndLimitedWidth(PSO):
    def __init__(
            self,
            edge:LineString,
            max_width:float,
            obstacles:List[Obstacle],
            *args,
            edge_prev: LineString | None = None,
            lbx: Tuple = DEFAULT_LBX, # Constraints particles to remain within [lbx, ubx] along optimization
            ubx: Tuple = DEFAULT_UBX,
            l_offset: float = 0.1,
            **kwargs
    ):
        self.lbx_for_rescaling = np.array(lbx)
        self.ubx_for_rescaling = np.array(ubx)
        self.max_width = max_width
        super().__init__(
            cost=CorridorCostProgAndLimitedWidth(edge, max_width, obstacles, self.lbx_for_rescaling, self.ubx_for_rescaling, edge_prev=edge_prev, l_offset=l_offset),
            lbx=(0, 0),
            ubx=(1, 1),
            *args,
            **kwargs
        )

    def get_rescaled_optimal_position(self) -> np.ndarray:
        return self.get_optimal_position() * (self.ubx_for_rescaling - self.lbx_for_rescaling) + self.lbx_for_rescaling

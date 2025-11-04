from pso.cost import CostBase
from shapely import LineString
import numpy as np
from typing import List
from corridor_opt.obstacle import Obstacle
from corridor_opt.corridor_utils import get_bend_obstacle

class CorridorCostProgAndBendRadius(CostBase):
    def __init__(
            self,
            edge:LineString,
            width:float,
            obstacles:List[Obstacle],
            lbx_for_rescaling:np.ndarray,
            ubx_for_rescaling:np.ndarray,
            *args,
            l_offset:float=0.1,
            edge_prev:LineString | None = None,
            width_prev:float | None = None,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.edge = edge
        self.width = width
        self.obstacles = obstacles
        self.l_offset = l_offset
        self.lbx_for_rescaling = lbx_for_rescaling
        self.ubx_for_rescaling = ubx_for_rescaling
        self.edge_prev = edge_prev
        self.width_prev = width_prev

    def eval(self, progression:float, relative_radius:float, *args, normalized:bool=True, **kwargs) -> float:
        """
        
        """
        if normalized:
            normalized_progression, normalized_relative_radius = progression, relative_radius
            normalized_particle = np.array([progression, relative_radius])
            scaled_particle = normalized_particle * (self.ubx_for_rescaling - self.lbx_for_rescaling) + self.lbx_for_rescaling # Does not change anything for progression since it is in [0, 1]
            progression, relative_radius = scaled_particle.tolist()
        else:
            particle = np.array([progression, relative_radius])
            normalized_particle = (particle - self.lbx_for_rescaling) / (self.ubx_for_rescaling - self.lbx_for_rescaling)
            normalized_progression, normalized_relative_radius = normalized_particle.tolist()

        r = get_bend_obstacle(self.edge, progression, self.width, margin=self.l_offset, edge_prev=self.edge_prev, width_prev=self.width_prev, radius=relative_radius*self.width)
        # print("\nr: ", r._geometry)
        if r is None:
            return 3000

        # Penalty for intersection with obstacles
        for obs in self.obstacles:
            if obs.intersects(r):
                cost = 1000*obs.intersection(r).area + 3000 # I need discontinuity to really make collision worst than collision-free

                return cost # 10000

        # Reward for path progression and width
        cost = 10*np.exp(-(2*normalized_progression**2 + normalized_relative_radius)) # A bigger radius means shorter path
        
        return cost
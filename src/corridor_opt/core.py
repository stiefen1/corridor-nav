from typing import List, Tuple
from corridor_opt.obstacle import Obstacle, Rectangle
from corridor_opt.corridor_utils import get_rectangle_from_progression_and_width, merge_list_of_corridors_with_bend
from corridor_opt.graph import get_min_distances_to_edges
from corridor_opt.pso_opt import CorridorPSOProgAndLimitedWidth
from shapely import LineString
import numpy as np

def build_corridors_graph(
        edges_as_linestring:List[LineString],
        obstacles:List[Obstacle],
        n_particles:int=50, max_iter:float=100, inertia:float=0.5, c_cognitive:float=0.2, c_social:float=0.5,
        distance_margin:float=15, # Minimum sideway distance from a corridor to the shore -> must be equal to the safety radius of own vessel
        min_corridor_width:float=10, # Minimal width that we expect a ship to be able to track
        max_corridor_width:float=100,
        **kwargs
        ) -> List[Rectangle]:
    
    min_distances_to_edges = get_min_distances_to_edges(edges_as_linestring, obstacles)

    # Corridor optimization (PSO)
    corridors:List[Rectangle] = []

    for i, (edge, dist) in enumerate(zip(edges_as_linestring, min_distances_to_edges)):
        
        if dist <= 0.0:
            print(f"edge {i} is intersecting obstacles (distance={dist:.1f}), skipping..")
            continue

        min_passage_width = min(2 * dist, max_corridor_width) # 0.8 # 2 * dist * 0.9
        if min_passage_width - 2 * distance_margin <= min_corridor_width:
            print(f"edge {i} is too narrow (min passage width={min_passage_width:.1f}, margin must be {2*distance_margin:.1f}), skipping..")
            continue
        
        
        progression = 0
        temp_corridors = []
        valid = True
        while progression < 1:
            sliced_edge = LineString(edge.interpolate(np.linspace(progression, 1, 30), normalized=True))
            pso = CorridorPSOProgAndLimitedWidth(
                sliced_edge,
                min_passage_width,
                obstacles,
                n_particles=n_particles,
                max_iter=max_iter,
                inertia=inertia,
                c_cognitive=c_cognitive,
                c_social=c_social,
                stop_at_variance=1e-4,
                lbx=np.array((0.01, ((min_corridor_width + 2 * distance_margin)/(min_passage_width))-1.0)), # lower bound for (progression, width)
            )
            # print(f"{i} length: ", sliced_edge.length)
            # print("variance before: ", pso._swarm.get_variance())
            pso.optimize()

            opt = pso.get_rescaled_optimal_position()
            # print(f"edge {i+1}/{len(edges_as_linestring)} (cost={pso.get_optimal_cost():.1f} | width={width:.1f}) - progression: {opt[0]:.1f} | width change: {opt[1]*100:.1f}%")            
            # print("variance after: ", pso._swarm.get_variance(), " iter: ", pso._swarm.iter)
            prog_opt, width_change_opt = opt
            width_with_margin = min_passage_width * (1 + width_change_opt) - 2 * distance_margin
            
            if pso.get_optimal_cost() > 1e3 or width_with_margin <= 0:
                print(f"invalid corridor, skipping...")
                valid = False
                break

            corridor = get_rectangle_from_progression_and_width(sliced_edge, prog_opt, width_with_margin)
            print(f"({i+1}/{len(edges_as_linestring)}) Corridor width: {width_with_margin:.2f}, width_change_opt: {width_change_opt}, max_corridor_width: {max_corridor_width}, min passage width: {min_passage_width} smin: {(min_corridor_width/min_passage_width)-1}")
            temp_corridors.append(corridor)
            progression += prog_opt * (1-progression)
            

        if valid:
            corridors.extend(temp_corridors)

    return corridors

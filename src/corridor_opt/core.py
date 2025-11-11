from typing import List, Tuple, Optional
from corridor_opt.obstacle import Obstacle, Rectangle
from corridor_opt.corridor_utils import get_rectangle_and_bend_from_progression_and_width
from corridor_opt.graph import get_min_distances_to_edges
from corridor_opt.pso_opt import CorridorPSOProgAndLimitedWidth
from shapely import LineString
import numpy as np
from colorama import Fore
from corridor_opt.corridor import Corridor

def build_corridors_graph(
        edges_as_linestring: List [LineString],
        main_nodes_ids: List[Tuple[int, int]],
        obstacles: List [Obstacle],
        n_particles: int = 50, max_iter: float = 100, inertia: float = 0.5, c_cognitive: float = 0.2, c_social: float = 0.5, stop_at_variance: float = 1e-4,
        distance_margin: float = 15, # Minimum sideway distance from a corridor to the shore -> must be equal to the safety radius of own vessel
        min_corridor_width: float = 10, # Minimal width that we expect a ship to be able to track
        max_corridor_width: float = 100,
        length_margin: float = 0.1, 
        save_folder: Optional[str] = None,
        **kwargs
        ) -> List[Corridor]:
    
    min_distances_to_edges = get_min_distances_to_edges(edges_as_linestring, obstacles)

    # Corridor optimization (PSO)
    corridors:List[Corridor] = []

    for i, (edge, main_node_id, dist) in enumerate(zip(edges_as_linestring, main_nodes_ids, min_distances_to_edges)):
        
        if dist <= 0.0:
            print(f"edge {i} is intersecting obstacles (distance={dist:.1f}), skipping..")
            continue

        min_passage_width = min(2 * dist, max_corridor_width) # 0.8 # 2 * dist * 0.9
        if min_passage_width - 2 * distance_margin <= min_corridor_width:
            print(f"edge {i} is too narrow (min passage width={min_passage_width:.1f}, margin must be {2*distance_margin:.1f}), skipping..")
            continue
        
        
        progression = 0
        temp_corridors: List[Corridor] = []
        valid = True
        edge_prev = None
        subsegment_index = 0
        retry_iter = 0
        max_retry_iter = 5
        while progression < 1:
            sliced_edge = LineString(edge.interpolate(np.linspace(progression, 1, 30), normalized=True))
            max_corridor_width_retry = max_corridor_width * 0.8**retry_iter
            pso = CorridorPSOProgAndLimitedWidth(
                sliced_edge,
                max_corridor_width_retry + 2 * distance_margin, # Maximum width allowed for a corridor
                obstacles,
                edge_prev=edge_prev,
                l_offset=length_margin,
                n_particles=n_particles,
                max_iter=max_iter,
                inertia=inertia,
                c_cognitive=c_cognitive,
                c_social=c_social,
                stop_at_variance=stop_at_variance,
                lbx = (0.05, (min_corridor_width + 2 * distance_margin) / (max_corridor_width_retry + 2 * distance_margin) - 1)
            )

            pso.optimize()
            opt = pso.get_rescaled_optimal_position()
            prog_opt, width_change_opt = opt
            width_opt = (max_corridor_width_retry + 2 * distance_margin) * (1 + width_change_opt)
            width_opt_with_margin = width_opt - 2 * distance_margin
            out = get_rectangle_and_bend_from_progression_and_width(sliced_edge, prog_opt, width_opt, edge_prev=edge_prev, length_margin=length_margin, width_margin=2*distance_margin)
            
            if pso.get_optimal_cost() > 1e3 or width_opt_with_margin <= 0:
                print(Fore.MAGENTA + f"invalid corridor ({pso.swarm.n_iter}/{max_iter} iterations) for max width = {max_corridor_width_retry:.1f}")
                
                if retry_iter >= max_retry_iter:
                    print(Fore.RED + f"max retry iter reached, skipping..")
                    valid = False
                    break
                else:
                    retry_iter += 1
                    print(Fore.BLUE + f"retrying with smaller width (iter {retry_iter+1}/{max_retry_iter})")


            elif out is not None:
                _, bend, info = out
                print(Fore.GREEN + "".join(subsegment_index*["\t"]) + f"({i+1}.{subsegment_index}/{len(edges_as_linestring)}) width: {min_corridor_width:.1f} <= {width_opt_with_margin:.1f} <= {max_corridor_width_retry:.1f}")
                
                new_corridor = Corridor(**info, prev_main_node=main_node_id[1], next_main_node=main_node_id[0], idx=subsegment_index, edge_id=i)
                
                temp_corridors.append(new_corridor)
                # temp_corridors.append(bend)
                retry_iter = 0

                delta_prog = prog_opt * (1-progression)
                if progression + delta_prog < 1:
                    delta_prog *= 0.97 # Ensure overlap if prog_opt does not lead to final main node

                progression += delta_prog
                if progression < 1:
                    edge_prev = LineString(edge.interpolate(np.linspace((progression-prog_opt)/(1-prog_opt), progression, 30), normalized=True))
            else:
                print(Fore.RED + f"Optimal solution leads to invalid shape, skipping...")
                valid = False
                break

            if valid:
                if progression < 1:
                    print(Fore.BLUE + "".join(subsegment_index*["\t"]) + f"------> Working on next sub-segment of edge {i+1}")
                    subsegment_index += 1
                else:
                    print("")


        if valid:
            for corridor in temp_corridors:
                corridor.export(folder=save_folder)
            corridors.extend(temp_corridors)

    return corridors

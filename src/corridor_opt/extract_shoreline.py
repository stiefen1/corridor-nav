from seacharts.enc import ENC
from typing import Tuple, List
from corridor_opt.obstacle import Obstacle
from shapely import Polygon, MultiPolygon, difference, intersection

def get_obstacles_in_window(enc: ENC, center: Tuple | None = None, size: Tuple | None = None, depth: int | List[int] = 0) -> List[Obstacle]:
    """
    If no size or center are provided, the obstacles will be collected according to the info available in seacharts config file.
    Specifying size and/or center allows the user to keep only obstacles from a specific region
    """
    # Convert depth into a list
    if isinstance(depth, list):
        pass
    else:
        depth = [depth]

    lx_enc, ly_enc = enc.size
    x0_enc, y0_enc = enc.center
    window_complete = Polygon(((x0_enc-lx_enc/2, y0_enc-ly_enc/2), (x0_enc+lx_enc/2, y0_enc-ly_enc/2), (x0_enc+lx_enc/2, y0_enc+ly_enc/2), (x0_enc-lx_enc/2, y0_enc+ly_enc/2)))

    size_focus = size or enc.size 
    center_focus = center or enc.center
    if center_focus is None:
        # meaning both center self.enc.center are None -> We have to compute it from self.enc.origin & self.enc.size
        xo, yo = enc.origin
        sx, sy = enc.size
        center_focus = (xo + sx/2, yo+sy/2)
    
    lx_focus, ly_focus = size_focus
    x0_focus, y0_focus = center_focus
    window_focus = Polygon(((x0_focus-lx_focus/2, y0_focus-ly_focus/2), (x0_focus+lx_focus/2, y0_focus-ly_focus/2), (x0_focus+lx_focus/2, y0_focus+ly_focus/2), (x0_focus-lx_focus/2, y0_focus+ly_focus/2)))

    obstacles = [] # Initialize list of obstacles representing the shore

    # Takes first depth_i that is greater or equal to depth
    # for depth_i, seabed in zip(self.enc.seabed.keys(), self.enc.seabed.values()):
    intersection_window_area = intersection(window_complete, window_focus).area

    for depth_i in depth:
        if depth_i in enc.seabed.keys():
            seabed = enc.seabed[depth_i]
            list_of_polygons = [seabed.geometry] if isinstance(seabed.geometry, Polygon) else list(seabed.geometry.geoms)
            for polygon in list_of_polygons:
                multi_diff = intersection(difference(window_complete, polygon), window_focus) # We make sure that the resulting polygon is both part of enc and focus regions
                if intersection_window_area != 0:
                    print(f"{intersection_window_area:.0f}, {multi_diff.area:.0f}, {multi_diff.area/intersection_window_area:.3f}")
                if 0.99*intersection_window_area <= multi_diff.area: # For some reasons, window_focus is sometimes the result of it
                        continue
                
                # Difference can lead to multiple polygons. In such case we add them all to the obstacles collection
                if isinstance(multi_diff, MultiPolygon):
                    cumsum = 0 
                    for diff in multi_diff.geoms:
                        cumsum += diff.area
                        obstacles.append(Obstacle(polygon=diff, depth=depth_i))
                            
                # If difference is a single polygon, we just add it to the collection
                elif isinstance(multi_diff, Polygon):
                    obstacles.append(Obstacle(polygon=multi_diff, depth=depth_i))
        else:
            print(f"Map - Depth {depth_i}m not found in ENC data - Use .available_depth_data to select an existing depth layer.")
        
    return obstacles

if __name__ == "__main__":
    import os
    from corridor_opt.obstacle import Rectangle
    import matplotlib.pyplot as plt

    r = Rectangle(10, 0, 10, 3).rotate(30)
    ax = r.plot()
    ax.set_aspect('equal')
    plt.show()
    plt.close()

    config_path = os.path.join('config', 'trondelag_corridors.yaml')
    enc = ENC(config_path)
    enc.display.start()
    enc.display.show()
from shapely import LineString
from corridor_opt.obstacle import Obstacle, Rectangle
import numpy as np
from typing import Dict
from corridor_opt.math_utils import normalize_angle_0_2pi


def get_rectangle_from_progression_and_width(edge:LineString, progression:float, width:float, margin:float=0.1) -> Rectangle:
    """
    rectangle starting from edge.interpolate(0) to edge.interpolate(progression) with a given width. margin is the additional length
    on each side of the rectangle, meaning the final rectange length will be || edge.interpolate(progression) - edge.interpolate(0) || + 2*margin
    """
    start = np.array(edge.interpolate(0).xy).squeeze()
    stop = np.array(edge.interpolate(progression, normalized=True).xy).squeeze()
    length = np.linalg.norm(stop-start) + 2*margin
    angle = np.atan2(start[0]-stop[0], stop[1]-start[1])
    return Rectangle(*(start + np.array([0, length/2-margin])), length, width).rotate(angle, start, use_radians=True)


def get_bend(edge:LineString, prog:float, width:float, edge_prev:LineString, width_prev:float, margin:float=0.3, radius: float | None = None) -> Dict:
    # Get rectangles
    r_prev = get_rectangle_from_progression_and_width(edge_prev, progression=1, width=width_prev, margin=margin)
    r = get_rectangle_from_progression_and_width(edge, progression=prog, width=width, margin=margin)

    # Extract corners of interest
    p1_lower_right = r_prev.lower_right_corner[:, None]
    p1_lower_left = r_prev.lower_left_corner[:, None]
    p2_upper_right = r.upper_right_corner[:, None]
    p2_upper_left = r.upper_left_corner[:, None]
    
    # Direction vectors
    anchor_prev = np.array(edge_prev.interpolate(0).xy)
    anchor = np.array(edge_prev.interpolate(1, normalized=True).xy)
    direction_prev = anchor - anchor_prev
    theta1 = np.atan2(direction_prev[0], direction_prev[1]).squeeze()
    
    direction = np.array(edge.interpolate(prog, normalized=True).xy) - anchor
    theta2 = np.atan2(direction[0], direction[1]).squeeze()
    
    # Normalize angles and determine orientation
    delta_normalized = normalize_angle_0_2pi(theta2-theta1)
    theta1 = normalize_angle_0_2pi(theta1)
    theta2 = normalize_angle_0_2pi(theta2)

    orient = -1 if 0 <= delta_normalized <= np.pi else 1
    # idx = 1 if orient < 0 else 2
    if orient < 0:
        p1 = r_prev.lower_right_corner[:, None]
        p2 = r.lower_right_corner[:, None]
    else:
        p1 = r_prev.lower_left_corner[:, None]
        p2 = r.lower_left_corner[:, None]
    
    # Check if direction vectors are parallel
    det = direction_prev[0] * (-direction[1]) - direction_prev[1] * (-direction[0])
    if abs(det) < 1e-6: # Parallel lines, no intersection
        return None
    
    # Intersection
    s12 = np.linalg.inv(np.hstack([direction_prev, -direction])) @ (p2 - p1)
    intersection = p1 + s12[0] * direction_prev

    # Angles
    theta_max = np.max([theta1, theta2])
    theta_min = np.min([theta1, theta2])
    delta_max_min = normalize_angle_0_2pi(theta_max - theta_min)
    alpha = normalize_angle_0_2pi(np.pi - delta_max_min)
    
    # Compute L (distance from center of bend to intersection)
    width_average = (width_prev + width)/2
    radius =  radius or 2 * abs(-width_average * normalize_angle_0_2pi(theta2-theta1) / np.pi + width_average)
    l = radius / np.cos((np.pi-alpha)/2)
    
    # Bend center direction vector
    d_bend = -orient*np.array([np.sin(theta_max+alpha/2), np.cos(theta_max+alpha/2)])[:, None]
    
    # Center of bend
    center: np.ndarray = intersection + l * d_bend
    d = np.linalg.norm(anchor.flatten() - center.flatten())
    gamma = min(normalize_angle_0_2pi(np.pi+theta1-theta2), normalize_angle_0_2pi(np.pi+theta2-theta1))
    
    # Length of corridors
    # print(d, gamma, r_prev.height, r.height, margin)
    # print(d * np.cos(gamma/2), min(r_prev.height, r.height) + margin)
    if d * np.cos(gamma/2) > min(r_prev.height, r.height) - margin:
        return None
    
    # Arc parameters
    if orient > 0:
        if 0 <= theta1 <= np.pi: ### EVERYTHING WORKS HERE
            if 0 <= theta2 <= np.pi:
                central_angle =  normalize_angle_0_2pi(- theta2 + alpha/2 - np.pi)
                theta1_arc_rad = ( central_angle + alpha/2 )
                theta2_arc_rad = ( central_angle - alpha/2 + np.pi )
            else:
                central_angle = normalize_angle_0_2pi(- theta2 - alpha/2)
                theta2_arc_rad = ( central_angle + alpha/2 )
                theta1_arc_rad = ( central_angle - alpha/2 + np.pi )

        else: ### EVERYTHING WORKS HERE
            central_angle =  normalize_angle_0_2pi(- theta2 + alpha/2 - np.pi)
            theta1_arc_rad = ( central_angle + alpha/2 )
            theta2_arc_rad = ( central_angle - alpha/2 + np.pi )
    else:
        if 0 <= theta1 <= np.pi: ### EVERYTHING WORKS HERE
            central_angle = normalize_angle_0_2pi(- theta2 - alpha/2 - np.pi/2)
            theta1_arc_rad = ( central_angle + alpha/2 - np.pi/2 )
            theta2_arc_rad = ( central_angle - alpha/2 + np.pi/2 )

        else:
            if 0 <= theta2 <= np.pi:
                central_angle =  normalize_angle_0_2pi(- theta2 + alpha/2)
                theta2_arc_rad = ( central_angle + alpha/2 )
                theta1_arc_rad = ( central_angle - alpha/2 + np.pi )
            else:
                central_angle = normalize_angle_0_2pi(- theta2 - alpha/2 + np.pi)
                theta1_arc_rad = ( central_angle + alpha/2 )
                theta2_arc_rad = ( central_angle - alpha/2 + np.pi )

    # Build complete corridor shape
    list_of_thetas = np.linspace(theta1_arc_rad, theta2_arc_rad, num=10)
    inner_arc = np.fliplr(center + radius * np.array([np.cos(list_of_thetas), np.sin(list_of_thetas)]))
    outter_arc = center + (radius+width_average) * np.array([np.cos(list_of_thetas), np.sin(list_of_thetas)])
    return {
            'p1_lower_right': p1_lower_right,
            'p1_lower_left': p1_lower_left,
            'p2_upper_right': p2_upper_right,
            'p2_upper_left': p2_upper_left,
            'inner_arc': inner_arc,
            'outter_arc': outter_arc,
            'orientation': orient
    }

def get_bend_obstacle(edge:LineString, prog:float, width:float, edge_prev:LineString=None, width_prev:float=None, margin:float=0.3, radius:float=None):
    """Calculate all the bend geometry based on current waypoints"""
    if edge_prev is None:
        return get_rectangle_from_progression_and_width(edge, prog, width, margin=margin)
    
    # Get bend: if not valid then return None
    bend = get_bend(edge, prog, width, edge_prev, width_prev, margin=margin, radius=radius)
    if bend is None:
        return None
    
    if bend['orientation'] > 0:
        corridor_with_bend = np.hstack([bend['p1_lower_right'], bend['outter_arc'], bend['p2_upper_right'], bend['p2_upper_left'], bend['inner_arc'], bend['p1_lower_left'], bend['p1_lower_right']])
    else:
        corridor_with_bend = np.hstack([bend['p1_lower_right'], bend['inner_arc'], bend['p2_upper_right'], bend['p2_upper_left'], bend['outter_arc'], bend['p1_lower_left'], bend['p1_lower_right']])

    return Obstacle(zip(*corridor_with_bend.tolist()))
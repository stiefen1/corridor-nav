from shapely import LineString, Polygon
from corridor_opt.obstacle import Obstacle, Rectangle
import numpy as np
from typing import Dict, List, Tuple
from corridor_opt.math_utils import normalize_angle_0_2pi, normalize_angle_0_pi, rotation_matrix

def get_rectangle_and_bend_from_progression_and_width(edge: LineString, progression: float, width: float, edge_prev: LineString | None = None, length_margin: float = 0.1, n_radius_approx: int = 10, width_margin: float = 0.0) -> Tuple[Rectangle, Obstacle, Dict] | None:
    """
    Return rectangle alone as well as an obstacle combining rectangle with appropriate bend if geometry is valid.
    Returns None if geometry is invalid.
    """
    assert width > width_margin, f"width must be greater than width_margin. Got width={width:.1f} <= {width_margin:.1f}"
    joint_point = np.array(edge.interpolate(0).xy).squeeze()
    end_point = np.array(edge.interpolate(progression, normalized=True).xy).squeeze()
    dp = np.array(edge.interpolate(1e-3, normalized=True).xy).squeeze() - joint_point
    start_point = joint_point - dp
    # if edge_prev is None:
    #     dp = np.array(edge.interpolate(1e-3, normalized=True).xy).squeeze() - joint_point
    #     start_point = joint_point - dp
    # else:
    #     start_point = np.array(edge_prev.interpolate(0).xy).squeeze()
    
    # direction vectors
    direction_1 = joint_point - start_point
    direction_2 = end_point - joint_point

    # add margin to joint_point
    # joint_point = joint_point - margin * direction_1 / np.linalg.norm(direction_1)

    # angles
    theta1 = np.atan2(direction_1[0], direction_1[1]).squeeze()
    theta2 = np.atan2(direction_2[0], direction_2[1]).squeeze()

    # radius & width
    # radius = np.linalg.norm(end_point - joint_point) / 4 + width
    # radius = np.max([np.linalg.norm(end_point - joint_point) * (4 / 16), 2*width]) # + width # width # Originally: 4/16 = 1/4 # --> BEFORE IT WAS NP.MAX
    distance = LineString(edge.interpolate(np.linspace(0, progression, 30), normalized=True)).length
    radius = np.max([distance / 4, width])

    # Normalize angle difference and determine orientation
    delta_normalized = normalize_angle_0_2pi(theta2-theta1)
    orient = -1 if 0 <= delta_normalized <= np.pi else 1
    ortho = np.array([direction_1[1], -direction_1[0]])
    ortho_norm = np.linalg.norm(ortho)
    if ortho_norm > 0:
        orthonormal = orient * ortho / ortho_norm
    else:
        return None
    
    center = np.array(joint_point) - orthonormal * radius
    start_inner_radius = center + orthonormal * (radius - (width-width_margin) / 2)
    start_outer_radius = center + orthonormal * (radius + (width-width_margin) / 2)  

    d1 = np.linalg.norm(center-start_point)
    d2 = np.linalg.norm(center-end_point)
    if d1 > radius and d2 > radius:
        alpha1 = np.asin(radius/d1)
        alpha2 = np.asin(radius/d2)
    else:
        return None

    # vectors from vertices to center of rotation
    direction_1_center = center - start_point
    direction_2_center = end_point - center

    # angles
    gamma1 = np.atan2(direction_1_center[0], direction_1_center[1]).squeeze()
    gamma2 = np.atan2(direction_2_center[0], direction_2_center[1]).squeeze()

    # Compute alpha and check for validity
    alpha = alpha1 + alpha2 + orient * (gamma1 - gamma2)
    alpha_norm_0_pi = normalize_angle_0_pi(alpha)
    alpha_norm_0_2pi = normalize_angle_0_2pi(alpha)
    if alpha_norm_0_2pi >= np.pi:
        return None
    
    # compute inner / outer radius approximation
    stop_inner_radius = (start_inner_radius-center) @ rotation_matrix(orient * alpha_norm_0_pi).T + center
    stop_outer_radius = (start_outer_radius-center) @ rotation_matrix(orient * alpha_norm_0_pi).T + center

    alphas = np.linspace(0, alpha_norm_0_pi, n_radius_approx) 
    inner_radius = (start_inner_radius-center) @ rotation_matrix(orient * alphas).T + center  
    outer_radius = (start_outer_radius-center) @ rotation_matrix(orient * alphas).T + center  

    # Compute rectangle boundaries
    center_of_rotation = 0.5 * (stop_inner_radius + stop_outer_radius)
    rect_dir_vec = end_point - center_of_rotation
    rect_dir_norm = np.linalg.norm(rect_dir_vec)
    rect_angle = np.atan2(-rect_dir_vec[0], rect_dir_vec[1])
    rect_dir_with_margin = rect_dir_vec * ( 1 + 1*length_margin / rect_dir_norm)
    rect_height = np.linalg.norm(rect_dir_with_margin)

    top_right = stop_inner_radius + rect_dir_vec * ( 1 + length_margin / rect_dir_norm)
    top_left = stop_outer_radius + rect_dir_vec * ( 1 + length_margin / rect_dir_norm)
    
    corridor_coords = np.hstack([inner_radius.T, top_right[:, None], top_left[:, None], np.fliplr(outer_radius.T), inner_radius.T[:, 0, None]])
    corridor = Obstacle(corridor_coords.T.tolist(), geometry_type=Polygon)
    rect = Rectangle(*(center_of_rotation + np.array([0, rect_height/2])), rect_height, width-width_margin).rotate(rect_angle, center_of_rotation, use_radians=True)

    info_to_export = {
        'wp1': tuple(joint_point.tolist()),
        'wp2': tuple(end_point.tolist()),
        'dir1': tuple(direction_1.tolist()),
        'radius': radius,
        'angle': alpha_norm_0_pi,
        'width': width-width_margin,
        'length_margin': length_margin,
    }

    return rect, corridor, info_to_export

def get_rectangle_and_bend_from_wpts(wp1: Tuple[float, float], wp2: Tuple[float, float], dir1: Tuple[float, float], radius: float, angle: float, width: float, length_margin: float, n_radius_approx: int = 10) -> Tuple[Rectangle, Obstacle, LineString]:
    # direction vectors 
    direction_2 = np.array(wp2) - np.array(wp1)

    # add margin to joint_point
    # joint_point = joint_point - margin * direction_1 / np.linalg.norm(direction_1)

    # angles
    theta1 = np.atan2(dir1[0], dir1[1]).squeeze()
    theta2 = np.atan2(direction_2[0], direction_2[1]).squeeze()

    
    # Normalize angle difference and determine orientation
    delta_normalized = normalize_angle_0_2pi(theta2-theta1)
    orient = -1 if 0 <= delta_normalized <= np.pi else 1
    ortho = np.array([dir1[1], -dir1[0]])
    ortho_norm = np.linalg.norm(ortho)
    orthonormal = orient * ortho / ortho_norm

    # Center of rotation
    center = np.array(wp1) - orthonormal * radius

    # Compute backbone radius approximation
    start_backbone = center + orthonormal * radius

    # compute inner / outer radius approximation
    start_inner_radius = center + orthonormal * (radius - width / 2)
    start_outer_radius = center + orthonormal * (radius + width / 2)  
    
    
    stop_inner_radius = (start_inner_radius-center) @ rotation_matrix(orient * angle).T + center
    stop_outer_radius = (start_outer_radius-center) @ rotation_matrix(orient * angle).T + center

    alphas = np.linspace(0, angle, n_radius_approx) 
    inner_radius = (start_inner_radius-center) @ rotation_matrix(orient * alphas).T + center  
    outer_radius = (start_outer_radius-center) @ rotation_matrix(orient * alphas).T + center  
    backbone_radius = (start_backbone-center) @ rotation_matrix(orient * alphas).T + center  

    # Compute rectangle boundaries
    center_of_rotation = 0.5 * (stop_inner_radius + stop_outer_radius)
    rect_dir_vec = np.array(wp2) - center_of_rotation
    rect_dir_norm = np.linalg.norm(rect_dir_vec)
    rect_angle = np.atan2(-rect_dir_vec[0], rect_dir_vec[1])
    rect_dir_with_margin = rect_dir_vec * ( 1 + 1*length_margin / rect_dir_norm)
    rect_height = np.linalg.norm(rect_dir_with_margin)

    top_right = stop_inner_radius + rect_dir_vec * ( 1 + length_margin / rect_dir_norm)
    top_left = stop_outer_radius + rect_dir_vec * ( 1 + length_margin / rect_dir_norm)
    
    corridor_coords = np.hstack([inner_radius.T, top_right[:, None], top_left[:, None], np.fliplr(outer_radius.T), inner_radius.T[:, 0, None]])
    corridor = Obstacle(corridor_coords.T.tolist(), geometry_type=Polygon)
    rect = Rectangle(*(center_of_rotation + np.array([0, rect_height/2])), rect_height, width).rotate(rect_angle, center_of_rotation, use_radians=True)

    # Backbone
    backbone_coords = np.hstack([np.array(wp1).reshape(2, 1), backbone_radius.T, np.array(wp2).reshape(2, 1)])
    backbone = LineString(backbone_coords.T.tolist())

    return rect, corridor, backbone




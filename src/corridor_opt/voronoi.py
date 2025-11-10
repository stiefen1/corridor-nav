from typing import Tuple, List
from corridor_opt.obstacle import Obstacle
from scipy.spatial import Voronoi
from shapely import LineString
import networkx as nx, numpy as np
from tqdm import tqdm
from pathlib import Path
import pickle

def sample_points_from_obstacles(obstacles:list[Obstacle], meters_between_points: float = 50, verbose:bool=False):
    """
    Sample multiple points from each obstacle's boundary.
    """
    if verbose:
        print("Sampling generator points from obstacles...")

    all_points = []
    for obs in obstacles:
        boundary = obs.exterior
        n_points = int(boundary.length // meters_between_points)
        # Sample points along the boundary
        distances = np.linspace(0, boundary.length, n_points, endpoint=False)
        
        for distance in distances:
            point = boundary.interpolate(distance)
            all_points.append([point.x, point.y])
    
    return np.array(all_points)

def sample_points_from_lim(xlim, ylim, meters_between_points: float = 50, verbose:bool=False):
    """
    Generate boundary points around the perimeter.
    
    Args:
        xlim: (min_x, max_x) tuple
        ylim: (min_y, max_y) tuple  
        n_points_per_side: Number of points per boundary side
    
    Returns:
        List of [x, y] boundary points
    """
    if verbose:
        print("Sampling generator points from boundaries...")

    boundary_points = []
    n_points_x = int((xlim[1] - xlim[0]) // meters_between_points)
    n_points_y = int((ylim[1] - ylim[0]) // meters_between_points)
    # Bottom edge (left to right)
    x_vals = np.linspace(xlim[0], xlim[1], n_points_x)
    for x in x_vals:
        boundary_points.append([x, ylim[0]])
    
    # Right edge (bottom to top, excluding corners to avoid duplicates)
    y_vals = np.linspace(ylim[0], ylim[1], n_points_y)[1:]
    for y in y_vals:
        boundary_points.append([xlim[1], y])
    
    # Top edge (right to left, excluding corners)
    x_vals = np.linspace(xlim[1], xlim[0], n_points_x)[1:]
    for x in x_vals:
        boundary_points.append([x, ylim[1]])
    
    # Left edge (top to bottom, excluding corners)
    y_vals = np.linspace(ylim[1], ylim[0], n_points_y)[1:-1]
    for y in y_vals:
        boundary_points.append([xlim[0], y])
    
    return boundary_points

def convert_voronoi_to_visibility_graph(vor:Voronoi, obstacles:list[Obstacle], min_clearance=0.5, verbose:bool=False):
    """
    Create a navigation graph where nodes are safe Voronoi vertices
    and edges represent collision-free paths.
    """
    if verbose:
        print(f"Converting voronoi diagram (Nodes: {vor.vertices.shape}, edges: {len(vor.ridge_vertices)}) to visibility graph...")

    G = nx.Graph()    
    # Add only vertices that are far enough from obstacles
    safe_vertices = {}
    print("########### Nodes ###########")
    for i, vertex in enumerate(tqdm(vor.vertices)):
        x, y = vertex
            
        # Check clearance from all obstacles
        min_dist = min(obs.distance((x, y)) for obs in obstacles)
        
        if min_dist >= min_clearance:
            safe_vertices[i] = (x, y)
            G.add_node(i, pos=(x, y), clearance=min_dist)
    
    # Add edges between safe vertices along Voronoi ridges
    print("########### Edges ###########")
    for ridge in tqdm(vor.ridge_vertices):
        if len(ridge) == 2 and ridge[0] != -1 and ridge[1] != -1:
            valid_edge = True
            v1, v2 = ridge
            if v1 in safe_vertices and v2 in safe_vertices:
                for obs in obstacles:
                    if obs.intersects(LineString([safe_vertices[v1], safe_vertices[v2]])):
                        valid_edge = False
                        break
                if valid_edge:
                    pos1 = safe_vertices[v1]
                    pos2 = safe_vertices[v2]
                    weight = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    G.add_edge(v1, v2, weight=weight)
    
    return G

def remove_leaf_nodes(G: nx.Graph):
    """
    Remove all leaf nodes (nodes with degree 1) and their edges.
    Continues until no more leaf nodes exist.
    
    Args:
        G: NetworkX graph
    
    Returns:
        Modified graph with leaf nodes removed
    """
    G_copy = G.copy()
    
    while True:
        # Find all nodes with degree 1 (leaf nodes)
        leaf_nodes = [node for node, degree in G_copy.degree if degree == 1]
        
        if not leaf_nodes:
            # No more leaf nodes to remove
            break
        
        # Remove all leaf nodes
        G_copy.remove_nodes_from(leaf_nodes)
    
    return G_copy

def remove_isolated_nodes(G):
    """
    Remove all isolated nodes (nodes with degree 0).
    
    Args:
        G: NetworkX graph
    
    Returns:
        Modified graph with isolated nodes removed
    """
    G_copy = G.copy()
    isolated_nodes = list(nx.isolates(G_copy))
    G_copy.remove_nodes_from(isolated_nodes)
    return G_copy

def prune_graph(G, verbose:bool=False):
    """
    Remove both leaf nodes and isolated nodes from the graph.
    
    Args:
        G: NetworkX graph
    
    Returns:
        Pruned graph
    """
    if verbose:
        print("Pruning graph...")

    G_pruned = remove_leaf_nodes(G)
    G_pruned = remove_isolated_nodes(G_pruned)
    return G_pruned

def get_safe_voronoi_graph(
        obstacles:List[Obstacle],
        xlim:Tuple[float, float],
        ylim:Tuple[float, float],
        distance_between_generator_points:float,
        min_clearance:float,
        verbose:bool=False,
        sample_boundary:bool=False,
        distance_between_generator_points_boundary:float | None = None
    ) -> nx.Graph:

    """
    Build Voronoi diagram using sampled boundary points. For now we do not integrate start and target points
    """
    seed_points = sample_points_from_obstacles(obstacles, meters_between_points=distance_between_generator_points, verbose=verbose)
    
    if sample_boundary:
        # Add boundary points
        boundary_points = sample_points_from_lim(xlim, ylim, meters_between_points=distance_between_generator_points_boundary or distance_between_generator_points, verbose=verbose)
        all_points = np.vstack([seed_points, boundary_points])
    else:
        all_points = seed_points

    # build Voronoi diagram
    vor = Voronoi(all_points)

    # Convert into nx.Graph
    graph = convert_voronoi_to_visibility_graph(vor, obstacles, min_clearance=min_clearance, verbose=verbose)

    # Prune graph
    graph = prune_graph(graph, verbose=verbose)
    
    return graph

def save_graph(graph, filename):
    """Save NetworkX graph to file"""
    with open(filename, 'wb') as f:
        pickle.dump(graph, f)
    print(f"Graph saved to {filename}")

def load_graph(filename) -> nx.Graph | None:
    """Load NetworkX graph from file"""
    if not Path(filename).exists():
        return None
    
    with open(filename, 'rb') as f:
        graph = pickle.load(f)
    print(f"Graph loaded from {filename}")
    return graph
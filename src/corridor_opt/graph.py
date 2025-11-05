
from typing import List, Tuple, Any
from shapely import LineString
import networkx as nx
from corridor_opt.obstacle import Obstacle

def find_high_degree_nodes(G:nx.Graph, min_degree=3):
    """
    Find all nodes with degree >= min_degree.
    
    Args:
        G: NetworkX graph
        min_degree: Minimum degree threshold (default: 3)
    
    Returns:
        List of nodes with degree >= min_degree
    """
    high_degree_nodes = [node for node, degree in G.degree if degree >= min_degree]
    return high_degree_nodes

def next_node(graph:nx.Graph, node, main_node_start, main_nodes, prev_nodes=[], i:int=0) -> Tuple[List, Any, int]:
    for neighbor in list(graph.neighbors(node)):
        if (neighbor in prev_nodes) or ((neighbor == main_node_start) and i==0): # Skip nodes that were already visited + start node
            pass
        elif neighbor in main_nodes:
            return prev_nodes, neighbor, i
        else: # neighbor is not a main node and was not part of the already seen nodes
            return next_node(graph, neighbor, main_node_start, main_nodes, prev_nodes=prev_nodes+[neighbor], i=i+1)
    return [], None, 0

def get_edges_as_linestring(graph:nx.Graph, verbose=False) -> Tuple[ List[LineString], List[Tuple[int, int]]]:
    """
    Returns a list of resampled edges. Each resampled edge is a numpy array of evenly spaced points. 
    """
    if verbose:
        print("Resampling edges of main graph...")

    main_nodes = find_high_degree_nodes(graph)
    edges_as_linestring = []
    combinations = []
    for _, main_node in enumerate(main_nodes):
        for neighbor in graph.neighbors(main_node):
            
            # If direct neighbor happen to be main nodes, then we directly add this edge
            if neighbor in main_nodes:
                if not((neighbor, main_node) in combinations):
                    edges_as_linestring.append(LineString([graph.nodes[main_node]["pos"], graph.nodes[neighbor]["pos"]]))
                    combinations.append((neighbor, main_node))
                continue


            out = next_node(graph, neighbor, main_node, main_nodes, prev_nodes=[neighbor])
            prev_nodes, main_node_final, _ = out

            if main_node_final is None or (main_node_final == main_node) or (main_node, main_node_final) in combinations: # or (main_node_final, main_node) in combinations:
                continue

            if len(prev_nodes) > 1:
                list_of_intermediate_nodes = []
                for node in prev_nodes:
                    list_of_intermediate_nodes.append(graph.nodes[node]["pos"])
                list_of_intermediate_nodes.insert(0, graph.nodes[main_node]["pos"])
                list_of_intermediate_nodes.append(graph.nodes[main_node_final]["pos"])
                edges_as_linestring.append(LineString(list_of_intermediate_nodes))
                combinations.append((main_node_final, main_node))
    return edges_as_linestring, combinations

def get_min_distances_to_edges(edges_as_linestring:List[LineString], obstacles:List[Obstacle]) -> List[float]:
    distances = []
    for edge in edges_as_linestring:
        min_dist = float('inf')
        for obs in obstacles:
            dist = obs.distance(edge)
            if dist < min_dist:
                min_dist = dist
        distances.append(min_dist)
    return distances
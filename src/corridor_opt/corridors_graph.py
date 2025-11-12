import networkx as nx
from corridor_opt.corridor import Corridor
from typing import List

class GraphOfCorridors(nx.MultiDiGraph):
    def __init__(
            self,
            corridors: List[Corridor]
    ):
        self.corridors = corridors
        super().__init__()
        self.build_graph()
        
        
    def build_graph(self) -> None:
        # Populate graph with nodes and corridors

        edges = {} # id: {'corridors': List[Corridor], 'prev_main_node': int, 'next_main_node': int}
        vertices = []
        edges_ids = []

        # First build the edges, assign them to the graph later.
        for corridor in self.corridors:
            n_prev, n_next = corridor.prev_main_node, corridor.next_main_node
            if n_prev not in vertices:
                vertices.append(n_prev)
                self.add_node(n_prev)

            if n_next not in vertices:
                vertices.append(n_next)
                self.add_node(n_next)

            if corridor.edge_id not in edges_ids:
                edges.update({
                    corridor.edge_id: {
                        'corridors': [corridor],
                        'prev_main_node': n_prev,
                        'next_main_node': n_next
                        }
                    })
                edges_ids.append(corridor.edge_id)
            else:
                edges[corridor.edge_id]['corridors'].append(corridor)

        print(f"Added {len(vertices)} new vertices and {len(edges_ids)} new edges!")

        # Add edges grouped by edge_id (each edge contains list of corridors)
        for edge_id, edge_data in edges.items():
            prev_node = edge_data['prev_main_node']
            next_node = edge_data['next_main_node']
            corridor_list = edge_data['corridors']
            
            # Forward edge
            self.add_edge(
                prev_node,
                next_node,
                key=edge_id,
                corridors=corridor_list,
                edge_id=edge_id,
                direction='forward'
            )
            
            # Backward edge (flipped corridors)
            flipped_corridors = [corridor.get_flipped() for corridor in corridor_list]
            self.add_edge(
                next_node,
                prev_node,
                key=edge_id,
                corridors=flipped_corridors,
                edge_id=edge_id,
                direction='backward'
            )
        
        print(f"Total edges in graph: {len(self.edges())}")
        print(f"Unique node pairs: {len(set((min(u,v), max(u,v)) for u, v in self.edges()))}")
    
    def get_corridors_between_nodes(self, u: int, v: int):
        """Get all corridors between two nodes."""
        if not self.has_edge(u, v):
            return []
        
        edge_data = self.get_edge_data(u, v)
        return edge_data.get('corridors', [])
    
    def get_best_edge(self, u: int, v: int, weight_key: str = 'total'):
        """Get the edge with minimum weight between two nodes."""
        if not self.has_edge(u, v):
            return None
        
        best_key = None
        best_weight = float('inf')
        
        for key, edge_data in self[u][v].items():
            if weight_key in edge_data and edge_data[weight_key] < best_weight:
                best_weight = edge_data[weight_key]
                best_key = key
        return best_key
    
    def get_corridors_on_edge(self, u: int, v: int, key=None):
        """Get the list of corridors on a specific edge.
        
        Args:
            u: Source node
            v: Target node 
            key: Edge key (for MultiDiGraph). If None, returns corridors from first edge.
        
        Returns:
            List[Corridor]: List of corridors on this edge
        """
        if not self.has_edge(u, v):
            return []
            
        if key is not None:
            edge_data = self.get_edge_data(u, v, key)
        else:
            # Get data from first edge
            edge_data = self.get_edge_data(u, v)
            
        return edge_data.get('corridors', []) if edge_data else []
    
    def update_corridor_weights(self, corridor_metrics: dict, weight_key: str = 'total'):
        """
        Update edge weights based on corridor metrics.
        
        Args:
            corridor_metrics: Dictionary mapping corridors to their metric values
                             {corridor: value} for the specific metric
            weight_key: Name of the weight attribute to set on edges
                       ('total', 'risk', 'energy', 'combined', etc.)
        
        Example: 
            # Set cost weights
            corridor_costs = {c: energy_estimator.get_energy(c) for c in corridors}
            graph.update_corridor_weights(corridor_costs, 'total')
            
            # Set risk weights  
            corridor_risks = {c: risk_estimator.get_risk(c) for c in corridors}
            graph.update_corridor_weights(corridor_risks, 'risk')
            
            # Set energy weights
            corridor_energy = {c: energy_estimator.get_energy(c) for c in corridors}
            graph.update_corridor_weights(corridor_energy, 'energy')
        """
        for u, v, key, edge_data in self.edges(keys=True, data=True):
            corridors = edge_data.get('corridors', [])
            # Calculate total metric for all corridors on this edge
            total_metric = sum(corridor_metrics.get(corridor, 0) for corridor in corridors)
            self[u][v][key][weight_key] = total_metric
    
    def update_multiple_corridor_weights(self, corridor_metrics_dict: dict):
        """
        Update multiple weight attributes simultaneously.
        
        Args:
            corridor_metrics_dict: Dictionary of metric dictionaries
                                  {weight_key: {corridor: value}}
        
        Example:
            metrics = {
                'total': {c: cost_estimator.get_cost(c) for c in corridors},
                'risk': {c: risk_estimator.get_risk(c) for c in corridors}, 
                'energy': {c: energy_estimator.get_energy(c) for c in corridors},
                'combined': {c: 0.6*cost + 0.4*risk for c in corridors}
            }
            graph.update_multiple_corridor_weights(metrics)
        """
        for weight_key, corridor_metrics in corridor_metrics_dict.items():
            self.update_corridor_weights(corridor_metrics, weight_key)
    
    def print_edge_summary(self):
        """Print summary of all edges."""
        for u, v, key, data in self.edges(keys=True, data=True):
            corridors = data.get('corridors', [])
            direction = data.get('direction', 'unknown')
            corridor_ids = [c.idx for c in corridors]
            print(f"Edge ({u}->{v}, key={key}): {len(corridors)} corridors {corridor_ids} ({direction})")
    
    def find_shortest_path(self, source: int, target: int, weight: str = 'total'):
        """
        Find shortest path using Dijkstra's algorithm.
        Automatically selects best corridor when multiple exist between nodes.
        
        Args:
            source: Starting node
            target: Destination node  
            weight: Edge attribute to minimize ('total', 'risk', 'energy', etc.)
        
        Returns:
            tuple: (path_nodes, total_distance, corridors_used)
        """
        # try:
        # Get shortest path and distance
        path = nx.shortest_path(self, source=source, target=target, weight=weight)
        distance = nx.shortest_path_length(self, source=source, target=target, weight=weight)
        
        # Get the corridors used in this path
        corridors_used = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            # Find the edge with minimum weight (the one Dijkstra selected)
            best_key = self.get_best_edge(u, v, weight_key=weight)
            if best_key:
                edge_data = self.get_edge_data(u, v, best_key)
                corridors_on_edge = edge_data.get('corridors', [])
                corridors_used.extend(corridors_on_edge)
        
        return path, distance, corridors_used
    
    def find_k_shortest_paths(self, source: int, target: int, k: int = 3, weight: str = 'total'):
        """
        Find k shortest paths using different corridor combinations.
        
        Args:
            source: Starting node
            target: Destination node
            k: Number of paths to find
            weight: Edge attribute to minimize
        
        Returns:
            list: List of (path, distance, corridors) tuples, sorted by distance
        """
        try:
            # Convert MultiDiGraph to simple DiGraph for k-shortest paths algorithm
            simple_graph = nx.DiGraph()
            
            # Add all nodes
            simple_graph.add_nodes_from(self.nodes())
            
            # Add edges, keeping only the best (lowest weight) edge between each pair of nodes
            for u in self.nodes():
                for v in self.nodes():
                    if self.has_edge(u, v):
                        best_key = self.get_best_edge(u, v, weight_key=weight)
                        if best_key:
                            edge_data = self.get_edge_data(u, v, best_key)
                            simple_graph.add_edge(u, v, **edge_data)
            
            paths = []
            # Use k-shortest paths on the simplified graph
            for i, path in enumerate(nx.shortest_simple_paths(simple_graph, source, target, weight=weight)):
                if i >= k:
                    break
                
                # Calculate distance for this path
                distance = sum(simple_graph[u][v].get(weight, 1) for u, v in zip(path[:-1], path[1:]))
                
                # Get corridors used (from original MultiDiGraph)
                corridors_used = []
                for u, v in zip(path[:-1], path[1:]):
                    best_key = self.get_best_edge(u, v, weight_key=weight)
                    if best_key:
                        edge_data = self.get_edge_data(u, v, best_key)
                        corridors_on_edge = edge_data.get('corridors', [])
                        corridors_used.extend(corridors_on_edge)
                
                paths.append((path, distance, corridors_used))
            
            return paths
                
        except (nx.NetworkXNoPath, nx.NetworkXNotImplemented):
            return []
    
    def get_all_distances_from_node(self, source: int, weight: str = 'total'):
        """
        Get shortest distances from source to all reachable nodes using Dijkstra.
        
        Args:
            source: Starting node
            weight: Edge attribute to minimize
        
        Returns:
            dict: {node: distance} for all reachable nodes
        """
        return nx.single_source_dijkstra_path_length(self, source, weight=weight)
        
        



if __name__ == "__main__":
    import os
    path_to_corridors = os.path.join('Scripts', 'kristiansund', 'output', 'corridors_best')
    corridors = Corridor.load_all_corridors_in_folder(path_to_corridors)
    for corridor in corridors:
        assert corridor is not None, f""
    graph = GraphOfCorridors(corridors)
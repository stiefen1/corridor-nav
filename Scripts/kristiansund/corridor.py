"""
Script for extracting optimal corridors based on backbone graph.
"""
from corridor_opt.voronoi import load_graph
from corridor_opt.graph import get_edges_as_linestring, find_high_degree_nodes
from corridor_opt.core import build_corridors_graph
from corridor_opt.extract_shoreline import get_obstacles_in_window
from seacharts.enc import ENC
import sys, os, pathlib, networkx as nx, matplotlib.pyplot as plt

sys.setrecursionlimit(2000)

enc_config = os.path.join('config', 'kristiansund.yaml')
path_to_graph = os.path.join(pathlib.Path(__file__).parent, 'output', 'graph.pkl')
pso_params = {
    'n_particles': 40,
    'max_iter': 100,
    'inertia': 0.5,
    'c_cognitive': 0.2,
    'c_social': 0.5
    }
corridor_params = {
    'distance_margin': 100,
    'min_corridor_width': 100,
    'max_corridor_width': 2000
}


# Load graph from pickle file
graph = load_graph(path_to_graph)

# Load obstacles from enc
enc = ENC(enc_config)
xlim = enc.bbox[0], enc.bbox[2]
ylim = enc.bbox[1], enc.bbox[3]
obstacles = get_obstacles_in_window(enc, depth=5)

# Convert graph edges into linestring object
assert graph is not None, f"Failed to load graph using path {path_to_graph}."
edges_as_linestring = get_edges_as_linestring(graph)

# Build corridors
corridors = build_corridors_graph(edges_as_linestring, obstacles, **pso_params, **corridor_params)

# Retrieve main nodes to plot
main_nodes = find_high_degree_nodes(graph)

# Plot both approaches
_, (ax3) = plt.subplots(1, 1, figsize=(15, 7))
pos = nx.get_node_attributes(graph, 'pos')
nx.draw(graph, pos, ax=ax3, node_color="white", node_size=0, edge_color='grey', width=3, alpha=0.5)

for obs in obstacles:
    obs.fill(ax=ax3, c='green')

for corridor in corridors:
    corridor.plot(ax=ax3, c='orange')
    corridor.fill(ax=ax3, c='orange', alpha=0.5)
for node in main_nodes:
    ax3.scatter(*graph.nodes[node]["pos"], c='red')

ax3.set_xlim(xlim)
ax3.set_ylim(ylim)
ax3.set_aspect('equal')
plt.tight_layout()
plt.show()

"""
Script for extracting backbone graph.
"""

from corridor_opt.voronoi import get_safe_voronoi_graph, save_graph
from corridor_opt.extract_shoreline import get_obstacles_in_window
from seacharts.enc import ENC
import os, pathlib, matplotlib.pyplot as plt, networkx as nx

graph_filename = 'graph_with_boundary.pkl'
path_to_graph = os.path.join(pathlib.Path(__file__).parent, 'output', graph_filename)

config = os.path.join('config', 'kristiansund.yaml')
enc = ENC(config)
xlim = enc.bbox[0], enc.bbox[2]
ylim = enc.bbox[1], enc.bbox[3]

obstacles = get_obstacles_in_window(enc, depth=10)

graph = get_safe_voronoi_graph(obstacles, xlim, ylim, 
                                distance_between_generator_points=10, 
                                min_clearance=200, verbose=True, sample_boundary=True,
                                distance_between_generator_points_boundary=1000)

save_graph(graph, path_to_graph)

_, ax = plt.subplots()
for obs in obstacles:
    obs.plot(ax=ax)
    obs.fill(ax=ax, c='green')
pos = nx.get_node_attributes(graph, 'pos')
nx.draw(graph, pos, ax=ax, node_color="white", node_size=0, edge_color='grey', width=3, alpha=0.5)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal')
plt.show()
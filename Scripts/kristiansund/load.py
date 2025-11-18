from corridor_opt.corridor import Corridor
import os, pathlib, matplotlib.pyplot as plt, random, networkx as nx
from corridor_opt.voronoi import load_graph 
from corridor_opt.extract_shoreline import get_obstacles_in_window
from seacharts import ENC

def random_color():
    """Generate a random color as (r, g, b) tuple with values 0-1"""
    return (random.random(), random.random(), random.random())

if __name__ == "__main__":
    start_pos = (1.3317e5, 7.02255e6)
    target_pos = (1.5192e5, 6.99523e6)

    path_to_corridors = os.path.join(pathlib.Path(__file__).parent, 'output', 'corridors_best')
    path_to_graph = os.path.join(pathlib.Path(__file__).parent, 'output', 'graph.pkl')
    config = os.path.join('config', 'kristiansund.yaml')
    enc = ENC(config)
    xlim = enc.bbox[0], enc.bbox[2]
    ylim = enc.bbox[1], enc.bbox[3]

    # Get seabed at 10 meters
    obstacles = get_obstacles_in_window(enc, depth=10)

    # Load corridor from txt files
    corridors = Corridor.load_all_corridors_in_folder(path_to_corridors)

    # Load graph from pickle file
    graph = load_graph(path_to_graph)

    fig, ax = plt.subplots(figsize=(7, 7))
    pos = nx.get_node_attributes(graph, 'pos')
    nx.draw(graph, pos, ax=ax, node_color="black", node_size=0, edge_color='dimgray', width=3, alpha=1, label='backbone')
    ax.set_axis_on()
    # ax.set_frame_on(True)
    # ax.patch.set_visible(True)
    ax.xaxis.set_visible(True); ax.yaxis.set_visible(True)
    ax.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)
    
    ax.set_facecolor('lightsteelblue')
    for i, obs in enumerate(obstacles):
        obs.fill(ax=ax, c='forestgreen', label='Seabed at 10m' if i==0 else None)

    # for j, corridor in enumerate(corridors):
    #     corridor.fill(ax=ax, c='peru', alpha=0.7, label='corridors' if j==0 else None)

    # ax.scatter(*start_pos, c='blue', label='start')
    # ax.scatter(*target_pos, c='red', label='goal')

    ax.set_title(f"Area of Kristiansund (NO) - Graph of Corridors (UTM)")
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    
    handles, labels = ax.get_legend_handles_labels()
    idx_to_remove = labels.index('backbone')
    filtered_handles = [h for i, h in enumerate(handles) if i != idx_to_remove]
    filtered_labels = [l for i, l in enumerate(labels) if i != idx_to_remove]
    ax.legend(filtered_handles, filtered_labels, loc='right', facecolor='white', framealpha=1.0)
    ax.set_aspect('equal')
    plt.show()
    plt.close()

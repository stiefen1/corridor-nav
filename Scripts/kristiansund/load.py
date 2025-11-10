from corridor_opt.corridor import Corridor
import os, pathlib, matplotlib.pyplot as plt, random, networkx as nx
from corridor_opt.voronoi import load_graph 

def random_color():
    """Generate a random color as (r, g, b) tuple with values 0-1"""
    return (random.random(), random.random(), random.random())

if __name__ == "__main__":
    path_to_corridors = os.path.join(pathlib.Path(__file__).parent, 'output', 'corridors_best')
    path_to_graph = os.path.join(pathlib.Path(__file__).parent, 'output', 'graph.pkl')
    
    # Load corridor from txt files
    corridors = Corridor.load_all_corridors_in_folder(path_to_corridors)

    # Load graph from pickle file
    graph = load_graph(path_to_graph)

    _, ax = plt.subplots()
    pos = nx.get_node_attributes(graph, 'pos')
    nx.draw(graph, pos, ax=ax, node_color="white", node_size=0, edge_color='grey', width=3, alpha=0.5)

    dict_of_colors = {}
    for corridor in corridors:
        pair = (corridor.prev_main_node, corridor.next_main_node)
        if pair in dict_of_colors.keys():
            ax = corridor.fill(ax=ax, c=dict_of_colors[pair], alpha=0.5)
            ax.scatter(*corridor.centroid, color=dict_of_colors[pair])
        else:
            dict_of_colors.update({pair: random_color()})
            ax = corridor.fill(ax=ax, c=dict_of_colors[pair], alpha=0.5)
            ax.scatter(*corridor.centroid, color=dict_of_colors[pair])
        

    ax.set_aspect('equal')
    plt.show()

import random

def random_color():
    """Generate a random color as (r, g, b) tuple with values 0-1"""
    return (random.random(), random.random(), random.random())

if __name__ == "__main__":
    from corridor_opt.corridor import Corridor
    import os, matplotlib.pyplot as plt, numpy as np

    path_to_corridors = os.path.join('Scripts', 'kristiansund', 'output', 'corridors_best')
    
    # Load corridor from txt files
    corridors = Corridor.load_all_corridors_in_folder(path_to_corridors)

    _, ax = plt.subplots()

    colors = {}
    for corridor in corridors:
        pair = (corridor.prev_main_node, corridor.next_main_node)
        if not pair in colors.keys():
            colors.update({pair: random_color()})
            
        # Fill corridor
        ax = corridor.fill(ax=ax, c=colors[pair], alpha=0.5)

        # Show centroid (not necessarily within the corridor)
        ax.scatter(*corridor.centroid, color=colors[pair])

        # Show middle of the backbone, i.e. when progression = 0.5
        ax.scatter(*corridor.backbone.interpolate(0.5, normalized=True).xy, color=colors[pair], marker='x')

        # Show backbone
        backbone = np.array([corridor.backbone.interpolate(prog, normalized=True).xy for prog in np.linspace(0, 1, 30).tolist()])
        ax.plot(backbone[:, 0], backbone[:, 1], '--', c=colors[pair])
        

    ax.set_aspect('equal')
    plt.show()
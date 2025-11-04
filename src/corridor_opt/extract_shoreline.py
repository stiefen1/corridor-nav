
if __name__ == "__main__":
    import os
    from seacharts.enc import ENC
    from .obstacle import Rectangle
    import matplotlib.pyplot as plt

    r = Rectangle(10, 0, 10, 3).rotate(30)
    ax = r.plot()
    ax.set_aspect('equal')
    plt.show()
    plt.close()

    config_path = os.path.join('config', 'trondelag_corridors.yaml')
    enc = ENC(config_path)
    enc.display.start()
    enc.display.show()
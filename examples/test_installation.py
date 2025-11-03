import os, numpy as np
from seacharts.enc import ENC
from pso import PSO, CostBase
from corridor_opt import Rectangle
from decision_making import *
import matplotlib.pyplot as plt

class QuadOpt(CostBase):
    def eval(self, x:float, y:float, *args, **kwargs) -> float:
        return np.min([(x+3)**2 + (y+3)**2, (x-3)**2 + (y-3)**2])

# Create PSO optimizer
optimizer = PSO(
    cost=QuadOpt(),
    lbx=(-10, -10),
    ubx=(10, 10), # Search space limits
    n_particles=50,
    max_iter=300,
    inertia=0.6,
    c_cognitive=0.1,
    c_social=0.3,
    stop_at_variance=1e-6
)

# Run optimization
optimizer.optimize()

# Get PSO results
best_position = optimizer.get_optimal_position()
best_cost = optimizer.get_optimal_cost()
print(f"Optimal particle: {best_position.tolist()}, Optimal cost: {best_cost:.2f}")

# SeaCharts ENC
config_path = os.path.join('src', 'corridor_opt', 'config', 'trondelag_1.yaml')
enc = ENC(config_path)
enc.display.start()
enc.display.show()
enc.display.close()

r = Rectangle(10, 5, 20, 5).rotate(30)
ax = r.plot()
ax.set_aspect('equal')
plt.show()
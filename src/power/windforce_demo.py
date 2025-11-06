"""Minimal interactive demo for WindForceEstimator

Run from repository root with your corridor-nav conda env activated:

python examples/windforce_demo.py

This creates a Matplotlib window with sliders for:
- u: surge speed (m/s)
- psi: heading (deg)
- wind_speed: wind speed (m/s)
- wind_dir: wind direction (deg, from north, clockwise)
- v: sway speed (m/s)

The figure shows a simple ship rectangle, the wind arrow and the resulting force vector (scaled for display).
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Make sure src/ is importable when running from repo root
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from power import WindForceEstimator, DefaultWindParameters
from dataclasses import asdict

# Create estimator with default ship geometry
EST = WindForceEstimator(
    loa=80.0,
    beam=16.0,
    **asdict(DefaultWindParameters())
)

# plotting helpers
def rotation_matrix_2d(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])

# initial parameters
init_u = 5.0          # surge speed (m/s)
init_psi = 0.0        # heading (deg)
init_wind_speed = 10.0
init_wind_dir = 270.0 # wind coming from west
init_v = 0.0          # sway speed

# create figure
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(left=0.1, bottom=0.25)
ax.set_aspect('equal', 'box')
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_title('Wind force demo')
ax.grid(True, alpha=0.4)

# ship geometry (centered at origin, pointing to +y (north) when psi=0)
loa = EST.loa
beam = EST.beam
# rectangle corners (clockwise) relative to center, with bow pointing north (+y)
ship_corners = np.array([
    [beam/2,  loa/2],   # starboard bow
    [beam/2, -loa/2],   # starboard stern  
    [-beam/2, -loa/2],  # port stern
    [-beam/2,  loa/2],  # port bow
    [beam/2,  loa/2],   # close the polygon
])
ship_patch, = ax.plot([], [], '-k', lw=2)

# initialize global quiver objects
wind_q = None
force_quiver = None

# wind arrow text
wind_text = ax.text(-80, 88, '', color='tab:blue')

# force text
force_text = ax.text(5, 5, '', color='tab:red')

# numeric annotation
info_text = ax.text(-95, -95, '', fontsize=9)

# slider axes
axcolor = 'lightgoldenrodyellow'
ax_u = plt.axes((0.15, 0.15, 0.65, 0.03), facecolor=axcolor)
ax_psi = plt.axes((0.15, 0.10, 0.65, 0.03), facecolor=axcolor)
ax_ws = plt.axes((0.15, 0.05, 0.30, 0.03), facecolor=axcolor)
ax_wd = plt.axes((0.50, 0.05, 0.30, 0.03), facecolor=axcolor)
ax_v = plt.axes((0.15, 0.02, 0.65, 0.03), facecolor=axcolor)

s_u = Slider(ax_u, 'u (m/s)', -5.0, 20.0, valinit=init_u)
s_psi = Slider(ax_psi, 'psi (deg)', 0.0, 360.0, valinit=init_psi)
s_ws = Slider(ax_ws, 'wind (m/s)', 0.0, 40.0, valinit=init_wind_speed)
s_wd = Slider(ax_wd, 'wind dir (deg)', 0.0, 360.0, valinit=init_wind_dir)
s_v = Slider(ax_v, 'v (m/s)', -5.0, 5.0, valinit=init_v)

# update function
def update(val=None):
    u = s_u.val
    psi = s_psi.val
    wind_speed = s_ws.val
    wind_dir = s_wd.val
    v = s_v.val

    # compute force from estimator (use degrees=True to pass human-friendly angles)
    tau = EST.get(u=u, psi=psi, wind_speed=wind_speed, wind_dir=wind_dir, v=v, degrees=True)
    Fx_ship = tau[0]
    Fy_ship = tau[1]
    Mz = tau[2]

    # rotate force and ship from ship frame to North-East frame 
    # psi is heading from north, clockwise positive (marine convention)
    psi_rad = np.deg2rad(psi)
    R_ship_to_world = rotation_matrix_2d(psi_rad)  # clockwise rotation
    F_NE = R_ship_to_world @ np.array([Fx_ship, Fy_ship])  # [North, East]
    
    # convert from North-East to plot (x,y) coordinates: North->y, East->x
    F_world = np.array([F_NE[1], F_NE[0]])  # [East, North] = [x, y]

    # update ship polygon: rotate from initial north-pointing orientation by psi clockwise
    ship_world = (rotation_matrix_2d(-psi_rad) @ ship_corners.T).T
    ship_patch.set_data(ship_world[:,0], ship_world[:,1])

    # update wind arrow: wind_dir given as direction wind comes from, w.r.t north clockwise.
    wd_rad = np.deg2rad(wind_dir)
    # wind vector in world coords: points toward where wind goes (so opposite of wind coming from)
    wind_vec_world = np.array([
        np.sin(wd_rad), # x component (east)
        np.cos(wd_rad)  # y component (north)
    ]) * wind_speed * 2.0  # scale for visibility

    # place arrow anchor
    anchor = np.array([-80, 80])
    # draw wind arrow manually using quiver for easier updates
    global wind_q
    if wind_q is not None:
        wind_q.remove()
    wind_q = ax.quiver(anchor[0], anchor[1], wind_vec_world[0], wind_vec_world[1], angles='xy', scale_units='xy', scale=1, color='tab:blue', width=0.01)
    wind_text.set_text(f'Wind: {wind_speed:.1f} m/s \nDir: {wind_dir:.0f}°')

    # update force quiver - show force in WORLD frame for intuitive visualization
    global force_quiver
    if force_quiver is not None:
        force_quiver.remove()
    # scale force for visibility
    scale = 1000.0
    force_quiver = ax.quiver(0, 0, F_world[0]/scale, F_world[1]/scale, angles='xy', scale_units='xy', scale=1, color='tab:red', width=0.02)
    
    # show coordinate systems clearly in text 
    force_text.set_text(f'Ship: [{Fx_ship:.1f}, {Fy_ship:.1f}] N (surge, sway)\nN-E: [{F_NE[0]:.1f}, {F_NE[1]:.1f}] N (north, east)\nPlot: [{F_world[0]:.1f}, {F_world[1]:.1f}] N (x, y)\nMz: {Mz:.1f} Nm')

    info_text.set_text(f'u={u:.2f} m/s, v={v:.2f} m/s, psi={psi:.0f}°')

    fig.canvas.draw_idle()

# init drawing
update()

# connect sliders
s_u.on_changed(update)
s_psi.on_changed(update)
s_ws.on_changed(update)
s_wd.on_changed(update)
s_v.on_changed(update)

# reset button
resetax = plt.axes((0.8, 0.9, 0.1, 0.04))
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    s_u.reset()
    s_psi.reset()
    s_ws.reset()
    s_wd.reset()
    s_v.reset()

button.on_clicked(reset)

plt.show()

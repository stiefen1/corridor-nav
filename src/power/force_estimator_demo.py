"""Minimal interactive demo for ForceEstimator (total environmental forces)

Run from repository root with your corridor-nav conda env activated:

python examples/force_estimator_demo.py

This creates a Matplotlib window with sliders for:
- u: surge speed (m/s)
- psi: heading (deg)
- wind_speed: wind speed (m/s) 
- wind_dir: wind direction (deg, from north, clockwise)
- current_speed: current speed (m/s)
- current_dir: current direction (deg, from north, clockwise)
- significant_wave_height: H_s (m)
- wave_dir: wave direction (deg, from north, clockwise)

The figure shows the ship, environmental loads (wind, current, waves) and the resulting actuator force.
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

from power.total_force_estimator import ForceEstimator

# Create estimator with default parameters
EST = ForceEstimator()

# plotting helpers
def rotation_matrix_2d(angle_rad: float) -> np.ndarray:
    """from global to body"""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])

# initial parameters
init_u = 5.0               # surge speed (m/s)
init_psi = 0.0             # heading (deg)
init_wind_speed = 10.0     # wind speed (m/s)
init_wind_dir = 270.0      # wind coming from west
init_current_speed = 2.0   # current speed (m/s)
init_current_dir = 90.0    # current from east
init_hs = 2.0              # significant wave height (m)
init_wave_dir = 180.0      # waves from south

# create figure
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.35)
ax.set_aspect('equal', 'box')
ax.set_xlim(-150, 150)
ax.set_ylim(-150, 150)
ax.set_title('Total Force Estimator Demo')
ax.grid(True, alpha=0.4)

# ship geometry (centered at origin, pointing north when psi=0)
loa = EST.ship_params.loa
beam = EST.ship_params.beam
ship_corners = np.array([
    [beam/2,  loa/2],   # starboard bow
    [beam/2, -loa/2],   # starboard stern  
    [-beam/2, -loa/2],  # port stern
    [-beam/2,  loa/2],  # port bow
    [beam/2,  loa/2],   # close the polygon
])
ship_patch, = ax.plot([], [], '-k', lw=3, label='Ship')

# initialize global objects
wind_q = None
current_q = None
wave_q = None
force_q = None

# text displays
wind_text = ax.text(-140, 120, '', color='tab:blue', fontsize=9)
current_text = ax.text(-140, 100, '', color='tab:green', fontsize=9)
wave_text = ax.text(-140, 80, '', color='tab:purple', fontsize=9)
force_text = ax.text(-140, 40, '', color='tab:red', fontsize=9)
info_text = ax.text(-140, -140, '', fontsize=9)

# slider setup
axcolor = 'lightgoldenrodyellow'
slider_height = 0.02
slider_spacing = 0.025

# position sliders
ax_u = plt.axes((0.15, 0.28, 0.3, slider_height), facecolor=axcolor)
ax_psi = plt.axes((0.55, 0.28, 0.3, slider_height), facecolor=axcolor)
ax_ws = plt.axes((0.15, 0.25, 0.3, slider_height), facecolor=axcolor)
ax_wd = plt.axes((0.55, 0.25, 0.3, slider_height), facecolor=axcolor)
ax_cs = plt.axes((0.15, 0.22, 0.3, slider_height), facecolor=axcolor)
ax_cd = plt.axes((0.55, 0.22, 0.3, slider_height), facecolor=axcolor)
ax_hs = plt.axes((0.15, 0.19, 0.3, slider_height), facecolor=axcolor)
ax_wadir = plt.axes((0.55, 0.19, 0.3, slider_height), facecolor=axcolor)

# create sliders
s_u = Slider(ax_u, 'Surge (m/s)', 0.0, 15.0, valinit=init_u)
s_psi = Slider(ax_psi, 'Heading (deg)', 0.0, 360.0, valinit=init_psi)
s_ws = Slider(ax_ws, 'Wind speed (m/s)', 0.0, 30.0, valinit=init_wind_speed)
s_wd = Slider(ax_wd, 'Wind dir (deg)', 0.0, 360.0, valinit=init_wind_dir)
s_cs = Slider(ax_cs, 'Current (m/s)', 0.0, 5.0, valinit=init_current_speed)
s_cd = Slider(ax_cd, 'Current dir (deg)', 0.0, 360.0, valinit=init_current_dir)
s_hs = Slider(ax_hs, 'Wave H_s (m)', 0.0, 8.0, valinit=init_hs)
s_wadir = Slider(ax_wadir, 'Wave dir (deg)', 0.0, 360.0, valinit=init_wave_dir)

def update(val=None):
    u = s_u.val
    psi = s_psi.val
    wind_speed = s_ws.val
    wind_dir = s_wd.val
    current_speed = s_cs.val
    current_dir = s_cd.val
    hs = s_hs.val
    wave_dir = s_wadir.val

    # compute total actuator forces needed
    tau = EST.get(
        u=u, psi=psi, 
        wind_speed=wind_speed, wind_dir=wind_dir,
        current_speed=current_speed, current_dir=current_dir,
        significant_wave_height=hs, wave_dir=wave_dir,
        degrees=True
    )
    Fx_act = tau[0]  # surge force from actuators
    Fy_act = tau[1]  # sway force from actuators
    Mz_act = tau[2]  # yaw moment from actuators

    # coordinate transformations (ship frame -> North-East -> plot x,y)
    psi_rad = np.deg2rad(psi)
    R_ship_to_world = rotation_matrix_2d(psi_rad)  # clockwise rotation (marine convention)
    
    # rotate ship geometry
    ship_world = (R_ship_to_world.T @ ship_corners.T).T
    ship_patch.set_data(ship_world[:,0], ship_world[:,1])

    # environmental vectors (all in North-East, then convert to plot coordinates)
    def env_vector_to_plot(speed, direction_deg, scale=3.0):
        """Convert environmental vector from speed/direction to plot coordinates
        
        Args:
            speed: magnitude of the vector
            direction_deg: direction in degrees (0° = north, clockwise positive)
            scale: scaling factor for visualization
            
        Returns:
            (x_component, y_component) for matplotlib plotting
        """
        dir_rad = np.deg2rad(direction_deg)
        # Direction 0° = north, clockwise positive
        north_component = speed * np.cos(dir_rad) * scale
        east_component = speed * np.sin(dir_rad) * scale
        # Convert North-East to plot x,y: East->x, North->y
        return east_component, north_component

    # actuator force (ship frame -> world frame -> plot)
    F_NE = R_ship_to_world @ np.array([Fx_act, Fy_act])
    F_plot = np.array([F_NE[1], F_NE[0]])  # [East, North] -> [x, y]

    # update environmental arrows
    global wind_q, current_q, wave_q, force_q
    
    # remove old arrows
    for q in [wind_q, current_q, wave_q, force_q]:
        if q is not None:
            q.remove()

    # environmental vector positions (around ship)
    positions = [
        (-100, 100),   # wind
        (-100, 60),    # current  
        (-100, 20),    # waves
    ]

    # wind arrow
    wind_vec = env_vector_to_plot(wind_speed, wind_dir)
    wind_q = ax.quiver(positions[0][0], positions[0][1], wind_vec[0], wind_vec[1], 
                       angles='xy', scale_units='xy', scale=1, color='tab:blue', width=0.008)
    wind_text.set_text(f'Wind: {wind_speed:.1f} m/s from {wind_dir:.0f}°')

    # current arrow  
    current_vec = env_vector_to_plot(current_speed, current_dir)
    current_q = ax.quiver(positions[1][0], positions[1][1], current_vec[0], current_vec[1],
                          angles='xy', scale_units='xy', scale=1, color='tab:green', width=0.008)
    current_text.set_text(f'Current: {current_speed:.1f} m/s from {current_dir:.0f}°')

    # wave arrow (represent wave energy/force direction)
    wave_vec = env_vector_to_plot(hs, wave_dir, scale=8.0)  # scale by wave height
    wave_q = ax.quiver(positions[2][0], positions[2][1], wave_vec[0], wave_vec[1],
                       angles='xy', scale_units='xy', scale=1, color='tab:purple', width=0.008)
    wave_text.set_text(f'Waves: H_s={hs:.1f} m from {wave_dir:.0f}°')

    # actuator force arrow (at ship center)
    force_scale = 5000.0  # scale for visibility
    force_q = ax.quiver(0, 0, F_plot[0]/force_scale, F_plot[1]/force_scale,
                        angles='xy', scale_units='xy', scale=1, color='tab:red', width=0.015)
    force_text.set_text(f'Actuator Force:\nSurge: {Fx_act:.0f} N\nSway: {Fy_act:.0f} N\nYaw: {Mz_act:.0f} Nm')

    info_text.set_text(f'Ship: u={u:.1f} m/s, ψ={psi:.0f}°')

    fig.canvas.draw_idle()

# initialize
update()

# connect sliders
for slider in [s_u, s_psi, s_ws, s_wd, s_cs, s_cd, s_hs, s_wadir]:
    slider.on_changed(update)

# reset button
resetax = plt.axes((0.8, 0.9, 0.1, 0.04))
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    for slider in [s_u, s_psi, s_ws, s_wd, s_cs, s_cd, s_hs, s_wadir]:
        slider.reset()

button.on_clicked(reset)

# add legend
ax.legend(loc='upper right')

plt.show()
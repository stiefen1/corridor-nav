"""
Goal: compute risk associated to a corridor given wave, current, wind, traffic density and own ship's parameters.

Traffic density, maneuverability and lateral (sway) force are classified as either LOW, MEDIUM, HIGH using two thresholds each.


"""

TRAFFIC_THRESH = (1e-4, 5e-4)
TORQUE_THRESH = (20e3, 50e3)    # Mean=28000, std=10800 in our data
SWAY_THRESH = (15e3, 25e3)        # Mean=
WIDTH_THRESH = (300, 600)
STATES = ('L', 'M', 'H')


import numpy as np, pandas as pd, os, pathlib
from typing import Literal, Tuple


def get_state(val: float, thresholds: Tuple[float, float]) -> Literal['L', 'M', 'H']:
    if val <= thresholds[0]:
        return 'L'
    elif val <= thresholds[1]:
        return 'M'
    else:
        return 'H'

class RiskModel:
    def __init__(
            self,
            t_grounding: float = 1e6,
            t_collision: float = 1e6,
            prob_grounding_given_exit: float = 0.5
        ):
        assert prob_grounding_given_exit < 1.0, f"prob_grounding_given_exit is probability and must hence be < 0. Got {prob_grounding_given_exit:.3f}."

        self.t_grounding = t_grounding # travel time after grounding
        self.t_collision = t_collision # travel time after collision
        self.prob_grounding_given_exit = prob_grounding_given_exit # Probability of grounding given the ship is outside a corridor
        self.bbn = BBN()

    def get(
            self,
            travel_time: float,
            traffic_density: float,
            forces: np.ndarray,
            width: float
        ) -> float:

        # Probability of collision (power + drifting)
        prob_powered_collision = self.bbn.prob_powered_collision(
            get_state(traffic_density, TRAFFIC_THRESH),
            get_state(forces[2], TORQUE_THRESH)
        )
        prob_drifting_collision = 0
        prob_collision = prob_powered_collision + prob_drifting_collision

        # Probability of powered exit
        ## Because of bad tracking. travel_time -> infty <-> prob -> 1. actual_width -> infty <-> prob -> 0
        prob_powered_exit_bad_tracking_per_second = self.bbn.prob_powered_exit_tracking_per_sec(
            get_state(width, WIDTH_THRESH),
            get_state(forces[1], SWAY_THRESH)
        ) # f(actual_width, travel_time) -> if tracking accuracy is +-2*sigma it means 95.5% error is <= tracking accuracy. So 4.5% of the time, we get outside of it -> maybe there is something to do here

        prob_powered_exit_bad_tracking = prob_powered_exit_bad_tracking_per_second * travel_time

        ## Because of COLAV
        prob_powered_exit_colav = self.bbn.prob_powered_exit_colav(
            get_state(width, WIDTH_THRESH),
            get_state(traffic_density, TRAFFIC_THRESH),
            get_state(forces[2], TORQUE_THRESH)
        )

        ## Total 
        prob_powered_exit = prob_powered_exit_bad_tracking + prob_powered_exit_colav
        prob_drifting_exit = 0

        # Probability of grounding
        prob_grounding = self.prob_grounding_given_exit * (prob_powered_exit + prob_drifting_exit)

        assert prob_grounding + prob_collision <= 1.0, f"Sum of probabilities must be <= 1.0. Got prob(grounding)={prob_grounding}, prob(collision)={prob_collision}"

        

        # Expected travel time
        expected_travel_time = prob_collision * self.t_collision + prob_grounding * self.t_grounding + (1-prob_collision-prob_grounding) * travel_time

        return expected_travel_time
    
class BBN:
    powered_exit_tracking_table: pd.DataFrame
    powered_exit_colav_table: pd.DataFrame
    powered_collision_table: pd.DataFrame
    
    def __init__(
        self,
        powered_exit_tracking_src: str = os.path.join(pathlib.Path(__file__).parent, 'prob_power_exit_tracking.csv'),
        powered_exit_colav_src: str = os.path.join(pathlib.Path(__file__).parent, 'prob_power_exit_colav.csv'),
        powered_collision_src: str = os.path.join(pathlib.Path(__file__).parent, 'prob_power_collision.csv'),
    ):
        (self.powered_exit_tracking_table, self.powered_exit_colav_table, self.powered_collision_table) = [pd.read_csv(p, delimiter=';') for p in (powered_exit_tracking_src, powered_exit_colav_src, powered_collision_src)]

    def prob_powered_exit_tracking_per_sec(self, width: Literal['L', 'M', 'H'], sway_force: Literal['L', 'M', 'H']) -> float:
        """
        Convert width and sway force status into a probability of powered exit tracking per second based on tables.

        'L' = 'Low'
        'M' = 'Medium'
        'H' = 'High'
        """
        assert width in STATES, f"width must be one of the following str: {STATES}. Got width={width}."
        assert sway_force in STATES, f"sway_force must be one of the following str: {STATES}. Got sway_force={sway_force}."
        return self.powered_exit_tracking_table.loc[
            (self.powered_exit_tracking_table['width'] == width) &
            (self.powered_exit_tracking_table['sway-force'] == sway_force),
            'p(exit)'
        ].iloc[0]
    
    def prob_powered_exit_colav(self, width: Literal['L', 'M', 'H'], traffic: Literal['L', 'M', 'H'], torque: Literal['L', 'M', 'H']) -> float:
        assert traffic in STATES, f"traffic must be one of the following str: {STATES}. Got traffic={traffic}."
        assert torque in STATES, f"torque must be one of the following str: {STATES}. Got torque={torque}."
        assert width in STATES, f"width must be one of the following str: {STATES}. Got width={width}."
        return self.powered_exit_colav_table.loc[
            (self.powered_exit_colav_table['width'] == width) &
            (self.powered_exit_colav_table['traffic'] == traffic) &
            (self.powered_exit_colav_table['torque'] == torque),
            'p(exit)'
        ].iloc[0]
    
    def prob_powered_collision(self, traffic: Literal['L', 'M', 'H'], torque: Literal['L', 'M', 'H']) -> float:
        assert traffic in STATES, f"traffic must be one of the following str: {STATES}. Got traffic={traffic}."
        assert torque in STATES, f"torque must be one of the following str: {STATES}. Got torque={torque}."
        return self.powered_collision_table.loc[
            (self.powered_collision_table['traffic'] == traffic) &
            (self.powered_collision_table['torque'] == torque),
            'p(collision)'
        ].iloc[0]

if __name__ == "__main__":
    # Minimal visual demo of how width, traffic and maneuverability levels impact probabilities
    import matplotlib.pyplot as plt

    bbn = BBN()

    # Helper to annotate heatmap cells
    def annotate(ax, data, vmin=None, vmax=None):
        # Choose text color based on midpoint of the data range for readability
        if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            mid = 0.5
        else:
            mid = 0.5 * (vmin + vmax)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = float(data[i, j])
                txt = f"{val:.3g}"  # compact formatting
                ax.text(j, i, txt, ha='center', va='center', color='white' if val > mid else 'black', fontsize=9)

    # Build 3x3 heatmap for powered collision: rows=traffic, cols=maneuverability
    col_collision = np.zeros((3, 3), dtype=float)
    for r, traffic in enumerate(STATES):
        for c, man in enumerate(STATES):
            col_collision[r, c] = bbn.prob_powered_collision(traffic, man)

    # Build 3 heatmaps for powered exit due to COLAV, one per width level
    col_colav = []
    for width in STATES:
        arr = np.zeros((3, 3), dtype=float)
        for r, traffic in enumerate(STATES):
            for c, man in enumerate(STATES):
                arr[r, c] = bbn.prob_powered_exit_colav(width, traffic, man)
        col_colav.append(arr)

    # Figure 1: P(exit|COLAV) per width (L, M, H), each panel with its own color scale
    fig_colav, axes_colav = plt.subplots(1, 3, figsize=(11, 3), constrained_layout=True)
    last_im = None
    for i, width in enumerate(STATES):
        arr = col_colav[i]
        vmin_i = float(np.nanmin(arr))
        vmax_i = float(np.nanmax(arr))
        if vmin_i == vmax_i:
            vmin_i, vmax_i = vmin_i - 1e-12, vmax_i + 1e-12
        im = axes_colav[i].imshow(arr, vmin=vmin_i, vmax=vmax_i, cmap='viridis', origin='upper')
        last_im = im
        axes_colav[i].set_title(f"P(exit|COLAV) width={width}")
        axes_colav[i].set_xticks([0, 1, 2])
        axes_colav[i].set_xticklabels(STATES)
        axes_colav[i].set_xlabel('Torque')
        axes_colav[i].set_yticks([0, 1, 2])
        axes_colav[i].set_yticklabels(STATES)
        axes_colav[i].set_ylabel('Traffic')
        annotate(axes_colav[i], arr, vmin_i, vmax_i)
        cbar_i = fig_colav.colorbar(im, ax=axes_colav[i], shrink=0.85)
        cbar_i.set_label('Probability')

    try:
        out_path_colav = os.path.join(pathlib.Path(__file__).parents[2], 'figures', 'bbn_probabilities_colav.png')
        fig_colav.savefig(out_path_colav, dpi=150)
        print(f"Saved figure to {out_path_colav}")
    except Exception:
        pass

    # Figure 2: P(collision) with its own color scale
    fig_col, ax_col = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
    vmin_col = float(np.nanmin(col_collision))
    vmax_col = float(np.nanmax(col_collision))
    if vmin_col == vmax_col:
        vmin_col, vmax_col = vmin_col - 1e-12, vmax_col + 1e-12
    im_col = ax_col.imshow(col_collision, vmin=vmin_col, vmax=vmax_col, cmap='viridis', origin='upper')
    ax_col.set_title("P(collision)")
    ax_col.set_xticks([0, 1, 2])
    ax_col.set_xticklabels(STATES)
    ax_col.set_xlabel('Torque')
    ax_col.set_yticks([0, 1, 2])
    ax_col.set_yticklabels(STATES)
    ax_col.set_ylabel('Traffic')
    annotate(ax_col, col_collision, vmin_col, vmax_col)
    cbar_col = fig_col.colorbar(im_col, ax=ax_col, shrink=0.85)
    cbar_col.set_label('Probability')

    try:
        out_path_col = os.path.join(pathlib.Path(__file__).parents[2], 'figures', 'bbn_probabilities_collision.png')
        fig_col.savefig(out_path_col, dpi=150)
        print(f"Saved figure to {out_path_col}")
    except Exception:
        pass

    # Additional figure: Powered exit due to tracking as a heatmap (rows=sway-force, cols=width)
    tracking = np.zeros((3, 3), dtype=float)
    for r, sway in enumerate(STATES):
        for c, w in enumerate(STATES):
            tracking[r, c] = bbn.prob_powered_exit_tracking_per_sec(w, sway)

    fig2, ax2 = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True)
    vmin_t = float(np.nanmin(tracking))
    vmax_t = float(np.nanmax(tracking))
    if vmin_t == vmax_t:
        vmin_t, vmax_t = vmin_t - 1e-12, vmax_t + 1e-12
    im2 = ax2.imshow(tracking, vmin=vmin_t, vmax=vmax_t, cmap='viridis', origin='upper')
    ax2.set_title("P(exit|tracking)")
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(STATES)
    ax2.set_xlabel('Width')
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(STATES)
    ax2.set_ylabel('Sway force')
    annotate(ax2, tracking, vmin_t, vmax_t)
    cbar2 = fig2.colorbar(im2, ax=ax2, shrink=0.85)
    cbar2.set_label('Probability')

    try:
        out_path2 = os.path.join(pathlib.Path(__file__).parents[2], 'figures', 'bbn_probabilities_tracking.png')
        fig2.savefig(out_path2, dpi=150)
        print(f"Saved figure to {out_path2}")
    except Exception:
        pass

    plt.show()
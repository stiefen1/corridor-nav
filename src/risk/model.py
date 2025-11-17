"""
Goal: compute risk associated to a corridor given wave, current, wind, traffic density and own ship's parameters.

Traffic density, maneuverability and lateral (sway) force are classified as either LOW, MEDIUM, HIGH using two thresholds each.


"""

TD_THRESH = (1e-5, 1e-4)
MAN_THRESH = (1e3, 1e4)
SWAY_THRESH = (1e3, 1e4)
STATES = ('L', 'M', 'H')


import numpy as np, pandas as pd, os, pathlib
from typing import Literal



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

    def get(
            self,
            travel_time: float,
            traffic_density: float,
            forces: np.ndarray,
            width: float,
            ship_nominal_max_psi_force: float, # 
            ship_nominal_tracking_accuracy: float # +- meters
        ) -> float:

        # We define maneuverability as the maximum force the ship can generate
        maneuverability = np.max([ship_nominal_max_psi_force - np.abs(forces[2]), 0]) # Maximum force ship can generate 

        # Ship tracking accuracy <= nominal tracking accuracy
        prob_powered_exit_bad_tracking = ...
        # print(f"tracking accuracy: {ship_tracking_accuracy} | ship maneuverability: {ship_maneuverability} | forces[2]: {forces[2]}")

        # Actual width
        actual_width = np.clip(width - 2 * ship_tracking_accuracy, 0, np.inf)

        # Probability of collision (power + drifting)
        prob_powered_collision = np.exp(-ship_maneuverability / traffic_density / 1e5) # TODO: Evaluate probability of powered collision -> f(ship_maneuverability, traffic_density) -> we do not use width because traffic density account for this.
        prob_drifting_collision = 0
        prob_collision = prob_powered_collision + prob_drifting_collision

        # Probability of powered exit
        ## Because of bad tracking. travel_time -> infty <-> prob -> 1. actual_width -> infty <-> prob -> 0
        prob_powered_exit_bad_tracking = 1-np.exp(-travel_time / np.clip(actual_width, 1e-6, np.inf) / 1e4) # f(actual_width, travel_time) -> if tracking accuracy is +-2*sigma it means 95.5% error is <= tracking accuracy. So 4.5% of the time, we get outside of it -> maybe there is something to do here

        ## Because of COLAV
        prob_powered_exit_colav = 1-np.exp(-ship_maneuverability * width / traffic_density / 1e12) # TODO: Evaluate probability of powered exit -> f(ship_maneuverability, traffic_density, width)

        ## Total 
        prob_powered_exit = prob_powered_exit_bad_tracking + prob_powered_exit_colav
        prob_drifting_exit = 0

        # Probability of grounding
        prob_grounding = self.prob_grounding_given_exit * (prob_powered_exit + prob_drifting_exit)

        assert prob_grounding + prob_collision <= 1.0, f"Sum of probabilities must be <= 1.0. Got prob(grounding)={prob_grounding}, prob(collision)={prob_collision}"

        # print(prob_grounding, prob_collision)
        # print(prob_powered_exit_bad_tracking, prob_powered_exit_colav)

        # Expected travel time
        expected_travel_time = travel_time # prob_collision * self.t_collision + prob_grounding * self.t_grounding + (1-prob_collision-prob_grounding) * travel_time

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

    def prob_powered_exit_tracking(self, width: Literal['L', 'M', 'H'], sway_force: Literal['L', 'M', 'H']) -> float:
        """
        Convert width and sway force status into a probability of powered exit tracking based on tables.

        'L' = 'Low'
        'M' = 'Medium'
        'H' = 'High'
        """
        assert width in STATES, f"width must be one of the following str: {STATES}. Got width={width}."
        assert sway_force in STATES, f"sway_force must be one of the following str: {STATES}. Got sway_force={sway_force}."
        return self.powered_exit_tracking_table.loc[(self.powered_exit_tracking_table['width'] == 'L') & (self.powered_exit_tracking_table['sway-force'] == 'M'), 'p(exit)'].iloc[0]
    
    def prob_powered_exit_colav(self, traffic: Literal['L', 'M', 'H'], maneuverability: Literal['L', 'M', 'H']) -> float:
        assert traffic in STATES, f"traffic must be one of the following str: {STATES}. Got traffic={traffic}."
        assert maneuverability in STATES, f"maneuverability must be one of the following str: {STATES}. Got maneuverability={maneuverability}."
        return self.powered_exit_colav_table.loc[(self.powered_exit_colav_table['traffic'] == 'L') & (self.powered_exit_colav_table['maneuverability'] == 'M'), 'p(exit)'].iloc[0]
    
    def prob_powered_collision(self, traffic: Literal['L', 'M', 'H'], maneuverability: Literal['L', 'M', 'H']) -> float:
        assert traffic in STATES, f"traffic must be one of the following str: {STATES}. Got traffic={traffic}."
        assert maneuverability in STATES, f"maneuverability must be one of the following str: {STATES}. Got maneuverability={maneuverability}."
        return self.powered_collision_table.loc[(self.powered_collision_table['traffic'] == 'L') & (self.powered_collision_table['maneuverability'] == 'M'), 'p(collision)'].iloc[0]

if __name__ == "__main__":
    bbn = BBN()
    print(bbn.prob_powered_exit_tracking('L', 'M'))
    print(bbn.prob_powered_exit_colav('L', 'M'))
    print(bbn.prob_powered_collision('L', 'M'))
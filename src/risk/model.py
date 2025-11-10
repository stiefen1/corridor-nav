"""
Goal: compute risk associated to a corridor given wave, current, wind, traffic density and own ship's parameters.
"""
import numpy as np

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
            ship_nominal_maneuverability: float, # 
            ship_nominal_tracking_accuracy: float # +- meters
        ) -> float:

        # Ship maneuverability
        ship_maneuverability = ... # TODO: Evaluate ship maneuverability -> f(ship_nominal_maneuverability, forces[2])

        # Ship tracking accuracy
        ship_tracking_accuracy = ... # TODO: Evaluate tracking accuracy -> f(ship_nominal_tracking_accuracy, forces[1])

        # Actual width
        actual_width = max([width - 2 * ship_nominal_tracking_accuracy, 0])

        # Probability of collision (power + drifting)
        prob_powered_collision = ... # TODO: Evaluate probability of powered collision -> f(ship_maneuverability, traffic_density, width) -> we do not use actual width because in COLAV mode we're not doing path tracking (usually)
        prob_drifting_collision = 0
        prob_collision = prob_powered_collision + prob_drifting_collision


        # Probability of powered exit
        ## Because of bad tracking
        prob_powered_exit_bad_tracking = ... # f(actual_width, travel_time) -> if tracking accuracy is +-2*sigma it means 95.5% error is <= tracking accuracy. So 4.5% of the time, we get outside of it -> maybe there is something to do here

        ## Because of COLAV
        prob_powered_exit_colav = ... # TODO: Evaluate probability of powered exit -> f(ship_maneuverability, traffic_density, width)

        ## Total 
        prob_powered_exit = prob_powered_exit_bad_tracking + prob_powered_exit_colav
        prob_drifting_exit = 0

        # Probability of grounding
        prob_grounding = self.prob_grounding_given_exit * (prob_powered_exit + prob_drifting_exit)

        assert prob_grounding + prob_collision <= 1.0, f"Sum of probabilities must be <= 1.0. Got prob(grounding)={prob_grounding}, prob(collision)={prob_collision}"

        # Expected travel time
        expected_travel_time = prob_collision * self.t_collision + prob_grounding * self.t_grounding + (1-prob_collision-prob_grounding) * travel_time

        return expected_travel_time
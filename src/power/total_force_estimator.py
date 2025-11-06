"""
Goal: Compute the total force needed for the ship to maintain u=u_des with u_dot, v_dot, n_dot = 0
It is basically computing the force required to maintain the ship at constant speed in the corridor
using the ship's dynamics. 
"""

from power import WaveForceEstimator, WindForceEstimator, DefaultWaveParameters, DefaultWindParameters
from dataclasses import dataclass, asdict
from typing import Dict
import numpy as np

@dataclass
class DefaultShipParams:
    # mass components
    coeff_of_deadweight_to_displacement: float = 0.7
    bunkers: float = 200_000
    ballast: float = 200_000
    dead_weight_tonnage: float = 3_850_000

    # center of gravity
    x_g: float = 0

    # Dimensions
    loa: float = 80
    beam: float = 16
    draft: float = 5

    # Added mass coefficients
    surge_coeff: float = 0.4
    sway_coeff: float = 0.4
    yaw_coeff: float = 0.4

    # linear damping
    tu: float = 130
    tv: float = 18
    tr: float = 90

    # nonlinear damping
    ku: float = 2400
    kv: float = 4000
    kr: float = 400

    def __post_init__(self):
        # mass
        payload = 0.9 * (self.dead_weight_tonnage - self.bunkers)
        lsw = self.dead_weight_tonnage / self.coeff_of_deadweight_to_displacement - self.dead_weight_tonnage
        self.mass = lsw + payload + self.bunkers + self.ballast

        # Inertia
        self.i_z = self.mass * (self.loa ** 2 + self.beam ** 2) / 12

        # Added mass
        self.x_du: float = self.mass * self.surge_coeff
        self.y_dv: float = self.mass * self.sway_coeff
        self.n_dr: float = self.i_z * self.yaw_coeff



class ForceEstimator:
    """
    Class to compute the total force required to maintain the desired speed of the ship given environmental loads. 
    """
    def __init__(
            self,
            ship_params = DefaultShipParams(),
            wave_params: Dict = asdict(DefaultWaveParameters()),
            wind_params: Dict = asdict(DefaultWindParameters())
    ):
        self.ship_params = ship_params
        self.wave_force_estimator = WaveForceEstimator(ship_params.loa, ship_params.beam, ship_params.draft, **wave_params)
        self.wind_force_estimator = WindForceEstimator(ship_params.loa, ship_params.beam, **wind_params)


    def mass_matrix(self):
        return np.array([[self.ship_params.mass + self.ship_params.x_du, 0, 0],
                         [0, self.ship_params.mass + self.ship_params.y_dv, self.ship_params.mass * self.ship_params.x_g],
                         [0, self.ship_params.mass * self.ship_params.x_g, self.ship_params.i_z + self.ship_params.n_dr]])

    def coriolis_matrix(self, u: float, v: float = 0, r: float = 0):
        return np.array([[0, 0, -self.ship_params.mass * (self.ship_params.x_g * r + v)],
                         [0, 0, self.ship_params.mass * u],
                         [self.ship_params.mass * (self.ship_params.x_g * r + v),
                          -self.ship_params.mass * u, 0]])

    def coriolis_added_mass_matrix(self, u_r: float, v_r: float):
        return np.array([[0, 0, self.ship_params.y_dv * v_r],
                        [0, 0, -self.ship_params.x_du * u_r],
                        [-self.ship_params.y_dv * v_r, self.ship_params.x_du * u_r, 0]])
    
    def non_linear_damping_matrix(self, u: float, v: float = 0, r: float = 0):
        return np.array([[self.ship_params.ku * u, 0, 0],
                       [0, self.ship_params.kv * v, 0],
                       [0, 0, self.ship_params.kr * r]])
    
    def linear_damping_matrix(self):
        return np.array([[self.ship_params.mass / self.ship_params.tu, 0, 0],
                      [0, self.ship_params.mass / self.ship_params.tv, 0],
                      [0, 0, self.ship_params.i_z / self.ship_params.tr]])
    
    def rotation(self, psi: float):
        ''' Specifies the rotation matrix for rotations about the z-axis, such that
            "body-fixed coordinates" = rotation x "North-east-down-fixed coordinates" .
        '''
        return np.array([[np.cos(psi), -np.sin(psi), 0],
                         [np.sin(psi), np.cos(psi), 0],
                         [0, 0, 1]])

    def get(
            self,
            u: float,
            psi: float,
            wind_speed: float,
            wind_dir: float,
            current_speed: float,
            current_dir: float,
            significant_wave_height: float,
            wave_dir: float,
            v: float = 0,
            r: float = 0,
            degrees=False
        ) -> np.ndarray:
        """
        Returns the generalized force that must be produced by actuators to maintain actual speed. 
        """
        current_dir = np.deg2rad(current_dir) if degrees else current_dir
        wind_dir = np.deg2rad(wind_dir) if degrees else wind_dir
        wave_dir = np.deg2rad(wave_dir) if degrees else wave_dir
        psi = np.deg2rad(psi) if degrees else psi

        # Environmental loads
        tau_wave = self.wave_force_estimator.get(u, psi, significant_wave_height, wave_dir)
        tau_wind = self.wind_force_estimator.get(u, psi, wind_speed, wind_dir, v=v)
        tau_ext = tau_wave + tau_wind

        # Ship velocity
        vel = np.array([u, v, r])

        # Current velocity
        current_vel = current_speed * np.array([
            np.cos(current_dir), # North
            np.sin(current_dir), # East
            0
        ])
        current_vel_in_ship_frame = self.rotation(psi).T @ current_vel
        vel_rel = vel - current_vel_in_ship_frame

        nu_dot_des = np.array([0, 0, 0])

        # Dynamics: M @ nu_dot + C(nu) @ nu + D(nu) = tau_ext + tau_actuators
        tau_actuators = self.mass_matrix() @ nu_dot_des \
            + np.dot(self.coriolis_matrix(u, v), vel) \
            + np.dot(self.coriolis_added_mass_matrix(*vel_rel[0:2].tolist()), vel_rel) \
            + np.dot(self.linear_damping_matrix() + self.non_linear_damping_matrix(*vel_rel.tolist()), vel_rel) \
            - tau_ext # tau = tau_ext + tau_actuators to guarantee nu_dot = 0
        
        return tau_actuators 
        

if __name__ == "__main__":
    f_est = ForceEstimator()
    print(f_est.get(2, 30, 2, 120, 1, 0, 0, 0, degrees=True))


import numpy as np
from dataclasses import dataclass

RHO_AIR = 1.2

@dataclass
class DefaultWindParameters: # From ship_in_transit simulator
    # loa: float = 80
    # beam: float = 16
    c_x: float = 0.5
    c_y: float = 0.7
    c_n: float = 0.08
    h_front: float = 8.0 # height of the ship above the waterline 
    h_lateral: float = 8.0

class WindForceEstimator:
    def __init__(
            self,
            loa: float,
            beam: float, 
            c_x: float = 0.5, # ship_in_transit parameters
            c_y: float = 0.7,
            c_n: float = 0.08,
            h_front: float = 8.0, # height of the ship above the waterline 
            h_lateral: float = 8.0
    ):
        self.loa = loa
        self.beam = beam
        self.c_x = c_x
        self.c_y = c_y
        self.c_n = c_n
        self.proj_area_f = self.beam * h_front # front/lateral projected areas
        self.proj_area_l = self.loa * h_lateral
        
    def get(
            self,
            u: float,
            psi: float,
            wind_speed: float,
            wind_dir: float,
            v: float = 0,
            degrees: float = False
        ) -> np.ndarray:
        """
        Angle are given w.r.t north, clockwise positive. Speed is assumed to be exclusively in the surge direction.
        """
        psi = np.deg2rad(psi) if degrees else psi
        wind_dir = np.deg2rad(wind_dir) if degrees else wind_dir

        # wind in ship frame
        u_w = wind_speed * np.cos(wind_dir - psi)
        v_w = wind_speed * np.sin(wind_dir - psi)

        # relative wind in ship frame
        V_rw = np.array([
            u - u_w,
            v - v_w
        ])

        # gamma_rw: Angle between wind vector and heading in ship frame
        gamma_rw = -np.atan2(V_rw[1], V_rw[0])

        # Coefficient approximation for symmetrical ships (Thor I. Fossen p.191, 1st edition)
        C_x = - self.c_x * np.cos(gamma_rw) 
        C_y = self.c_y * np.sin(gamma_rw) 
        C_n = self.c_n * np.sin(2*gamma_rw) 

        # Generalized force        
        tau = 0.5 * RHO_AIR * np.linalg.norm(V_rw)**2 * np.array([
            C_x * self.proj_area_f,
            C_y * self.proj_area_l,
            C_n * self.proj_area_l * self.loa
        ]) 

        return tau
    
if __name__ == "__main__":
    from dataclasses import asdict

    f_wind_est = WindForceEstimator(
        loa=80,
        beam=16,
        **asdict(DefaultWindParameters())
    )
    print(f_wind_est.get(0, 90, 1, 0, degrees=True))
import numpy as np
from dataclasses import dataclass
from power.wave_model import WaveModelConfiguration, JONSWAPWaveModel, RHO, GRAVITY

SIGNIFICANT_WAVE_HEIGHT_TABLE = [(0, 0.5), (0.5, 1.25), (1.25, 2.5), (2.5, 4), (4, 6), (6, 9), (9, 14), (9, float('inf'))]
PEAK_PERIOD_TABLE = [7.5, 7.5, 8.8, 9.7, 12.4, 15, 16.4, 20]

@dataclass
class DefaultWaveParameters:
    w_min: float = 0.4
    w_max: float = 2.5
    n_omega: int = 50
    spread_angle_min: float = -np.pi
    spread_angle_max: float = np.pi
    n_spread: int = 10
    spreading_coeff: int = 1

class WaveForceEstimator:
    """
    JONSWAP model (better suited for inland waterways)
    """
    def __init__(
            self,
            loa: float, # 80
            beam: float, # 16
            draft: float, # 5
            w_min: float,
            w_max: float,
            n_omega: int,
            spread_angle_min: float,
            spread_angle_max: float,
            n_spread: int,
            spreading_coeff: int,
    ):
        # Ship's dimensions
        self.loa = loa
        self.beam = beam
        self.draft = draft

        # omegas
        self.omega_vec = np.linspace(w_min, w_max, n_omega)
        self.delta_omega = self.omega_vec[1] - self.omega_vec[0]
        self.k_vec = self.omega_vec**2 / GRAVITY
        
        # spread angles
        self.spread_vec = np.linspace(spread_angle_min, spread_angle_max, n_spread)
        self.delta_spread = self.spread_vec[1] - self.spread_vec[0]
        
        # Random phase vector across all spread angles and frequencies
        self.theta = 2 * np.pi * np.random.rand(n_omega, n_spread)

        # Wave model
        self.wave_model = JONSWAPWaveModel(WaveModelConfiguration(
            w_min,
            w_max,
            n_omega,
            spread_angle_min,
            spread_angle_max,
            n_spread,
            spreading_coeff
        ))


    def get(self, u: float, psi: float, H_s: float, wave_dir: float, T_p: float | None = None, degrees: bool = False) -> np.ndarray:
        '''
        Parameters:
        u: float
            Ship forward speed
        psi: float
            Ship heading
        H_s : float
            Significant wave height [m].
        T_p : float
            Peak period [s].
        wave_dir : float
            Mean wave direction [rad | deg]
            
        Returns (F_wx, F_wy): First order wave load computed with Froude-Krylof force approximation. [N]
        '''
        if T_p is None:
            idx = sum([i if H_s in h_range else 0 for i, h_range in enumerate(SIGNIFICANT_WAVE_HEIGHT_TABLE)])
            T_p = PEAK_PERIOD_TABLE[idx]

        psi = np.deg2rad(psi) if degrees else psi
        wave_dir = np.deg2rad(psi) if degrees else wave_dir
        return self.wave_model.get_direct_wave_force(
            u, psi, self.loa, self.beam, self.draft, H_s, T_p, wave_dir
        )
    
if __name__ == "__main__":
    from dataclasses import asdict
    wave_force_est = WaveForceEstimator(80, 16, 5, **asdict(DefaultWaveParameters()))
    print(wave_force_est.get(1, 0, 2.5, 1))

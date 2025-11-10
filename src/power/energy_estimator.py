from corridor_opt.corridor import Corridor
from power.total_force_estimator import ForceEstimator
from typing import Dict, Optional
import numpy as np

class EnergyEstimator:
    """
    Convert a generalized force to be applied on the own ship into an energy consumption estimation
    """
    def __init__(
            self,
            params: Optional[Dict] = None
    ):
        self.params = params

    def get(self, corridor: Corridor, force: np.ndarray, travel_time: float) -> float:
        return 0
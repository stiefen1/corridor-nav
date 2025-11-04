"""
Goal: compute risk associated to a corridor given wave, current, wind, traffic density and own ship's parameters.
"""
from typing import Any

class RiskModel:
    def __init__(
            self
    ):
        pass

    def get(self, own_ship: Any, traffic_density: Any, wind: Any, wave: Any, current: Any) -> float:
        return 0.0
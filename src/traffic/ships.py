"""
Example of dataclasses to describe target / own ships. 
"""

from dataclasses import dataclass

@dataclass
class TargetShip:
    north: float
    east: float
    loa: float
    beam: float
    ...

@dataclass
class OwnShip:
    north: float
    east: float
    loa: float
    beam: float
    draft: float
    available_power: float
    ...
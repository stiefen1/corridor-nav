"""
Example of dataclasses to describe target / own ships. 
"""

from dataclasses import dataclass
from datetime import datetime

@dataclass
class TargetShipOld:
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

@dataclass
class TargetShip:

    courseOverGround: float
    latitude: float
    longitude: float
    name: str
    rateOfTurn: float
    shipType: float
    speedOverGround: float
    trueHeading: float
    navigationalStatus: float
    mmsi: float
    msgtime: datetime
    


    # #Example Query Received
    # courseOverGround: 316.2
    # latitude: 63.10453
    # longitude: 7.769148
    # name: 'SEVEN OCEANIC'
    # rateOfTurn: 0
    # shipType: 70 
    # speedOverGround: 0
    # trueHeading: 176
    # navigationalStatus: 5
    # mmsi: 232026676
    # msgtime: '2025-11-03T22:59:11+00:00'
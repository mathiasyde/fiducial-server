from dataclasses import dataclass


@dataclass
class Marker:
    x: float
    y: float
    z: float
    euler_x: float
    euler_y: float
    euler_z: float

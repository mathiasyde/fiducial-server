from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import toml

from server.utils import make_address


@dataclass
class Config:
    source: Any = None

    # calibration
    homography_top_left: int = 0
    homography_top_right: int = 0
    homography_bottom_left: int = 0
    homography_bottom_right: int = 0

    # detecting
    camera_focal_x: float = 0
    camera_focal_y: float = 0
    camera_center_x: float = 0
    camera_center_y: float = 0
    tag_size: float = 0.0

    # detector
    tag_families: list[str] = field(default_factory=lambda: ["tag36h11"])
    nthreads: int = 1
    quad_decimate: float = 2.0
    quad_sigma: float = 0.0
    refine_edges: bool = True
    decode_sharpening: float = 0.25
    debug: bool = False

    # network
    udp: tuple[str, int] = field(default_factory=lambda: ("localhost", 5005))
    tcp: tuple[str, int] = field(default_factory=lambda: ("localhost", 5006))

    @staticmethod
    def from_dict(data: dict) -> Config:
        values = {
            key: value for section in data.values() for key, value in section.items()
        }

        values["udp"] = make_address(values["udp"].split(":"))
        values["tcp"] = make_address(values["tcp"].split(":"))

        return Config(**values)

    @staticmethod
    def from_file(path: Path) -> Config:
        data = toml.load(path)
        return Config.from_dict(data)

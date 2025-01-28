from dataclasses import dataclass

import numpy as np


@dataclass
class FractalSettings:
    u: list  # vector 1
    o: list  # origin
    v: list  # point 2
    center: tuple
    rotation: float
    scale: float
    escape_counts: np.ndarray | None = None
    sampled_points: np.ndarray | None = None

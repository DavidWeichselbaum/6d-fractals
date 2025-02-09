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
    base_iterations: float
    iterations_growth: float
    escape_counts: np.ndarray | None = None
    sampled_points: np.ndarray | None = None


default_settings = FractalSettings(
    u=np.array([1, 0, 0, 0, 2, 0], dtype=np.float64),
    o=np.array([0, 0, 0, 0, 2, 0], dtype=np.float64),
    v=np.array([0, 1, 0, 0, 2, 0], dtype=np.float64),
    center=(-0.5, 0),
    rotation=0,
    scale=2.5,
    base_iterations=119,
    iterations_growth=20,
)


def settings_to_dict(settings):
    """Convert FractalSettings to a dictionary for YAML serialization."""
    return {
        "points": {
            "u": settings.u.tolist(),  # Convert numpy array to list
            "o": settings.o.tolist(),  # Convert numpy array to list
            "v": settings.v.tolist(),  # Convert numpy array to list
        },
        "center": list(settings.center),  # Convert tuple to list for YAML compatibility
        "rotation": settings.rotation,
        "scale": settings.scale,
        "base_iterations": settings.base_iterations,
    }


def dict_to_settings(settings_dict):
    """Convert a dictionary to a FractalSettings object."""
    points = settings_dict["points"]
    return FractalSettings(
        u=np.array(points["u"]),  # Convert list back to numpy array
        o=np.array(points["o"]),  # Convert list back to numpy array
        v=np.array(points["v"]),  # Convert list back to numpy array
        center=tuple(settings_dict["center"]),  # Convert list back to tuple
        rotation=settings_dict["rotation"],
        scale=settings_dict["scale"],
        base_iterations=settings_dict["base_iterations"],
    )

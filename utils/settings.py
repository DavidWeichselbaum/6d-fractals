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
    escape_radius: float
    colormap: str
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
    escape_radius=2,
    colormap="inferno",
)


def settings_to_dict(settings):
    """Convert FractalSettings to a dictionary for YAML serialization."""
    return {
        "basis": {
            "u": settings.u.tolist(),
            "o": settings.o.tolist(),
            "v": settings.v.tolist(),
        },
        "location": {
            "center": {
                "x": settings.center[0],
                "y": settings.center[1],
            },
            "rotation": settings.rotation,
            "scale": settings.scale,
        },
        "computation": {
            "iterations": {
                "base": settings.base_iterations,
                "growth": settings.iterations_growth,
            },
            "radius": settings.escape_radius,
        },
        "presentation": {
            "colormap": settings.colormap,
        },
    }


def dict_to_settings(settings_dict):
    """Convert a dictionary to a FractalSettings object."""
    basis = settings_dict["basis"]
    location = settings_dict["location"]
    center = location["center"]
    computation = settings_dict["computation"]
    iterations = computation["iterations"]
    presentation = settings_dict["presentation"]
    return FractalSettings(
        u=np.array(basis["u"]),  # Convert list back to numpy array
        o=np.array(basis["o"]),  # Convert list back to numpy array
        v=np.array(basis["v"]),  # Convert list back to numpy array
        center=(center["x"], center["y"]),
        rotation=location["rotation"],
        scale=location["scale"],
        base_iterations=iterations["base"],
        iterations_growth=iterations["growth"],
        escape_radius=computation["radius"],
        colormap=presentation["colormap"],
    )

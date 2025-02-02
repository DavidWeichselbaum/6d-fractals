import numpy as np

from utils.datatypes import FractalSettings


def settings_to_dict(settings):
    """Convert FractalSettings to a dictionary for YAML serialization."""
    return {
        "u": settings.u.tolist(),  # Convert numpy array to list
        "o": settings.o.tolist(),  # Convert numpy array to list
        "v": settings.v.tolist(),  # Convert numpy array to list
        "center": list(settings.center),  # Convert tuple to list for YAML compatibility
        "rotation": settings.rotation,
        "scale": settings.scale,
        "escape_counts": None,  # Skip escape_counts for serialization
    }


def dict_to_settings(settings_dict):
    """Convert a dictionary to a FractalSettings object."""
    return FractalSettings(
        u=np.array(settings_dict["u"]),  # Convert list back to numpy array
        o=np.array(settings_dict["o"]),  # Convert list back to numpy array
        v=np.array(settings_dict["v"]),  # Convert list back to numpy array
        center=tuple(settings_dict["center"]),  # Convert list back to tuple
        rotation=settings_dict["rotation"],
        scale=settings_dict["scale"],
    )

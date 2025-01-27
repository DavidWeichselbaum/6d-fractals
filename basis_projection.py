import sys
import yaml

import matplotlib.pyplot as plt
import numpy as np

from utils.datatypes import FractalSettings


save_path = sys.argv[1]

def project_to_2d(v, o, u, target_vector):
    # Compute the basis vectors of the plane
    p1 = v - o
    p2 = u - o

    # Form the dot product matrix M (Gram matrix)
    M = np.array([
        [np.dot(p1, p1), np.dot(p1, p2)],
        [np.dot(p2, p1), np.dot(p2, p2)]
    ])

    # Center the target vector relative to the plane origin
    target_prime = target_vector - o

    # Compute the RHS of the linear system
    b = np.array([np.dot(target_prime, p1), np.dot(target_prime, p2)])

    # Solve for the 2D projection coefficients
    a = np.linalg.solve(M, b)
    return a


def project_basis_to_2d(v, o, u):
    # Define standard basis vectors in R^6 using np.eye
    basis_vectors = np.eye(6)

    # Map the projections to named basis vectors (ca, cb, za, zb, pa, pb)
    names = ["ca", "cb", "za", "zb", "pa", "pb"]
    projections = {
        names[i]: project_to_2d(v, o, u, basis_vectors[i]) for i in range(6)
    }
    return projections


def plot_2d_projections(projections):
    styles = {
        "ca": ("red", "-"),
        "cb": ("red", ":"),
        "za": ("green", "-"),
        "zb": ("green", ":"),
        "pa": ("blue", "-"),
        "pb": ("blue", ":"),
    }
    max_length = max(np.linalg.norm(coords) for coords in projections.values())
    limit = 1.5 * max_length

    plt.figure(figsize=(8, 8))
    for name, coords in projections.items():
        color, linestyle = styles[name]
        plt.arrow(
            0, 0, coords[0], coords[1],
            head_width=0.05, head_length=0.05, fc=color, ec=color, linestyle=linestyle
        )
        plt.text(coords[0] * 1.1, coords[1] * 1.1, f" {name}", fontsize=12, color=color)

    plt.xlabel("Pixels x")
    plt.ylabel("Pixels y")
    plt.title("Projections of Basis Vectors on Pixel Space")
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    plt.show()


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

with open(save_path, "r") as file:
    settings_dict = yaml.safe_load(file)
    settings = dict_to_settings(settings_dict)

projections = project_basis_to_2d(settings.v, settings.o, settings.u)
plot_2d_projections(projections)

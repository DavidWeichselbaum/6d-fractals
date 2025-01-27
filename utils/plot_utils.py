import sys
import yaml

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from io import BytesIO
from PIL import Image
from matplotlib.patches import FancyArrowPatch

from utils.datatypes import FractalSettings


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
    basis_vectors = np.eye(6)
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
    limit = 1.5 * max_length  # Set the limits to 1.5 times the maximum length

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    fig.patch.set_alpha(0.0)  # Transparent background
    ax.set_alpha(0.0)  # Transparent plot area

    for name, coords in projections.items():
        color, linestyle = styles[name]
        start = np.array([0, 0])
        end = np.array(coords)

        # Offset starting and end point slightly
        offset_fraction = 0.05
        offset = offset_fraction * (end - start)
        line_start = start + offset
        line_end = end - offset

        ax.plot(
            [line_start[0], line_end[0]], [line_start[1], line_end[1]],
            linestyle=linestyle, color=color, linewidth=2, zorder=1
        )
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle='-|>', color=color,
            mutation_scale=50, linewidth=0, zorder=2
        )
        ax.add_patch(arrow)

        # ax.text(coords[0] * 1.1, coords[1] * 1.1, f" {name}", fontsize=12, color=color)

    ax.axis('off')
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    plt.gca().set_aspect('equal', adjustable='box')

    return fig


def render_figure_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight")
    buf.seek(0)
    image = Image.open(BytesIO(buf.read()))  # Create a new buffer to keep the image open
    buf.close()
    plt.close(fig)
    return image


def get_basis_projection_image(settings):
    projections = project_basis_to_2d(settings.v, settings.o, settings.u)
    fig = plot_2d_projections(projections)
    return render_figure_to_image(fig)


if __name__ == "__main__":
    from utils.utils import dict_to_settings

    save_path = sys.argv[1]

    with open(save_path, "r") as file:
        settings_dict = yaml.safe_load(file)
        settings = dict_to_settings(settings_dict)

    image = get_basis_projection_image(settings)
    image.show()

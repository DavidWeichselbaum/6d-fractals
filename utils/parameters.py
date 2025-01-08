import numpy as np

import numpy as np


def sample_plane(u, o, v, center=(0.0, 0.0), rotation=0.0, scale=4.0, resolution=(500, 500)):
    """
    Sample a 2D plane in R^4 defined by three points: u, v, and o (origin point).
    """
    w, h = resolution
    x, y = center

    # Compute vectors spanning the plane
    u_prime = np.array(u) - np.array(o)
    v_prime = np.array(v) - np.array(o)
    n_components = len(u)

    # Sampling grid
    aspect_ratio = w / h
    s = np.linspace(-0.5 * aspect_ratio, 0.5 * aspect_ratio, w)
    t = np.linspace(-0.5, 0.5, h)
    S, T = np.meshgrid(s, t)

    # Rotate, scale, translate
    rotation_matrix = np.array([[np.cos(rotation), -np.sin(rotation)],
                                 [np.sin(rotation),  np.cos(rotation)]])
    points = np.stack([S.ravel(), T.ravel()])  # Shape (2, N)
    rotated_points = rotation_matrix @ points
    S = rotated_points[0, :].reshape(S.shape) * scale + x
    T = rotated_points[1, :].reshape(T.shape) * scale + y

    # Compute points in the plane
    grid = np.stack([S, T], axis=-1).reshape(-1, 2)  # Shape (w*h, 2)
    points = np.array(o) + grid[:, 0, None] * u_prime + grid[:, 1, None] * v_prime
    points = points.reshape(h, w, n_components)

    # Convert to complex points
    complex_points = points[..., ::2] + 1j * points[..., 1::2]

    return complex_points

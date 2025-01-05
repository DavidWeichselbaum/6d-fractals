import numpy as np


def sample_plane(u, o, v, center=(0.0, 0.0), rotation=0.0, scale=4.0, resolution=(500, 500)):
    """
    Sample a 2D plane in R^4 defined by three points: u, v, and o (origin point).
    """

    w, h = resolution
    x, y = center
    n_components = len(u)

    # Compute vectors spanning the plane
    u_prime = np.array(u) - np.array(o)
    v_prime = np.array(v) - np.array(o)

    # Sampling grid
    aspect_ratio = w / h
    s = np.linspace(-0.5 * aspect_ratio, 0.5 * aspect_ratio, w)
    t = np.linspace(-0.5, 0.5, h)
    S, T = np.meshgrid(s, t)

    # Rotate, scale, translate
    rotation_matrix = np.array([[np.cos(rotation), -np.sin(rotation)],
                                 [np.sin(rotation),  np.cos(rotation)]])
    points = np.vstack([S.ravel(), T.ravel()])  # Shape (2, N)
    rotated_points = rotation_matrix @ points  # Matrix multiplication
    S = rotated_points[0, :].reshape(S.shape)
    T = rotated_points[1, :].reshape(T.shape)
    S = S * scale
    T = T * scale
    S = S + x
    T = T + y

    # Compute points in the plane
    points = np.zeros((h, w, n_components))
    for i in range(h):
        for j in range(w):
            points[i, j, :] = np.array(o) + S[i, j] * u_prime + T[i, j] * v_prime

    # Convert to complex points
    complex_points = np.zeros((h, w, n_components // 2), dtype=complex)
    for i in range(n_components // 2):
        real_part = points[..., i*2]
        imaginary_part = points[..., i*2+1]
        complex_points[..., i] = real_part + 1j * imaginary_part
    return complex_points

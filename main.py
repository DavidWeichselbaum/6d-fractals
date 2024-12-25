from time import time

import numpy as np
import matplotlib.pyplot as plt
from numba import njit


def sample_plane(u, v, o, center, rotation, scale, resolution):
    """
    Sample a 2D plane in R^4 defined by three points: u, v, and o (origin point).

    Parameters:
        u (array-like): Point defining the plane in R^4.
        v (array-like): Point defining the plane in R^4.
        o (array-like): Origin point of the plane in R^4.
        center (tuple): Center of the rectangle in the plane (x, y).
        rotation (float): Rotation angle in radians.
        scale (float): Scale factor for the rectangle.
        resolution (tuple): Resolution of the sampling grid (w, h).

    Returns:
        np.ndarray: Array of sampled points of shape (h, w, 4).
    """
    w, h = resolution
    x, y = center

    # Compute vectors spanning the plane
    u_prime = np.array(u) - np.array(o)
    v_prime = np.array(v) - np.array(o)

    # Normalize basis vectors
    u_prime = u_prime / np.linalg.norm(u_prime)
    v_prime = v_prime - np.dot(v_prime, u_prime) * u_prime
    v_prime = v_prime / np.linalg.norm(v_prime)

    # Scale and rotate
    u_s = scale * u_prime
    v_s = scale * v_prime
    rotation_matrix = np.array([[np.cos(rotation), -np.sin(rotation)],
                                 [np.sin(rotation),  np.cos(rotation)]])
    u_r = rotation_matrix[0, 0] * u_s + rotation_matrix[0, 1] * v_s
    v_r = rotation_matrix[1, 0] * u_s + rotation_matrix[1, 1] * v_s

    # Sampling grid
    s = np.linspace(-0.5, 0.5, w) + x / scale
    t = np.linspace(-0.5, 0.5, h) + y / scale
    S, T = np.meshgrid(s, t)

    # Compute points in the plane
    points = np.zeros((h, w, 4))
    for i in range(h):
        for j in range(w):
            points[i, j, :] = np.array(o) + S[i, j] * u_r + T[i, j] * v_r

    return points


@njit
def compute_fractal(c_z_array, max_iterations=100, escape_radius=2):
    """
    Computes a fractal based on the formula z = z^2 + c.

    Args:
        c_z_array: A 2D array of [c, z_0] values, where c and z_0 are complex numbers.
        max_iterations: Maximum number of iterations to run for each point.
        escape_radius: The magnitude at which the iteration is considered to have escaped.

    Returns:
        A 2D array of iteration counts for each point, indicating how quickly it escaped.
    """
    height, width, _ = c_z_array.shape
    escape_counts = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            c, z = c_z_array[i, j]
            count = 0
            while abs(z) <= escape_radius and count < max_iterations:
                z = z**2 + c
                count += 1
            escape_counts[i, j] = count

    return escape_counts


# # # mandelbrot
# u = [1, 0, 0, 0]
# v = [0, 1, 0, 0]
# o = [0, 0, 0, 0]

# # julia
# # u = [0.45, 0.1428, 1, 0]
# # v = [0.45, 0.1428, 0, 1]
# # o = [0.45, 0.1428, 0, 0]

# center = (-0.4, 0)  # Center at the origin
# rotation = 0  # No rotation
# scale = 3.0     # Unit scale
# resolution = (500, 500)  # 5x5 grid

# sampled_points = sample_plane(u, v, o, center, rotation, scale, resolution)
# complex_points = np.zeros((resolution[0], resolution[1], 2), dtype=complex)
# complex_points[..., 0] = sampled_points[..., 0] + 1j * sampled_points[..., 1]
# complex_points[..., 1] = sampled_points[..., 2] + 1j * sampled_points[..., 3]
# print(complex_points)
# print(complex_points.shape)

# t1 = time()
# fractal = compute_fractal(complex_points)
# t2 = time()
# print(t2-t1)

# plt.figure(figsize=(10, 10))
# plt.imshow(fractal, cmap='inferno')
# plt.colorbar(label="Iterations")
# plt.show()


for i in (2**2, 2**1, 2**-1, 2**-2, 2**-3, 2**-4):
    u = [1, 0, 0, 0]
    v = [0, 1, 0, 0]
    o = [0, 0, 0, 0]

    center = (-1.5, -0.1)  # Center at the origin
    rotation = 0.1  # No rotation
    scale = i     # Unit scale
    resolution = (500, 500)  # 5x5 grid

    sampled_points = sample_plane(u, v, o, center, rotation, scale, resolution)
    complex_points = np.zeros((resolution[0], resolution[1], 2), dtype=complex)
    complex_points[..., 0] = sampled_points[..., 0] + 1j * sampled_points[..., 1]
    complex_points[..., 1] = sampled_points[..., 2] + 1j * sampled_points[..., 3]

    t1 = time()
    fractal = compute_fractal(complex_points)
    t2 = time()
    print(i)
    print(t2-t1)

    plt.figure(figsize=(10, 10))
    plt.imshow(fractal, cmap='inferno')
    plt.colorbar(label="Iterations")
    plt.show()

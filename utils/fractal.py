import numpy as np
from numba import njit, prange


@njit(parallel=True)
def compute_fractal(c_z_p_array, max_iterations=100, escape_radius=2):
    """
    Compute a fractal based on the formula z = z^e + c.
    """
    height, width, _ = c_z_p_array.shape
    escape_counts = np.zeros((height, width), dtype=np.int32)

    for i in prange(height):  # parallelized
        for j in range(width):
            c, z, p = c_z_p_array[i, j]
            count = 0
            while abs(z) <= escape_radius and count < max_iterations:
                if not z:
                    z = c
                z = z**p + c
                count += 1
            escape_counts[i, j] = count

    return escape_counts

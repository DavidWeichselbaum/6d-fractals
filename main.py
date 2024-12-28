from time import time
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from numba import njit
from copy import deepcopy


@dataclass
class FractalSettings:
    name: str
    u: list
    v: list
    o: list
    center: tuple
    rotation: float
    scale: float
    resolution: tuple
    start_iterations: int
    iteration_growth: int
    escape_radius: float


class FractalRenderer:

    MIN_RECTANGLE = (5, 5)

    def __init__(self, settings):
        self.settings = settings
        self.history = [deepcopy(self.settings)]

        self.fig, self.ax = None, None
        self.image, self.colorbar = None, None
        self.rect_selector = None

    def sample_plane(self):
        """
        Sample a 2D plane in R^4 defined by three points: u, v, and o (origin point).
        """
        u, v, o, center, rotation, scale, resolution = (
            self.settings.u,
            self.settings.v,
            self.settings.o,
            self.settings.center,
            self.settings.rotation,
            self.settings.scale,
            self.settings.resolution,
        )

        w, h = resolution
        x, y = center
        n_components = len(u)

        # Compute vectors spanning the plane
        u_prime = np.array(u) - np.array(o)
        v_prime = np.array(v) - np.array(o)

        # Normalize basis vectors
        u_prime = u_prime / np.linalg.norm(u_prime)
        v_prime = v_prime - np.dot(v_prime, u_prime) * u_prime
        v_prime = v_prime / np.linalg.norm(v_prime)

        # Sampling grid
        s = np.linspace(-0.5, 0.5, w)
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
        complex_points = np.zeros((resolution[0], resolution[1], n_components // 2), dtype=complex)
        for i in range(n_components // 2):
            real_part = points[..., i*2]
            imaginary_part = points[..., i*2+1]
            complex_points[..., i] = real_part + 1j * imaginary_part
        return complex_points

    @staticmethod
    @njit
    def compute_fractal(c_z_e_array, max_iterations=100, escape_radius=2):
        """
        Compute a fractal based on the formula z = z^e + c.
        """
        height, width, _ = c_z_e_array.shape
        escape_counts = np.zeros((height, width), dtype=np.int32)

        for i in range(height):
            for j in range(width):
                c, z, e = c_z_e_array[i, j]
                count = 0
                while abs(z) <= escape_radius and count < max_iterations:
                    if not z:
                        z = c
                    z = z**e + c
                    count += 1
                escape_counts[i, j] = count

        return escape_counts

    def render_fractal(self):
        """Generate fractal data for the current settings."""
        t1 = time()

        sampled_points = self.sample_plane()
        max_iterations = self.settings.start_iterations + self.settings.iteration_growth * np.log(1 / self.settings.scale)
        fractal = self.compute_fractal(sampled_points, max_iterations=max_iterations, escape_radius=self.settings.escape_radius)
        t2 = time()
        print(f"{t2-t1:.2f} sec, {max_iterations:.0f} its,  {(t2-t1)/max_iterations:.2f} it/sec")
        return fractal

    def update_view(self, press=None, release=None):
        """Update fractal view based on rectangle selection or re-render."""
        if press and release:
            # Extract rectangle bounds
            x0, y0 = press.xdata, press.ydata
            x1, y1 = release.xdata, release.ydata

            width_pixels = x1 - x0
            height_pixels = y1 - y0
            if width_pixels < self.MIN_RECTANGLE[0] or height_pixels < self.MIN_RECTANGLE[1]:
                print("Selection too small!")
                return

            # Convert to normalized coordinates
            cx = (x0 + x1) / 2 / self.settings.resolution[0] - 0.5
            cy = (y0 + y1) / 2 / self.settings.resolution[1] - 0.5
            self.settings.center = (
                self.settings.center[0] + cx * self.settings.scale,
                self.settings.center[1] + cy * self.settings.scale
            )
            self.settings.scale *= min(abs(x1 - x0) / self.settings.resolution[0], abs(y1 - y0) / self.settings.resolution[1])

        fractal = self.render_fractal()

        self.history.append(deepcopy(self.settings))

        self.image.set_data(fractal)
        self.image.set_clim(vmin=fractal.min(), vmax=fractal.max())
        self.colorbar.update_normal(self.image)
        self.fig.canvas.draw()

    def reset_view(self):
        """Reset fractal view to initial settings."""
        self.settings = self.history[0]
        self.history = self.history[:1]

    def go_back(self):
        """Go back in history."""
        if len(self.history) <= 1:
            print("Can't go back")
            return

        self.settings = self.history[-2]
        self.history = self.history[:-2]

    def on_key(self, event):
        """Handle key press events."""
        step = self.settings.scale * 0.1
        if event.key == 'w':
            self.settings.center = (self.settings.center[0], self.settings.center[1] - step)
            self.update_view()
        elif event.key == 's':
            self.settings.center = (self.settings.center[0], self.settings.center[1] + step)
            self.update_view()
        elif event.key == 'a':
            self.settings.center = (self.settings.center[0] - step, self.settings.center[1])
            self.update_view()
        elif event.key == 'd':
            self.settings.center = (self.settings.center[0] + step, self.settings.center[1])
            self.update_view()
        elif event.key == 'q':
            self.settings.scale *= 1.1
            self.update_view()
        elif event.key == 'e':
            self.settings.scale /= 1.1
            self.update_view()
        elif event.key == 'f':
            self.settings.rotation += np.pi / 16
            self.update_view()
        elif event.key == 'r':
            self.settings.rotation -= np.pi / 16
            self.update_view()
        elif event.key == 'home':
            self.reset_view()
            self.update_view()
        elif event.key == 'backspace':
            self.go_back()
            self.update_view()

    def draw(self):
        """Set up the Matplotlib plot and event handlers."""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        fractal = self.render_fractal()
        self.image = self.ax.imshow(fractal, cmap="inferno")
        self.colorbar = plt.colorbar(self.image, ax=self.ax, label="Iterations")

        self.rect_selector = RectangleSelector(
            self.ax,
            self.update_view,
            interactive=True,
            button=[1]  # Left mouse button
        )
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()


mandelbrot_settings = FractalSettings(
    name="Mandelbrot",
    u = [1, 0, 0, 0, 2, 0],
    v = [0, 1, 0, 0, 2, 0],
    o = [0, 0, 0, 0, 2, 0],
    center=(-0.4, 0),
    rotation=0,
    scale=4.0,
    resolution=(500, 500),
    start_iterations=100,
    iteration_growth=20,
    escape_radius=2.0
)

julia_settings = FractalSettings(
    name="Julia",
    u = [0.45, 0.1428, 1, 0, 2, 0],
    v = [0.45, 0.1428, 0, 1, 2, 0],
    o = [0.45, 0.1428, 0, 0, 2, 0],
    center=(-0.4, 0),
    rotation=0,
    scale=4.0,
    resolution=(500, 500),
    start_iterations=100,
    iteration_growth=20,
    escape_radius=2.0
)

expulia_settings = FractalSettings(
    name="Expulia",
    u=[0.45, 0.1428, 0, 0, 2, 0],
    v=[0.45, 0.1428, 0, 0, 0, 0],
    o=[0.45, 0.1428, 0, 0, 0, 2],
    center=(-0.4, 0),
    rotation=0,
    scale=16.0,
    resolution=(500, 500),
    start_iterations=100,
    iteration_growth=20,
    escape_radius=2.0
)

renderer = FractalRenderer(expulia_settings)
renderer.draw()

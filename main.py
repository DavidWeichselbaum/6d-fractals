from time import time
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from numba import njit, prange
from copy import deepcopy
from pprint import pprint


# RESOLUTION = (1000, 800)
RESOLUTION = (500, 500)
START_ITERATIONS = 100
ITERATION_GROWTH = 20
ESCAPE_RADIUS = 2.0


@dataclass
class FractalSettings:
    u: list  # vector 1
    o: list  # origin
    v: list  # point 2
    center: tuple
    rotation: float
    scale: float
    escape_counts: np.ndarray | None = None


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
        u, o, v, center, rotation, scale = (
            self.settings.u,
            self.settings.o,
            self.settings.v,
            self.settings.center,
            self.settings.rotation,
            self.settings.scale,
        )

        w, h = RESOLUTION
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

    @staticmethod
    @njit(parallel=True)
    def compute_fractal(c_z_e_array, max_iterations=100, escape_radius=2):
        """
        Compute a fractal based on the formula z = z^e + c.
        """
        height, width, _ = c_z_e_array.shape
        escape_counts = np.zeros((height, width), dtype=np.int32)

        for i in prange(height):  # parallelized
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
        max_iterations = START_ITERATIONS + ITERATION_GROWTH * np.log(1 / self.settings.scale)
        escape_counts = self.compute_fractal(sampled_points, max_iterations=max_iterations, escape_radius=ESCAPE_RADIUS)
        t2 = time()
        print(f"{t2-t1:.2f} sec, {max_iterations:.0f} its,  {(t2-t1)/max_iterations:.2f} it/sec")
        return escape_counts

    def update_view(self, press=None, release=None):
        """Update fractal view based on rectangle selection or re-render."""
        if press and release:
            self.handle_selection(press, release)

        escape_counts = self.settings.escape_counts
        if escape_counts is None:
            escape_counts = self.render_fractal()
        else:
            self.settings.escape_counts = None  # make historic settings changeable again

        self.image.set_data(escape_counts)
        self.image.set_clim(vmin=escape_counts.min(), vmax=escape_counts.max())
        self.colorbar.update_normal(self.image)
        self.fig.canvas.draw()

        history_settings = deepcopy(self.settings)
        history_settings.escape_counts = escape_counts
        self.history.append(history_settings)

    def handle_selection(self, press, release):
        # Extract rectangle bounds
        x0, y0 = press.xdata, press.ydata
        x1, y1 = release.xdata, release.ydata

        width_pixels = x1 - x0
        height_pixels = y1 - y0
        if width_pixels < self.MIN_RECTANGLE[0] or height_pixels < self.MIN_RECTANGLE[1]:
            print("Selection too small!")
            return

        # Convert to normalized coordinates
        cx = (x0 + x1) / 2 / RESOLUTION[0] - 0.5
        cy = (y0 + y1) / 2 / RESOLUTION[1] - 0.5
        self.settings.center = (
            self.settings.center[0] + cx * self.settings.scale,
            self.settings.center[1] + cy * self.settings.scale
        )
        self.settings.scale *= min(abs(x1 - x0) / RESOLUTION[0], abs(y1 - y0) / RESOLUTION[1])

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

    def randomize_settings(self):
        self.settings = FractalSettings(
            u=np.random.random(6) * 4 -2,
            o=np.random.random(6) * 4 -2,
            v=np.random.random(6) * 4 -2,
            center=(-0.4, 0),
            rotation=0,
            scale=16.0,
        )
        pprint(self.settings)

    def perturb_settings(self):
        u_p = np.random.random(6) * 0.2 -0.1
        o_p = np.random.random(6) * 0.2 -0.1
        v_p = np.random.random(6) * 0.2 -0.1
        self.settings.u += u_p
        self.settings.o += o_p
        self.settings.v += v_p
        pprint(self.settings)

    def on_key(self, event):
        """Handle key press events."""
        if event.key == 'escape':
            exit()
        elif event.key in 'wasd':
            step = self.settings.scale * 0.1
            if event.key == 'w':
                self.settings.center = (self.settings.center[0], self.settings.center[1] - step)
            elif event.key == 's':
                self.settings.center = (self.settings.center[0], self.settings.center[1] + step)
            elif event.key == 'a':
                self.settings.center = (self.settings.center[0] - step, self.settings.center[1])
            elif event.key == 'd':
                self.settings.center = (self.settings.center[0] + step, self.settings.center[1])
        elif event.key == 'q':
            self.settings.scale *= 1.1
        elif event.key == 'e':
            self.settings.scale /= 1.1
        elif event.key == 'f':
            self.settings.rotation += np.pi / 16
        elif event.key == 'r':
            self.settings.rotation -= np.pi / 16
        elif event.key == 'home':
            self.reset_view()
        elif event.key == 'backspace':
            self.go_back()
        if event.key == 'x':
            self.randomize_settings()
        if event.key == 'z':
            self.perturb_settings()

        self.update_view()

    def draw(self):
        """Set up the Matplotlib plot and event handlers."""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        escape_counts = self.render_fractal()
        self.image = self.ax.imshow(escape_counts, cmap="inferno")
        self.colorbar = plt.colorbar(self.image, ax=self.ax, label="Iterations")
        self.ax.axis('off')
        plt.tight_layout()

        self.rect_selector = RectangleSelector(
            self.ax,
            self.update_view,
            interactive=True,
            button=[1]  # Left mouse button
        )

        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()


mandelbrot_settings = FractalSettings(
    u = np.array([1, 0, 0, 0, 2, 0], dtype=np.float64),
    o = np.array([0, 0, 0, 0, 2, 0], dtype=np.float64),
    v = np.array([0, 1, 0, 0, 2, 0], dtype=np.float64),
    center=(-0.4, 0),
    rotation=0,
    scale=4.0,
)

julia_settings = FractalSettings(
    u = np.array([0.45, 0.1428, 1, 0, 2, 0]),
    o = np.array([0.45, 0.1428, 0, 0, 2, 0]),
    v = np.array([0.45, 0.1428, 0, 1, 2, 0]),
    center=(-0.4, 0),
    rotation=0,
    scale=4.0,
)

expulia_settings = FractalSettings(
    u=np.array([0.45, 0.1428, 0, 0, 2, 0]),
    o=np.array([0.45, 0.1428, 0, 0, 0, 0]),
    v=np.array([0.45, 0.1428, 0, 0, 0, 2]),
    center=(-0.4, 0),
    rotation=0,
    scale=16.0,
)

xmas_fractal = FractalSettings(
    u=np.array([0.45, 0.1428, 0, 0, 0.70710678, 1.29289322]),
    o=np.array([0.45, 0.1428, 0, 0, 0, 2]),
    v=np.array([0.45, 0.1428, 0, 0, -0.70710678, 1.29289322]),
    center=(-0.4, 0),
    rotation=0,
    scale=16.0,
)

renderer = FractalRenderer(julia_settings)
renderer.draw()

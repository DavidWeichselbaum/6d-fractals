import sys
from copy import deepcopy

import numpy as np
import matplotlib.cm as cm
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QHBoxLayout, QGraphicsView, QGraphicsScene
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPen

from main import sample_plane, compute_fractal, FractalSettings


# RESOLUTION = (1000, 800)
RESOLUTION = (500, 500)
START_ITERATIONS = 100
ITERATION_GROWTH = 20
ESCAPE_RADIUS = 2.0



class FractalWorker(QThread):
    finished = pyqtSignal(np.ndarray)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings

    def run(self):
        """Perform fractal computation in a separate thread."""
        sampled_points = sample_plane(
            self.settings.u,
            self.settings.o,
            self.settings.v,
            self.settings.center,
            self.settings.rotation,
            self.settings.scale
        )
        max_iterations = START_ITERATIONS + ITERATION_GROWTH * np.log(1 / self.settings.scale)
        escape_counts = compute_fractal(
            sampled_points,
            max_iterations=int(max_iterations),
            escape_radius=ESCAPE_RADIUS
        )
        self.finished.emit(escape_counts)


class FractalApp(QMainWindow):
    MIN_RECTANGLE = (5, 5)  # Minimum rectangle size in pixels

    def __init__(self, initial_settings):
        super().__init__()
        self.settings = deepcopy(initial_settings)
        self.history = [deepcopy(initial_settings)]
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Fractal Explorer with Rectangle Selection")
        self.setGeometry(100, 100, 800, 600)

        # Main layout
        main_layout = QVBoxLayout()

        # Graphics view for fractal image
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setRenderHints(self.graphics_view.renderHints() | Qt.SmoothTransformation)
        main_layout.addWidget(self.graphics_view)

        # Controls layout
        controls = QHBoxLayout()

        # Zoom buttons
        zoom_in_btn = QPushButton("Zoom In")
        zoom_in_btn.clicked.connect(self.zoom_in)
        controls.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton("Zoom Out")
        zoom_out_btn.clicked.connect(self.zoom_out)
        controls.addWidget(zoom_out_btn)

        # Reset button
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_view)
        controls.addWidget(reset_btn)

        # Add controls to the main layout
        main_layout.addLayout(controls)

        # Main widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Variables for rectangle selection
        self.start_pos = None
        self.selection_rect = None

        # Enable mouse events for rectangle selection
        self.graphics_view.setMouseTracking(True)
        self.graphics_view.viewport().installEventFilter(self)

        # Initial render
        self.render_fractal()

    def render_fractal(self):
        """Start fractal rendering in a separate thread."""
        self.worker = FractalWorker(self.settings)
        self.worker.finished.connect(self.display_fractal)
        self.worker.start()

    def display_fractal(self, escape_counts):
        """Convert fractal data to an image with colormap and display it."""
        # Scale the escape counts to the range [0, 1] based on min and max values
        min_val = escape_counts.min()
        max_val = escape_counts.max()
        normalized = (escape_counts - min_val) / (max_val - min_val)

        # Apply colormap
        from matplotlib import colormaps
        colormap = colormaps["inferno"]
        colored = (colormap(normalized)[:, :, :3] * 255).astype(np.uint8)  # Convert RGBA to RGB

        # Convert to QImage
        height, width, _ = colored.shape
        q_image = QImage(colored.data, width, height, 3 * width, QImage.Format_RGB888)

        # Convert QImage to QPixmap and display
        pixmap = QPixmap.fromImage(q_image)
        self.graphics_scene.clear()
        self.graphics_scene.addPixmap(pixmap)

    def zoom_in(self):
        """Zoom in on the fractal."""
        self.settings.scale /= 1.5
        self.render_fractal()

    def zoom_out(self):
        """Zoom out on the fractal."""
        self.settings.scale *= 1.5
        self.render_fractal()

    def reset_view(self):
        """Reset to the initial fractal view."""
        self.settings = deepcopy(self.history[0])
        self.render_fractal()

    def eventFilter(self, source, event):
        """Handle mouse events for rectangle selection."""
        if event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            return True
        elif event.type() == event.MouseMove and self.start_pos:
            end_pos = event.pos()
            self.draw_selection_rectangle(self.start_pos, end_pos)
            return True
        elif event.type() == event.MouseButtonRelease and event.button() == Qt.LeftButton:
            if self.start_pos:
                self.handle_selection(self.start_pos, event.pos())
                self.start_pos = None
            return True
        return super().eventFilter(source, event)

    def draw_selection_rectangle(self, start_pos, end_pos):
        """Draw the selection rectangle with aspect ratio constraint."""
        if self.selection_rect:
            self.graphics_scene.removeItem(self.selection_rect)

        # Map start and end positions to scene coordinates
        start_scene = self.graphics_view.mapToScene(start_pos)
        end_scene = self.graphics_view.mapToScene(end_pos)

        # Calculate the width and height with aspect ratio constraint
        dx = abs(end_scene.x() - start_scene.x())
        dy = abs(end_scene.y() - start_scene.y())
        aspect_ratio = RESOLUTION[0] / RESOLUTION[1]

        if dx / RESOLUTION[0] > dy / RESOLUTION[1]:
            dy = dx / aspect_ratio
        else:
            dx = dy * aspect_ratio

        # Adjust for dragging direction
        if end_scene.x() < start_scene.x():
            dx = -dx
        if end_scene.y() < start_scene.y():
            dy = -dy

        # Compute the constrained end point
        constrained_end_scene = QPointF(start_scene.x() + dx, start_scene.y() + dy)

        # Create a rectangle with the constrained dimensions
        rect = QRectF(start_scene, constrained_end_scene)
        self.selection_rect = self.graphics_scene.addRect(rect, QPen(Qt.red, 2))

        # Store the constrained end position for use in handle_selection
        self.constrained_end_pos = self.graphics_view.mapFromScene(constrained_end_scene)

    def handle_selection(self, press_pos, release_pos):
        """Handle rectangle selection to adjust the fractal view."""
        if self.selection_rect:
            self.graphics_scene.removeItem(self.selection_rect)
            self.selection_rect = None

        # Use the constrained end position
        press_scene = self.graphics_view.mapToScene(press_pos)
        release_scene = self.graphics_view.mapToScene(self.constrained_end_pos)

        x0, y0 = press_scene.x(), press_scene.y()
        x1, y1 = release_scene.x(), release_scene.y()

        # Calculate rectangle dimensions
        width_pixels = abs(x1 - x0)
        height_pixels = abs(y1 - y0)

        if width_pixels < self.MIN_RECTANGLE[0] or height_pixels < self.MIN_RECTANGLE[1]:
            print("Selection too small!")
            return

        # Normalize coordinates to fractal space
        cx = (x0 + x1) / 2 / RESOLUTION[0] - 0.5
        cy = (y0 + y1) / 2 / RESOLUTION[1] - 0.5

        self.settings.center = (
            self.settings.center[0] + cx * self.settings.scale,
            self.settings.center[1] + cy * self.settings.scale
        )
        self.settings.scale *= min(width_pixels / RESOLUTION[0], height_pixels / RESOLUTION[1])

        self.render_fractal()



if __name__ == "__main__":
    mandelbrot_settings = FractalSettings(
        u=np.array([1, 0, 0, 0, 2, 0], dtype=np.float64),
        o=np.array([0, 0, 0, 0, 2, 0], dtype=np.float64),
        v=np.array([0, 1, 0, 0, 2, 0], dtype=np.float64),
        center=(-0.4, 0),
        rotation=0,
        scale=4.0,
    )

    app = QApplication(sys.argv)
    main_window = FractalApp(mandelbrot_settings)
    main_window.show()
    sys.exit(app.exec())

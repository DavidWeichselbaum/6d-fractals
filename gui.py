import sys
import logging
from copy import deepcopy

import numpy as np
import matplotlib.cm as cm
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QGraphicsView, QGraphicsScene, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPen

from main import sample_plane, compute_fractal, FractalSettings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RESOLUTION = (1000, 800)
# RESOLUTION = (1920, 1080)
ASPECT_RATIO = RESOLUTION[0] / RESOLUTION[1]
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
        logging.info("Starting fractal computation...")
        sampled_points = sample_plane(
            self.settings.u,
            self.settings.o,
            self.settings.v,
            self.settings.center,
            self.settings.rotation,
            self.settings.scale,
            RESOLUTION,
        )
        max_iterations = START_ITERATIONS + ITERATION_GROWTH * np.log(1 / self.settings.scale)
        escape_counts = compute_fractal(
            sampled_points,
            max_iterations=int(max_iterations),
            escape_radius=ESCAPE_RADIUS
        )
        logging.info("Fractal computation completed.")
        self.finished.emit(escape_counts)


class FractalApp(QMainWindow):
    MIN_RECTANGLE = (5, 5)  # Minimum rectangle size in pixels

    def __init__(self, initial_settings):
        super().__init__()
        self.settings = deepcopy(initial_settings)
        self.history = [deepcopy(initial_settings)]
        self.colormap = cm.get_cmap("inferno")

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Fractal Explorer with Rectangle Selection")
        self.setGeometry(100, 100, 1000, 600)

        # Main layout: Horizontal layout with fractal display and controls
        main_layout = QHBoxLayout()

        # Add the fractal display
        fractal_layout = QVBoxLayout()
        self.setup_fractal_display(fractal_layout)
        main_layout.addLayout(fractal_layout)

        # Add the control buttons
        controls_layout = self.setup_controls()
        main_layout.addLayout(controls_layout)

        # Main widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Variables for rectangle selection
        self.start_pos = None
        self.constrained_end_pos = None
        self.selection_rect = None

        # Enable mouse events for rectangle selection
        self.graphics_view.setMouseTracking(True)
        self.graphics_view.viewport().installEventFilter(self)

        # Connect key press events
        self.graphics_view.setFocus()
        self.graphics_view.keyPressEvent = self.on_key

        # Initial render
        self.render_fractal()

    def setup_fractal_display(self, layout):
        """Set up the fractal display area."""
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setRenderHints(self.graphics_view.renderHints() | Qt.SmoothTransformation)
        layout.addWidget(self.graphics_view)

    def setup_controls(self):
        """Set up the control buttons."""
        controls_layout = QVBoxLayout()

        # Add zoom controls
        controls_layout.addLayout(self.setup_zoom_controls())

        # Add move controls
        controls_layout.addLayout(self.setup_move_controls())

        # Add rotate controls
        controls_layout.addLayout(self.setup_rotate_controls())

        # Add reset and history controls
        controls_layout.addWidget(self.create_button("Reset", "Shortcut: Home", self.reset_view))
        controls_layout.addWidget(self.create_button("⟲", "Go Back (Shortcut: Backspace)", self.go_back))

        # Add randomize and perturb controls
        controls_layout.addWidget(self.create_button("Randomize", "Randomize Settings (Shortcut: X)", self.randomize_settings))
        controls_layout.addWidget(self.create_button("Perturb", "Perturb Settings (Shortcut: Z)", self.perturb_settings))

        # Add a spacer to center the controls vertically
        controls_layout.addStretch()

        return controls_layout

    def setup_zoom_controls(self):
        """Set up zoom in and zoom out controls."""
        zoom_layout = QHBoxLayout()
        zoom_out_btn = self.create_button("-", "Zoom Out (Shortcut: E)", self.zoom_out)
        zoom_in_btn = self.create_button("+", "Zoom In (Shortcut: Q)", self.zoom_in)
        zoom_layout.addWidget(zoom_out_btn)
        zoom_layout.addWidget(zoom_in_btn)
        return zoom_layout

    def setup_move_controls(self):
        """Set up move controls in an arrow key layout."""
        move_layout = QGridLayout()
        move_layout.addWidget(self.create_button("↑", "Move Up (Shortcut: W)", lambda: self.move("W")), 0, 1)
        move_layout.addWidget(self.create_button("←", "Move Left (Shortcut: A)", lambda: self.move("A")), 1, 0)
        move_layout.addWidget(self.create_button("↓", "Move Down (Shortcut: S)", lambda: self.move("S")), 1, 1)
        move_layout.addWidget(self.create_button("→", "Move Right (Shortcut: D)", lambda: self.move("D")), 1, 2)
        return move_layout

    def setup_rotate_controls(self):
        """Set up rotate clockwise and counterclockwise controls."""
        rotate_layout = QHBoxLayout()
        rotate_ccw_btn = self.create_button("↺", "Rotate Counterclockwise (Shortcut: R)", lambda: self.rotate("CCW"))
        rotate_cw_btn = self.create_button("↻", "Rotate Clockwise (Shortcut: F)", lambda: self.rotate("CW"))
        rotate_layout.addWidget(rotate_ccw_btn)
        rotate_layout.addWidget(rotate_cw_btn)
        return rotate_layout

    def create_button(self, label, tooltip, callback):
        """Create a reusable button."""
        button = QPushButton(label)
        button.setToolTip(tooltip)
        button.clicked.connect(callback)
        return button

    def render_fractal(self):
        """Start fractal rendering in a separate thread."""
        if self.settings.escape_counts is not None:
            logging.info("Using previously rendered fractal...")
            self.display_fractal(self.settings.escape_counts)
        else:
            logging.info("Rendering fractal...")
            self.worker = FractalWorker(self.settings)
            self.worker.finished.connect(self.display_fractal)
            self.worker.start()

    def display_fractal(self, escape_counts):
        """Convert fractal data to an image with colormap and display it."""
        logging.info("Displaying fractal...")
        min_val = escape_counts.min()
        max_val = escape_counts.max()
        normalized = (escape_counts - min_val) / (max_val - min_val)

        colored = (self.colormap(normalized)[:, :, :3] * 255).astype(np.uint8)

        height, width, _ = colored.shape
        q_image = QImage(colored.data, width, height, 3 * width, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        self.graphics_scene.clear()
        self.graphics_scene.addPixmap(pixmap)

        history_settings = deepcopy(self.settings)
        history_settings.escape_counts = escape_counts
        self.history.append(history_settings)

        logging.info("Fractal display updated.")

    def zoom_in(self):
        logging.info("Zooming in...")
        self.settings.scale /= 1.5
        self.render_fractal()

    def zoom_out(self):
        logging.info("Zooming out...")
        self.settings.scale *= 1.5
        self.render_fractal()

    def reset_view(self):
        logging.info("Resetting view...")
        self.settings = self.history[0]
        self.history = self.history[:1]
        self.render_fractal()

    def go_back(self):
        logging.info("Going back in history...")
        if len(self.history) <= 1:
            logging.info("No history left.")
            return

        self.settings = self.history[-2]
        self.history = self.history[:-2]
        self.render_fractal()

    def randomize_settings(self):
        logging.info("Randomizing fractal settings...")
        self.settings = FractalSettings(
            u=np.random.normal(size=6),
            o=np.random.normal(size=6),
            v=np.random.normal(size=6),
            center=(0.0, 0.0),
            rotation=0.0,
            scale=4.0,
        )
        self.render_fractal()

    def perturb_settings(self):
        logging.info("Perturbing fractal settings...")
        self.settings.u += np.random.normal(scale=0.05, size=self.settings.u.shape)
        self.settings.o += np.random.normal(scale=0.05, size=self.settings.o.shape)
        self.settings.v += np.random.normal(scale=0.05, size=self.settings.v.shape)
        self.render_fractal()

    def move(self, direction):
        logging.info(f"Moving {direction}...")
        step = self.settings.scale * 0.1
        if direction == "W":
            self.settings.center = (self.settings.center[0], self.settings.center[1] - step)
        elif direction == "S":
            self.settings.center = (self.settings.center[0], self.settings.center[1] + step)
        elif direction == "A":
            self.settings.center = (self.settings.center[0] - step, self.settings.center[1])
        elif direction == "D":
            self.settings.center = (self.settings.center[0] + step, self.settings.center[1])
        self.render_fractal()

    def rotate(self, direction):
        logging.info(f"Rotating {direction}...")
        if direction == "CW":
            self.settings.rotation += np.pi / 8
        elif direction == "CCW":
            self.settings.rotation -= np.pi / 8
        self.render_fractal()

    def on_key(self, event):
        """Handle key press events."""
        logging.info(f"Key pressed: {event.key()}")
        step = self.settings.scale * 0.1
        if event.key() == Qt.Key_Escape:
            exit()
        elif event.key() == Qt.Key_W:
            self.move("W")
        elif event.key() == Qt.Key_S:
            self.move("S")
        elif event.key() == Qt.Key_A:
            self.move("A")
        elif event.key() == Qt.Key_D:
            self.move("D")
        elif event.key() == Qt.Key_Q:
            self.zoom_out()
        elif event.key() == Qt.Key_E:
            self.zoom_in()
        elif event.key() == Qt.Key_F:
            self.rotate("CW")
        elif event.key() == Qt.Key_R:
            self.rotate("CCW")
        elif event.key() == Qt.Key_Home:
            self.reset_view()
        elif event.key() == Qt.Key_Backspace:
            self.go_back()
        elif event.key() == Qt.Key_X:
            self.randomize_settings()
        elif event.key() == Qt.Key_Z:
            self.perturb_settings()

    def eventFilter(self, source, event):
        """Handle mouse events for rectangle selection."""
        if event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            return True
        elif event.type() == event.MouseMove and self.start_pos:
            end_pos = event.pos()
            self.constrained_end_pos = self.draw_selection_rectangle(self.start_pos, end_pos)
            return True
        elif event.type() == event.MouseButtonRelease and event.button() == Qt.LeftButton:
            if self.start_pos and self.constrained_end_pos:
                self.handle_selection(self.start_pos, self.constrained_end_pos)
                self.start_pos = None
                self.constrained_end_pos = None
            return True
        return super().eventFilter(source, event)

    def draw_selection_rectangle(self, start_pos, end_pos):
        """Draw the selection rectangle with aspect ratio constraint and boundary checks."""
        if self.selection_rect:
            self.graphics_scene.removeItem(self.selection_rect)

        start_scene = self.graphics_view.mapToScene(start_pos)
        end_scene = self.graphics_view.mapToScene(end_pos)

        dx = abs(end_scene.x() - start_scene.x())
        dy = abs(end_scene.y() - start_scene.y())

        if dx / RESOLUTION[0] > dy / RESOLUTION[1]:
            dy = dx / ASPECT_RATIO
        else:
            dx = dy * ASPECT_RATIO

        if end_scene.x() < start_scene.x():
            dx = -dx
        if end_scene.y() < start_scene.y():
            dy = -dy

        constrained_end_scene = QPointF(start_scene.x() + dx, start_scene.y() + dy)
        scene_rect = self.graphics_scene.sceneRect()
        if not scene_rect.contains(start_scene) or not scene_rect.contains(constrained_end_scene):
            return

        rect = QRectF(start_scene, constrained_end_scene)
        self.selection_rect = self.graphics_scene.addRect(rect, QPen(Qt.red, 2))
        constrained_end_pos = self.graphics_view.mapFromScene(constrained_end_scene)
        return constrained_end_pos

    def handle_selection(self, press_pos, release_pos):
        """Handle rectangle selection to adjust the fractal view."""
        if self.selection_rect:
            self.graphics_scene.removeItem(self.selection_rect)
            self.selection_rect = None

        press_scene = self.graphics_view.mapToScene(press_pos)
        release_scene = self.graphics_view.mapToScene(release_pos)

        x0, y0 = press_scene.x(), press_scene.y()
        x1, y1 = release_scene.x(), release_scene.y()

        width_pixels = abs(x1 - x0)
        height_pixels = abs(y1 - y0)

        if width_pixels < self.MIN_RECTANGLE[0] or height_pixels < self.MIN_RECTANGLE[1]:
            logging.info("Selection too small!")
            return

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

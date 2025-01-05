import sys
import logging
from copy import deepcopy

import yaml
import numpy as np
from matplotlib import colormaps
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QGraphicsView, QGraphicsScene, QGridLayout, QFileDialog, QLineEdit, QLabel
)

from PyQt5.QtWidgets import QComboBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor, QBrush

from utils.parameters import sample_plane
from utils.fractal import compute_fractal
from utils.datatypes import FractalSettings


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

        self.init_ui()
        self.update_colormap("inferno")

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
        self.selection_rect = None
        self.selection_rect_visual = None

        # Enable mouse events for rectangle selection
        self.graphics_view.setMouseTracking(True)
        self.graphics_view.viewport().installEventFilter(self)

        # Connect key press events
        self.graphics_view.setFocus()
        self.graphics_view.keyPressEvent = self.on_key

    def setup_fractal_display(self, layout):
        """Set up the fractal display area."""
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setRenderHints(self.graphics_view.renderHints() | Qt.SmoothTransformation)
        layout.addWidget(self.graphics_view)

    def setup_controls(self):
        """Set up the control buttons and input fields."""
        controls_layout = QVBoxLayout()

        # Add save and load controls
        controls_layout.addWidget(self.create_button("Save Settings", "Save current settings to a file", self.save_settings))
        controls_layout.addWidget(self.create_button("Load Settings", "Load settings from a file", self.load_settings))

        # Add reset and history controls
        controls_layout.addWidget(self.create_button("Reset", "Shortcut: Home", self.reset_view))
        controls_layout.addWidget(self.create_button("⟲", "Go Back (Shortcut: Backspace)", self.go_back))

        # Add randomize and perturb controls
        controls_layout.addWidget(self.create_button("Randomize", "Randomize Settings (Shortcut: X)", self.randomize_settings))
        controls_layout.addWidget(self.create_button("Perturb", "Perturb Settings (Shortcut: Z)", self.perturb_settings))

        # Add colormap dropdown
        controls_layout.addWidget(self.setup_colormap_dropdown())

        # Add zoom controls
        controls_layout.addLayout(self.setup_zoom_controls())

        # Add move controls
        controls_layout.addLayout(self.setup_move_controls())

        # Add rotate controls
        controls_layout.addLayout(self.setup_rotate_controls())

        # Add fields for u, o, v vectors
        controls_layout.addWidget(QLabel("Edit Fractal Parameters"))
        controls_layout.addLayout(self.setup_uov_inputs())

        # Add a spacer to center the controls vertically
        controls_layout.addStretch()

        return controls_layout

    def setup_uov_inputs(self):
        """Create compact input fields for u, o, v vectors with labels."""
        uov_layout = QGridLayout()
        self.u_fields = []
        self.o_fields = []
        self.v_fields = []
        input_width = 60

        # Labels for the first column
        row_labels = [
            "c a",
            "c b",
            "z_0 a",
            "z_0 b",
            "power a",
            "power b",
        ]
        for row, label in enumerate(row_labels):
            row_label = QLabel(label)
            row_label.setFixedWidth(input_width)  # Set a fixed width for the input field
            uov_layout.addWidget(row_label, row + 1, 0)  # First column for row labels

        # Add column headers and input fields
        column_labels = ["Top", "Center", "Right"]
        fields = [self.u_fields, self.o_fields, self.v_fields]
        values = [self.settings.u, self.settings.o, self.settings.v]
        for col, (header, field_list, value) in enumerate(zip(column_labels, fields, values)):
            uov_layout.addWidget(QLabel(header), 0, col + 1)  # Column headers (shifted by 1)
            for row in range(6):  # 6 values in each vector
                line_edit = QLineEdit(str(value[row]))
                line_edit.setToolTip(f"{header} Component {row_labels[row]}")
                line_edit.setFixedWidth(input_width)  # Set a fixed width for the input field
                line_edit.returnPressed.connect(self.update_uov)
                uov_layout.addWidget(line_edit, row + 1, col + 1)
                field_list.append(line_edit)

        return uov_layout

    def update_uov(self):
        """Update u, o, v vectors from input fields and re-render."""
        try:
            logging.info(f"Updated u, o, v vectors from u={self.settings.u}, o={self.settings.o}, v={self.settings.v}")
            logging.warning(self.u_fields == self.o_fields)
            # Update settings.u, settings.o, settings.v based on input
            self.settings.u = np.array([float(field.text()) for field in self.u_fields])
            self.settings.o = np.array([float(field.text()) for field in self.o_fields])
            self.settings.v = np.array([float(field.text()) for field in self.v_fields])
            logging.info(f"Updated u, o, v vectors: u={self.settings.u}, o={self.settings.o}, v={self.settings.v}")

            # Re-render fractal
            self.render_fractal()
        except ValueError:
            logging.warning("Invalid input in one or more fields. Please enter numeric values.")

    def update_uov_inputs(self):
        """Update the input fields to reflect the current u, o, v settings."""
        fields = [self.u_fields, self.o_fields, self.v_fields]
        values = [self.settings.u, self.settings.o, self.settings.v]
        for field_list, value in zip(fields, values):
            for i in range(6):
                field_list[i].setText(str(value[i]))
                field_list[i].setCursorPosition(0)

#         for point_field, point in (self.u_fields, self.o_fields, self.v_fields):
#             for i in range(6):
#                 point_field[i].setText(str(self.settings.u[i]))
#                 point_field[i].setCursorPosition(0)

    def setup_colormap_dropdown(self):
        """Set up a dropdown menu for selecting colormaps."""
        colormap_dropdown = QComboBox()
        colormap_dropdown.setToolTip("Select a colormap for the fractal")
        colormap_dropdown.addItems(sorted(colormaps.keys()))  # Add all available colormaps
        colormap_dropdown.setCurrentText("inferno")  # Default colormap
        colormap_dropdown.currentTextChanged.connect(self.update_colormap)
        return colormap_dropdown

    def update_colormap(self, colormap_name, render = True):
        """Update the colormap and re-render the fractal."""
        logging.info(f"Changing colormap to: {colormap_name}")
        self.colormap = colormaps.get_cmap(colormap_name)
        self.update_interface_color()
        self.render_fractal()

    def update_interface_color(self):
        """Update the interface colors to match the colormap."""
        # Extract colors from the colormap
        background_color = self.colormap(0)  # RGBA tuple for background
        text_color = self.colormap(0.3)  # RGBA tuple for text
        border_color = self.colormap(0.6)  # RGBA tuple for borders
        input_bg_color = self.colormap(0.1)  # Slightly lighter for input fields

        # Convert colors to RGB values
        r_bg, g_bg, b_bg, _ = [int(c * 255) for c in background_color]
        r_text, g_text, b_text, _ = [int(c * 255) for c in text_color]
        r_border, g_border, b_border, _ = [int(c * 255) for c in border_color]
        r_input_bg, g_input_bg, b_input_bg, _ = [int(c * 255) for c in input_bg_color]

        # Update stylesheet for the entire app
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: rgb({r_bg}, {g_bg}, {b_bg});
            }}
            QLabel, QPushButton, QComboBox {{
                color: rgb({r_text}, {g_text}, {b_text});
                background-color: rgb({r_bg}, {g_bg}, {b_bg});
                border: 1px solid rgb({r_border}, {g_border}, {b_border});
            }}
            QPushButton:hover {{
                background-color: rgb({r_border}, {g_border}, {b_border});
                color: rgb({r_bg}, {g_bg}, {b_bg});
            }}
            QComboBox {{
                color: rgb({r_text}, {g_text}, {b_text});
                background-color: rgb({r_bg}, {g_bg}, {b_bg});
                border: 1px solid rgb({r_border}, {g_border}, {b_border});
            }}
            QComboBox QAbstractItemView {{
                background-color: rgb({r_bg}, {g_bg}, {b_bg});
                color: rgb({r_text}, {g_text}, {b_text});
                border: 1px solid rgb({r_border}, {g_border}, {b_border});
            }}
            QScrollBar {{
                background-color: rgb({r_bg}, {g_bg}, {b_bg});
            }}
            QScrollBar::handle {{
                background-color: rgb({r_border}, {g_border}, {b_border});
            }}
            QScrollBar::add-line, QScrollBar::sub-line {{
                background-color: rgb({r_border}, {g_border}, {b_border});
            }}
            QGraphicsView {{
                background-color: rgb({r_bg}, {g_bg}, {b_bg});
                border: 1px solid rgb({r_border}, {g_border}, {b_border});
            }}
            QLineEdit {{
                color: rgb({r_text}, {g_text}, {b_text});
                background-color: rgb({r_input_bg}, {g_input_bg}, {b_input_bg});
                border: 1px solid rgb({r_border}, {g_border}, {b_border});
            }}
        """)

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
            self.settings.escape_counts = None  # make historic settings changeable again
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
        self.update_uov_inputs()
        self.render_fractal()

    def perturb_settings(self):
        logging.info("Perturbing fractal settings...")
        self.settings.u += np.random.normal(scale=0.05, size=self.settings.u.shape)
        self.settings.o += np.random.normal(scale=0.05, size=self.settings.o.shape)
        self.settings.v += np.random.normal(scale=0.05, size=self.settings.v.shape)
        self.update_uov_inputs()
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

    def save_settings(self):
        """Save the current fractal settings to a YAML file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Fractal Settings", "", "YAML Files (*.yaml);;All Files (*)", options=options)
        if file_path:
            settings_dict = self.settings_to_dict(self.settings)
            with open(file_path, "w") as file:
                yaml.dump(settings_dict, file, default_flow_style=False)
            logging.info(f"Settings saved to {file_path}")

    def load_settings(self):
        """Load fractal settings from a YAML file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Fractal Settings", "", "YAML Files (*.yaml);;All Files (*)", options=options)
        if file_path:
            with open(file_path, "r") as file:
                settings_dict = yaml.safe_load(file)
                self.settings = self.dict_to_settings(settings_dict)
            logging.info(f"Settings loaded from {file_path}")
            self.update_uov_inputs()
            self.render_fractal()

    @staticmethod
    def settings_to_dict(settings):
        """Convert FractalSettings to a dictionary for YAML serialization."""
        return {
            "u": settings.u.tolist(),  # Convert numpy array to list
            "o": settings.o.tolist(),  # Convert numpy array to list
            "v": settings.v.tolist(),  # Convert numpy array to list
            "center": list(settings.center),  # Convert tuple to list for YAML compatibility
            "rotation": settings.rotation,
            "scale": settings.scale,
            "escape_counts": None,  # Skip escape_counts for serialization
        }

    @staticmethod
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
            if self.selection_rect:
                self.handle_selection()
                self.start_pos = None
            return True
        return super().eventFilter(source, event)

    def draw_selection_rectangle(self, start_pos, end_pos):
        """Draw the selection rectangle with aspect ratio constraint and boundary checks."""
        if self.selection_rect_visual:
            self.graphics_scene.removeItem(self.selection_rect_visual)

        # Map start and end positions to scene coordinates
        start_scene = self.graphics_view.mapToScene(start_pos)
        end_scene = self.graphics_view.mapToScene(end_pos)

        # Calculate dx and dy while constraining to the aspect ratio
        dx = abs(end_scene.x() - start_scene.x())
        dy = abs(end_scene.y() - start_scene.y())

        if dx / RESOLUTION[0] > dy / RESOLUTION[1]:
            dy = dx / ASPECT_RATIO
        else:
            dx = dy * ASPECT_RATIO

        # Adjust for dragging direction
        if end_scene.x() < start_scene.x():
            dx = -dx
        if end_scene.y() < start_scene.y():
            dy = -dy

        # Constrain the rectangle to stay within the scene boundaries
        constrained_end_scene = QPointF(start_scene.x() + dx, start_scene.y() + dy)
        scene_rect = self.graphics_scene.sceneRect()
        if not scene_rect.contains(start_scene) or not scene_rect.contains(constrained_end_scene):
            return

        # Create and display the selection rectangle
        self.selection_rect = QRectF(start_scene, constrained_end_scene)
        self.selection_rect_visual = self.graphics_scene.addRect(self.selection_rect, QPen(Qt.red, 2))

    def handle_selection(self):
        """Handle rectangle selection to adjust the fractal view."""
        if self.selection_rect_visual:
            self.graphics_scene.removeItem(self.selection_rect_visual)
            self.selection_rect_visual = None

        # Get top-left and bottom-right corners of the rectangle
        top_left = self.selection_rect.topLeft()
        bottom_right = self.selection_rect.bottomRight()
        x0, y0 = top_left.x(), top_left.y()
        x1, y1 = bottom_right.x(), bottom_right.y()

        # Calculate the width and height of the rectangle in pixels
        width_pixels = abs(x1 - x0)
        height_pixels = abs(y1 - y0)

        if width_pixels < self.MIN_RECTANGLE[0] or height_pixels < self.MIN_RECTANGLE[1]:
            logging.info("Selection too small!")
            return

        # Normalize the rectangle center and adjust the fractal's center and scale
        cx = (x0 + x1) / 2 - self.graphics_scene.sceneRect().x()
        cy = (y0 + y1) / 2 - self.graphics_scene.sceneRect().y()

        cx_normalized = (cx / self.graphics_scene.sceneRect().width() - 0.5) * RESOLUTION[0] / RESOLUTION[1]
        cy_normalized = (cy / self.graphics_scene.sceneRect().height() - 0.5)

        self.settings.center = (
            self.settings.center[0] + cx_normalized * self.settings.scale,
            self.settings.center[1] + cy_normalized * self.settings.scale
        )
        self.settings.scale *= min(width_pixels / RESOLUTION[0], height_pixels / RESOLUTION[1])

        self.render_fractal()
        self.selection_rect = None


if __name__ == "__main__":
    mandelbrot_settings = FractalSettings(
        u=np.array([1, 0, 0, 0, 2, 0], dtype=np.float64),
        o=np.array([0, 0, 0, 0, 2, 0], dtype=np.float64),
        v=np.array([0, 1, 0, 0, 2, 0], dtype=np.float64),
        center=(0, 0),
        rotation=0,
        scale=4.0,
    )

    app = QApplication(sys.argv)
    main_window = FractalApp(mandelbrot_settings)
    main_window.show()
    sys.exit(app.exec())

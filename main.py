import sys
import logging
from copy import deepcopy
from time import time
from multiprocessing import Process, Queue

import yaml
import numpy as np
from matplotlib import colormaps
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout,
    QGraphicsView, QGraphicsScene, QGridLayout, QFileDialog, QLineEdit, QLabel, QCheckBox, QGroupBox,
    QDialog, QMessageBox
)
from PyQt5.QtCore import QTimer

from PyQt5.QtWidgets import QComboBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor, QBrush

from utils.parameters import sample_plane
from utils.fractal import compute_fractal
from utils.datatypes import FractalSettings
from utils.styles import get_stylesheet, get_toggled_style, get_untoggled_style

# Setup logging
LOG_FILE = "log.txt"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


RESOLUTION = (1000, 800)
# RESOLUTION = (1920, 1080)
ASPECT_RATIO = RESOLUTION[0] / RESOLUTION[1]
START_ITERATIONS = 100
ITERATION_GROWTH = 20
ESCAPE_RADIUS = 2.0

mandelbrot_settings = FractalSettings(
    u=np.array([1, 0, 0, 0, 2, 0], dtype=np.float64),
    o=np.array([0, 0, 0, 0, 2, 0], dtype=np.float64),
    v=np.array([0, 1, 0, 0, 2, 0], dtype=np.float64),
    center=(0, 0),
    rotation=0,
    scale=4.0,
)


def rgba_to_rgb(color):
    return [int(c * 255) for c in color[:3]]


class FractalWorker(QThread):
    finished = pyqtSignal(np.ndarray)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings

    def run(self):
        """Perform fractal computation in a separate thread."""
        logging.info("Starting fractal computation...")
        start_time = time()
        sampled_points = sample_plane(
            self.settings.u,
            self.settings.o,
            self.settings.v,
            self.settings.center,
            self.settings.rotation,
            self.settings.scale,
            RESOLUTION,
        )
        sample_time = time()
        max_iterations = START_ITERATIONS + ITERATION_GROWTH * np.log(1 / self.settings.scale)
        escape_counts = compute_fractal(
            sampled_points,
            max_iterations=int(max_iterations),
            escape_radius=ESCAPE_RADIUS
        )
        end_time = time()
        logging.info(f"Fractal computation completed in {end_time - start_time:.2f} seconds. {sample_time - start_time:.2f} seconds for paramerters.")
        self.finished.emit(escape_counts)


class FractalApp(QMainWindow):
    MIN_RECTANGLE = (5, 5)  # Minimum rectangle size in pixels
    INPUT_WIDTH = 60
    CONTROLLS_WIDTH = INPUT_WIDTH * 4
    VECTOR_COMPONENT_NAMES = ["cₐ", "cᵦ", "zₐ", "zᵦ", "pₐ", "pᵦ"]
    VECTOR_COMPONENT_NAME_INDICES = {name: i for i, name in enumerate(VECTOR_COMPONENT_NAMES)}
    DEFAULT_SAVE_PATH = "./saves"
    DEFAULT_EXPORT_PATH = "./exports"

    def __init__(self, initial_settings):
        super().__init__()
        self.settings = deepcopy(initial_settings)
        self.history = [deepcopy(initial_settings)]

        self.update_colormap("inferno", render=False)
        self.init_ui()
        self.render_fractal()

    def init_ui(self):
        self.setWindowTitle("6D Fractal Explorer")
        # self.setGeometry(100, 100, 1000, 600)
        self.showMaximized()

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
        """Set up the control buttons and input fields with compact frames."""
        controls_layout = QVBoxLayout()

        controls_layout.addWidget(self.create_settings_group(), alignment=Qt.AlignTop)
        controls_layout.addWidget(self.create_movement_group(), alignment=Qt.AlignTop)
        controls_layout.addWidget(self.create_translation_group(), alignment=Qt.AlignTop)
        controls_layout.addWidget(self.create_rotation_group(), alignment=Qt.AlignTop)
        controls_layout.addWidget(self.create_parameters_group(), alignment=Qt.AlignTop)

        controls_layout.addStretch()
        return controls_layout

    def create_settings_group(self):
        """Create the Settings group combining file, history, and color controls."""
        settings_group = QGroupBox("Settings")
        settings_group.setMaximumWidth(self.CONTROLLS_WIDTH)
        settings_layout = QVBoxLayout()

        # File controls: Load and Save
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.create_button("Load", "Load settings from a file", self.load_settings))
        file_layout.addWidget(self.create_button("Save", "Save current settings to a file", self.save_settings))
        file_layout.addWidget(self.create_button("Export", "Export fractal as an image", self.export_fractal))

        settings_layout.addLayout(file_layout)

        # History controls: Undo and Reset
        history_layout = QHBoxLayout()
        history_layout.addWidget(self.create_button("Undo", "Undo last action (Shortcut: Backspace)", self.go_back))
        history_layout.addWidget(self.create_button("Reset", "Reset view (Shortcut: Home)", self.reset_view))
        settings_layout.addLayout(history_layout)

        # Color controls
        settings_layout.addWidget(self.setup_colormap_dropdown())

        settings_group.setLayout(settings_layout)
        return settings_group

    def create_movement_group(self):
        """Create the Movement group with zoom, move, rotation, and additional parameters."""
        movement_group = QGroupBox("Movement")
        movement_group.setToolTip("Movement along the view plane.")
        movement_group.setMaximumWidth(self.CONTROLLS_WIDTH)
        movement_layout = QGridLayout()

        # Add zoom and movement controls
        movement_layout.addWidget(self.create_button("-", "Zoom Out (Shortcut: E)", self.zoom_out), 0, 0)
        movement_layout.addWidget(self.create_button("↑", "Move Up (Shortcut: W)", lambda: self.move("W")), 0, 1)
        movement_layout.addWidget(self.create_button("+", "Zoom In (Shortcut: Q)", self.zoom_in), 0, 2)
        movement_layout.addWidget(self.create_button("←", "Move Left (Shortcut: A)", lambda: self.move("A")), 1, 0)
        movement_layout.addWidget(self.create_button("↓", "Move Down (Shortcut: S)", lambda: self.move("S")), 1, 1)
        movement_layout.addWidget(self.create_button("→", "Move Right (Shortcut: D)", lambda: self.move("D")), 1, 2)
        movement_layout.addWidget(self.create_button("↺", "Rotate CCW (Shortcut: R)", lambda: self.rotate("CCW")), 2, 0)
        movement_layout.addWidget(self.create_button("↻", "Rotate CW (Shortcut: F)", lambda: self.rotate("CW")), 2, 2)

        # Add Center (Offset) fields
        center_layout = QHBoxLayout()
        center_label = QLabel("Center:")
        center_label.setStyleSheet("border: none;")
        center_label.setAlignment(Qt.AlignRight)
        self.center_x_field = QLineEdit(str(self.settings.center[0]))
        self.center_x_field.setFixedWidth(self.INPUT_WIDTH)
        self.center_x_field.setToolTip("X offset of the fractal center")
        self.center_x_field.returnPressed.connect(self.update_center)
        self.center_y_field = QLineEdit(str(self.settings.center[1]))
        self.center_y_field.setFixedWidth(self.INPUT_WIDTH)
        self.center_y_field.setToolTip("Y offset of the fractal center")
        self.center_y_field.returnPressed.connect(self.update_center)
        center_layout.addWidget(center_label)
        center_layout.addWidget(self.center_x_field)
        center_layout.addWidget(self.center_y_field)
        movement_layout.addLayout(center_layout, 3, 0, 1, 3)

        # Add Scale field
        scale_layout = QHBoxLayout()
        scale_label = QLabel("Scale:")
        scale_label.setStyleSheet("border: none;")
        scale_label.setAlignment(Qt.AlignRight)
        self.scale_field = QLineEdit(str(self.settings.scale))
        self.scale_field.setFixedWidth(self.INPUT_WIDTH)
        self.scale_field.setToolTip("Scale of the fractal (zoom level)")
        self.scale_field.returnPressed.connect(self.update_scale)
        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(self.scale_field)
        movement_layout.addLayout(scale_layout, 4, 0, 1, 3)

        # Add Rotation field
        rotation_layout = QHBoxLayout()
        rotation_label = QLabel("Rotation:")
        rotation_label.setStyleSheet("border: none;")
        rotation_label.setAlignment(Qt.AlignRight)
        self.rotation_field = QLineEdit(str(self.settings.rotation))
        self.rotation_field.setFixedWidth(self.INPUT_WIDTH)
        self.rotation_field.setToolTip("Rotation angle of the fractal (in radians)")
        self.rotation_field.returnPressed.connect(self.update_rotation)
        rotation_layout.addWidget(rotation_label)
        rotation_layout.addWidget(self.rotation_field)
        movement_layout.addLayout(rotation_layout, 5, 0, 1, 3)

        movement_group.setLayout(movement_layout)
        return movement_group

    def create_parameters_group(self):
        """Create the Parameters group with vector controls and randomize/perturb options."""
        parameters_group = QGroupBox("Parameters")
        parameters_group.setToolTip("Change the 3 points that define the view plane in parameter space.")
        parameters_group.setMaximumWidth(self.CONTROLLS_WIDTH)

        parameters_layout = QVBoxLayout()
        parameters_layout.addLayout(self.setup_uov_inputs())

        # Randomize and Perturb Buttons
        randomize_perturb_layout = QHBoxLayout()
        randomize_perturb_layout.addWidget(self.create_button("Randomize", "Randomize Settings (Shortcut: X)", self.randomize_settings))
        randomize_perturb_layout.addWidget(self.create_button("Perturb", "Perturb Settings (Shortcut: Z)", self.perturb_settings))
        parameters_layout.addLayout(randomize_perturb_layout)

        parameters_group.setLayout(parameters_layout)
        return parameters_group

    def setup_colormap_dropdown(self):
        """Set up a dropdown menu for selecting colormaps with a label."""
        colormap_layout = QHBoxLayout()

        # Create a borderless label for the colormap
        colormap_label = QLabel("Colors:")
        colormap_label.setStyleSheet("border: none;")
        colormap_label.setAlignment(Qt.AlignRight)

        # Create the dropdown menu for colormaps
        colormap_dropdown = QComboBox()
        colormap_dropdown.setToolTip("Select a colormap for the fractal")
        colormap_dropdown.addItems(sorted(colormaps.keys()))  # Add all available colormaps
        colormap_dropdown.setCurrentText("inferno")  # Default colormap
        colormap_dropdown.currentTextChanged.connect(self.update_colormap)

        # Add the label and dropdown to the layout
        colormap_layout.addWidget(colormap_label)
        colormap_layout.addWidget(colormap_dropdown)

        # Create a container widget to return
        colormap_widget = QWidget()
        colormap_widget.setLayout(colormap_layout)
        return colormap_widget

    def setup_uov_inputs(self):
        """Create compact input fields for u, o, v vectors with toggleable labels."""
        uov_layout = QGridLayout()
        self.u_fields = []
        self.o_fields = []
        self.v_fields = []
        self.row_toggled = {i: False for i in range(6)}  # Track toggled state for rows
        self.col_toggled = {i: False for i in range(3)}  # Track toggled state for columns

        # Row labels with toggle behavior
        for row, label in enumerate(self.VECTOR_COMPONENT_NAMES):
            row_label = QLabel(label)
            row_label.setObjectName(f"row_{row}")  # Assign a unique objectName
            row_label.setToolTip("Toogle to exempt from randomization.")
            row_label.setStyleSheet(self.get_toggle_style(self.row_toggled[row]))
            row_label.setAlignment(Qt.AlignCenter)
            row_label.mousePressEvent = lambda event, r=row: self.toggle_row(r)  # Bind toggle event
            row_label.setFixedWidth(self.INPUT_WIDTH)
            uov_layout.addWidget(row_label, row + 1, 0)  # First column for row labels

        # Column headers with toggle behavior
        column_labels = ["Top", "Center", "Right"]
        fields = [self.u_fields, self.o_fields, self.v_fields]
        values = [self.settings.u, self.settings.o, self.settings.v]
        for col, (header, field_list, value) in enumerate(zip(column_labels, fields, values)):
            col_label = QLabel(header)
            col_label.setObjectName(f"col_{col}")  # Assign a unique objectName
            col_label.setToolTip("Toogle to exempt from randomization.")
            col_label.setStyleSheet(self.get_toggle_style(self.col_toggled[col]))
            col_label.setAlignment(Qt.AlignCenter)
            col_label.mousePressEvent = lambda event, c=col: self.toggle_column(c)  # Bind toggle event
            uov_layout.addWidget(col_label, 0, col + 1)  # Column headers (shifted by 1)

        # paramerters
        for col, (header, field_list, value) in enumerate(zip(column_labels, fields, values)):
            for row in range(6):
                line_edit = QLineEdit(str(value[row]))
                line_edit.setToolTip(f"{header} Component {self.VECTOR_COMPONENT_NAMES[row]}")
                line_edit.setFixedWidth(self.INPUT_WIDTH)
                line_edit.returnPressed.connect(self.update_uov)
                uov_layout.addWidget(line_edit, row + 1, col + 1)
                field_list.append(line_edit)

        return uov_layout

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

    def create_translation_group(self):
        """Create a 6x3 translation control table with column headers between the up and down arrows."""
        translation_group = QGroupBox("Translation")
        translation_group.setToolTip("Translating the view plane in parameter space.")
        translation_group.setMaximumWidth(self.CONTROLLS_WIDTH)

        translation_layout = QVBoxLayout()
        table_layout = QGridLayout()

        # Up arrow buttons
        for col, header in enumerate(self.VECTOR_COMPONENT_NAMES):
            up_button = QPushButton("↑")
            up_button.setToolTip(f"Translate positively along {header}")
            up_button.clicked.connect(lambda _, dim=header: self.translate_plane(dim, 1))
            table_layout.addWidget(up_button, 0, col)  # Place up arrows in the first row (index 0)

        # Column headers
        for col, header in enumerate(self.VECTOR_COMPONENT_NAMES):
            header_label = QLabel(header)
            header_label.setStyleSheet("border: none;")
            header_label.setAlignment(Qt.AlignCenter)
            table_layout.addWidget(header_label, 1, col)  # Place headers in the second row (index 1)

        # Down arrow buttons
        for col, header in enumerate(self.VECTOR_COMPONENT_NAMES):
            down_button = QPushButton("↓")
            down_button.setToolTip(f"Translate negatively along {header}")
            down_button.clicked.connect(lambda _, dim=header: self.translate_plane(dim, -1))
            table_layout.addWidget(down_button, 2, col)  # Place down arrows in the third row (index 2)

        # Add the table to the translation layout
        translation_layout.addLayout(table_layout)

        # Add displacement field
        displacement_layout = QHBoxLayout()
        displacement_label = QLabel("Distance:")
        displacement_label.setStyleSheet("border: none;")
        displacement_label.setAlignment(Qt.AlignRight)
        self.displacement = QLineEdit("0.1")
        self.displacement.setFixedWidth(self.INPUT_WIDTH)
        self.displacement.setToolTip("Set the displacement magnitude for translation")
        displacement_layout.addWidget(displacement_label)
        displacement_layout.addWidget(self.displacement)
        translation_layout.addLayout(displacement_layout)

        translation_group.setLayout(translation_layout)
        return translation_group

    def create_rotation_group(self):
        """Create a full rotation control table with bidirectional rotation buttons."""
        rotation_group = QGroupBox("Rotation")
        rotation_group.setToolTip("Rotating the view plane in parameter space.")
        rotation_group.setMaximumWidth(self.CONTROLLS_WIDTH)

        rotation_layout = QVBoxLayout()
        table_layout = QGridLayout()

        headers = self.VECTOR_COMPONENT_NAMES

        # Add headers for rows and columns
        for col, header in enumerate(headers):
            header_label = QLabel(header)
            header_label.setStyleSheet("border: none;")
            header_label.setAlignment(Qt.AlignCenter)
            table_layout.addWidget(header_label, 0, col + 1)  # Top headers

            header_label = QLabel(header)
            header_label.setStyleSheet("border: none;")
            header_label.setAlignment(Qt.AlignCenter)
            table_layout.addWidget(header_label, col + 1, 0)  # Left headers

        # Add bidirectional rotation buttons
        for row in range(len(headers)):
            for col in range(len(headers)):
                if row != col:  # No self-rotation
                    rotation_button = QPushButton("↻" if col > row else "↺")
                    tooltip = f"Rotate along the plane of {headers[col]} and {headers[row]}"
                    rotation_button.setToolTip(tooltip)
                    rotation_button.clicked.connect(lambda _, a=row, b=col: self.rotate_plane(a, b))
                    table_layout.addWidget(rotation_button, row + 1, col + 1)

        # Add the table layout to the rotation layout
        rotation_layout.addLayout(table_layout)

        # Add rotation amount field
        rotation_amount_layout = QHBoxLayout()
        rotation_amount_label = QLabel("Angle:")
        rotation_amount_label.setStyleSheet("border: none;")
        rotation_amount_label.setAlignment(Qt.AlignRight)
        self.rotation_amount = QLineEdit("0.1")
        self.rotation_amount.setFixedWidth(self.INPUT_WIDTH)
        self.rotation_amount.setToolTip("Set the rotation angle in radians")
        rotation_amount_layout.addWidget(rotation_amount_label)
        rotation_amount_layout.addWidget(self.rotation_amount)
        rotation_layout.addLayout(rotation_amount_layout)
        rotation_group.setLayout(rotation_layout)

        return rotation_group

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

    def update_movement_inputs(self):
        self.center_x_field.setText(str(self.settings.center[0]))
        self.center_x_field.setCursorPosition(0)

        self.center_y_field.setText(str(self.settings.center[1]))
        self.center_y_field.setCursorPosition(0)

        self.scale_field.setText(str(self.settings.scale))
        self.scale_field.setCursorPosition(0)

        self.rotation_field.setText(str(self.settings.rotation))
        self.rotation_field.setCursorPosition(0)

    def update_colormap(self, colormap_name, render=True):
        """Update the colormap and re-render the fractal."""
        logging.info(f"Changing colormap to: {colormap_name}")
        self.colormap = colormaps.get_cmap(colormap_name)
        self.update_interface_color()
        if render:
            self.render_fractal()

    def update_interface_color(self):
        """Update the interface colors to match the colormap."""
        selector_color = self.colormap(0.5)
        background_color = self.colormap(0.0)  # RGBA tuple for background
        text_color = self.colormap(0.3)  # RGBA tuple for text
        border_color = self.colormap(0.5)  # RGBA tuple for borders
        border_group_color = self.colormap(0.1)  # RGBA tuple for group borders
        input_bg_color = self.colormap(0.1)  # Slightly lighter for input fields
        untoggled_bg_color = self.colormap(0.0)  # Background for untoggled elements
        untoggled_text_color = self.colormap(0.3)  # Text for untoggled elements
        toggled_bg_color = self.colormap(0.3)  # Background for toggled elements
        toggled_text_color = self.colormap(0.0)  # Text for toggled elements

        # Convert RGBA to RGB
        r_bg, g_bg, b_bg = rgba_to_rgb(background_color)
        r_text, g_text, b_text = rgba_to_rgb(text_color)
        r_border, g_border, b_border = rgba_to_rgb(border_color)
        r_border_group, g_border_group, b_border_group = rgba_to_rgb(border_group_color)
        r_input_bg, g_input_bg, b_input_bg = rgba_to_rgb(input_bg_color)
        r_untoggled_bg, g_untoggled_bg, b_untoggled_bg = rgba_to_rgb(untoggled_bg_color)
        r_untoggled_text, g_untoggled_text, b_untoggled_text = rgba_to_rgb(untoggled_text_color)
        r_toggled_bg, g_toggled_bg, b_toggled_bg = rgba_to_rgb(toggled_bg_color)
        r_toggled_text, g_toggled_text, b_toggled_text = rgba_to_rgb(toggled_text_color)

        # Selector rectangle
        selector_color_rgb = rgba_to_rgb(selector_color)
        self.selector_color = QColor(*selector_color_rgb)

        # Save toggled and untoggled styles
        self.toggled_style = get_toggled_style().format(
            r_toggled_bg=r_toggled_bg, g_toggled_bg=g_toggled_bg, b_toggled_bg=b_toggled_bg,
            r_toggled_text=r_toggled_text, g_toggled_text=g_toggled_text, b_toggled_text=b_toggled_text,
        )
        self.untoggled_style = get_untoggled_style().format(
            r_untoggled_bg=r_untoggled_bg, g_untoggled_bg=g_untoggled_bg, b_untoggled_bg=b_untoggled_bg,
            r_untoggled_text=r_untoggled_text, g_untoggled_text=g_untoggled_text, b_untoggled_text=b_untoggled_text,
        )

        # Variables for the stylesheet
        variables = {
            "r_bg": r_bg, "g_bg": g_bg, "b_bg": b_bg,
            "r_text": r_text, "g_text": g_text, "b_text": b_text,
            "r_border": r_border, "g_border": g_border, "b_border": b_border,
            "r_border_group": r_border_group, "g_border_group": g_border_group, "b_border_group": b_border_group,
            "r_input_bg": r_input_bg, "g_input_bg": g_input_bg, "b_input_bg": b_input_bg,
            "r_untoggled_bg": r_untoggled_bg, "g_untoggled_bg": g_untoggled_bg, "b_untoggled_bg": b_untoggled_bg,
            "r_untoggled_text": r_untoggled_text, "g_untoggled_text": g_untoggled_text, "b_untoggled_text": b_untoggled_text,
            "r_toggled_bg": r_toggled_bg, "g_toggled_bg": g_toggled_bg, "b_toggled_bg": b_toggled_bg,
            "r_toggled_text": r_toggled_text, "g_toggled_text": g_toggled_text, "b_toggled_text": b_toggled_text,
        }

        # Apply the stylesheet
        stylesheet = get_stylesheet().format(**variables)
        self.setStyleSheet(stylesheet)

        # Update row and column labels to reflect their toggled state
        for row in range(6):
            label = self.findChild(QLabel, f"row_{row}")
            if label:
                label.setStyleSheet(self.toggled_style if self.row_toggled[row] else self.untoggled_style)
        for col in range(3):
            label = self.findChild(QLabel, f"col_{col}")
            if label:
                label.setStyleSheet(self.toggled_style if self.col_toggled[col] else self.untoggled_style)

    def update_center(self):
        """Update the center (offset) of the fractal."""
        try:
            x = float(self.center_x_field.text())
            y = float(self.center_y_field.text())
            self.settings.center = (x, y)
            logging.info(f"Updated center to: {self.settings.center}")
            self.render_fractal()
        except ValueError:
            logging.warning("Invalid input for center. Please enter numeric values.")

    def update_scale(self):
        """Update the scale (zoom level) of the fractal."""
        try:
            scale = float(self.scale_field.text())
            if scale > 0:
                self.settings.scale = scale
                logging.info(f"Updated scale to: {self.settings.scale}")
                self.render_fractal()
            else:
                logging.warning("Scale must be positive.")
        except ValueError:
            logging.warning("Invalid input for scale. Please enter a numeric value.")

    def update_rotation(self):
        """Update the rotation angle of the fractal."""
        try:
            rotation = float(self.rotation_field.text())
            self.settings.rotation = rotation
            logging.info(f"Updated rotation to: {self.settings.rotation} radians")
            self.render_fractal()
        except ValueError:
            logging.warning("Invalid input for rotation. Please enter a numeric value.")

    def get_toggle_style(self, toggled):
        """Return the stylesheet for a toggled or untoggled label."""
        return self.toggled_style if toggled else self.untoggled_style

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

        self.update_uov_inputs()
        self.update_movement_inputs()

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
        for col in range(3):
            if not self.col_toggled[col]:  # Column is not toggled
                new_values = np.random.normal(size=6)
                for row in range(6):
                    if not self.row_toggled[row]:  # Row is not toggled
                        if col == 0:  # u
                            self.settings.u[row] = new_values[row]
                        elif col == 1:  # o
                            self.settings.o[row] = new_values[row]
                        elif col == 2:  # v
                            self.settings.v[row] = new_values[row]
        self.render_fractal()

    def perturb_settings(self):
        logging.info("Perturbing fractal settings...")
        for col in range(3):
            if not self.col_toggled[col]:  # Column is not toggled
                perturbation = np.random.normal(scale=0.05, size=6)
                for row in range(6):
                    if not self.row_toggled[row]:  # Row is not toggled
                        if col == 0:  # u
                            self.settings.u[row] += perturbation[row]
                        elif col == 1:  # o
                            self.settings.o[row] += perturbation[row]
                        elif col == 2:  # v
                            self.settings.v[row] += perturbation[row]
        self.render_fractal()

    def translate_plane(self, dimension, direction):
        """
        Translate the entire plane along the specified dimension.

        Args:
            dimension (str): The axis to translate.
            direction (int): +1 for positive, -1 for negative translation.
        """
        try:
            displacement = float(self.displacement.text()) * direction
        except ValueError:
            logging.warning("Invalid displacement value. Please enter a numeric value.")
            return

        translation_vector = np.zeros(6)  # Initialize a zero vector for translation

        if dimension in self.VECTOR_COMPONENT_NAME_INDICES:
            translation_vector[self.VECTOR_COMPONENT_NAME_INDICES[dimension]] = displacement
        else:
            logging.warning(f"Unknown dimension: {dimension}")
            return

        # Apply the translation to the entire plane (u, o, v)
        self.settings.u += translation_vector
        self.settings.o += translation_vector
        self.settings.v += translation_vector

        # Update the UI and re-render the fractal
        self.render_fractal()
        logging.warning(f"Translated {dimension} by {displacement} in the {direction} direction.")

    def rotate_plane(self, axis1_index, axis2_index):
        """
        Rotate the entire plane around the specified axis defined by two dimensions,
        considering the center offset.

        Args:
            axis1_index (int): The index of the first component.
            axis2_index (int): The index of the second component.
        """
        try:
            angle_radians = float(self.rotation_amount.text())
        except ValueError:
            logging.warning("Invalid rotation amount. Please enter a numeric value.")
            return

        cos_theta = np.cos(angle_radians)
        sin_theta = np.sin(angle_radians)

        # Rotation matrix for 2D plane rotation in 6D space
        rotation_matrix = np.eye(6)
        rotation_matrix[axis1_index, axis1_index] = cos_theta
        rotation_matrix[axis1_index, axis2_index] = -sin_theta
        rotation_matrix[axis2_index, axis1_index] = sin_theta
        rotation_matrix[axis2_index, axis2_index] = cos_theta

        # Translate to origin, rotate, and translate back
        translation_vector = np.zeros(6)
        translation_vector[self.VECTOR_COMPONENT_NAME_INDICES["cₐ"]] = self.settings.center[0]
        translation_vector[self.VECTOR_COMPONENT_NAME_INDICES["cᵦ"]] = self.settings.center[1]

        self.settings.u -= translation_vector  # Translate to origin
        self.settings.o -= translation_vector
        self.settings.v -= translation_vector

        # Apply rotation
        self.settings.u = rotation_matrix @ self.settings.u
        self.settings.o = rotation_matrix @ self.settings.o
        self.settings.v = rotation_matrix @ self.settings.v

        self.settings.u += translation_vector  # Translate back
        self.settings.o += translation_vector
        self.settings.v += translation_vector

        # Update the UI and re-render
        self.render_fractal()
        logging.info(f"Rotated around axis ({self.VECTOR_COMPONENT_NAMES[axis1_index]}, {self.VECTOR_COMPONENT_NAMES[axis2_index]}) by {angle_radians:.4f} radians.")

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
        elif event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showMaximized()
            else:
                self.showFullScreen()

    def save_settings(self):
        """Save the current fractal settings to a YAML file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Fractal Settings",
            self.DEFAULT_SAVE_PATH,
            "YAML Files (*.yaml);;All Files (*)",
            options=options,
        )
        if file_path:
            settings_dict = self.settings_to_dict(self.settings)
            with open(file_path, "w") as file:
                yaml.dump(settings_dict, file, default_flow_style=False)
            logging.info(f"Settings saved to {file_path}")

    def load_settings(self):
        """Load fractal settings from a YAML file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Fractal Settings",
            self.DEFAULT_SAVE_PATH,
            "YAML Files (*.yaml);;All Files (*)",
            options=options,
        )
        if file_path:
            with open(file_path, "r") as file:
                settings_dict = yaml.safe_load(file)
                self.settings = self.dict_to_settings(settings_dict)
            logging.info(f"Settings loaded from {file_path}")
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

    def toggle_row(self, row):
        """Toggle the state of a row."""
        self.row_toggled[row] = not self.row_toggled[row]
        label = self.findChild(QLabel, f"row_{row}")
        if label:
            label.setStyleSheet(self.get_toggle_style(self.row_toggled[row]))

    def toggle_column(self, col):
        """Toggle the state of a column."""
        self.col_toggled[col] = not self.col_toggled[col]
        label = self.findChild(QLabel, f"col_{col}")
        if label:
            label.setStyleSheet(self.get_toggle_style(self.col_toggled[col]))

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
        self.selection_rect_visual = self.graphics_scene.addRect(self.selection_rect, QPen(self.selector_color, 2))

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

    def export_fractal(self):
        """Export the fractal as an image at a specified resolution."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Fractal Image",
            self.DEFAULT_EXPORT_PATH,
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)",
            options=options,
        )

        if not file_path:
            return

        if not file_path.endswith('.png') or file_path.endswith('.jpg'):
            file_path += ".png"

        resolution_dialog = QDialog(self)
        resolution_dialog.setWindowTitle("Specify Resolution")
        layout = QGridLayout()

        width_label = QLabel("Width:")
        layout.addWidget(width_label, 0, 0)
        width_input = QLineEdit(str(RESOLUTION[0]))
        layout.addWidget(width_input, 0, 1)

        height_label = QLabel("Height:")
        layout.addWidget(height_label, 1, 0)
        height_input = QLineEdit(str(RESOLUTION[1]))
        layout.addWidget(height_input, 1, 1)

        save_button = QPushButton("Save")
        layout.addWidget(save_button, 2, 0, 1, 2)
        resolution_dialog.setLayout(layout)

        def on_save():
            try:
                width = int(width_input.text())
                height = int(height_input.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter valid integers for resolution.")
            self.render_and_save_fractal_pil(file_path, (width, height))
            resolution_dialog.accept()

        save_button.clicked.connect(on_save)
        resolution_dialog.exec()

    def render_and_save_fractal_pil(self, file_path, resolution):
        """Render the fractal at a given resolution and save it to a file using Pillow."""
        logging.info(f"Exporting fractal to {file_path} at resolution {resolution}...")

        # Compute fractal data
        sampled_points = sample_plane(
            self.settings.u,
            self.settings.o,
            self.settings.v,
            self.settings.center,
            self.settings.rotation,
            self.settings.scale,
            resolution,
        )
        max_iterations = START_ITERATIONS + ITERATION_GROWTH * np.log(1 / self.settings.scale)
        escape_counts = compute_fractal(
            sampled_points,
            max_iterations=int(max_iterations),
            escape_radius=ESCAPE_RADIUS,
        )

        # Normalize and apply the colormap
        min_val = escape_counts.min()
        max_val = escape_counts.max()
        normalized = (escape_counts - min_val) / (max_val - min_val)
        colored = (self.colormap(normalized)[:, :, :3] * 255).astype(np.uint8)

        # Create a PIL Image
        image = Image.fromarray(colored, mode="RGB")

        # Save the image
        image.save(file_path)
        logging.info(f"Fractal successfully exported to {file_path}.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = FractalApp(mandelbrot_settings)
    main_window.show()
    sys.exit(app.exec())

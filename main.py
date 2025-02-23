import logging
import sys
from copy import deepcopy
from time import time

import numpy as np
import yaml
from matplotlib import colormaps
from PIL import Image
from PyQt5.QtCore import QPointF, QRectF, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFontMetrics, QImage, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

from utils.cli import parse_args
from utils.fractal import compute_fractal
from utils.parameters import get_base_iterations, get_max_iterattions, sample_plane
from utils.plot import get_basis_projection_image
from utils.settings import default_settings, dict_to_settings, settings_to_dict
from utils.styles import get_font, get_stylesheet, get_toggled_style, get_untoggled_style

LOG_PATH = "log.txt"
DEFAULT_SAVE_PATH = "./saves"
DEFAULT_EXPORT_PATH = "./exports"

install_rich_traceback()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = RichHandler(rich_tracebacks=True)
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_PATH)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def rgba_to_rgb(color):
    return [int(c * 255) for c in color[:3]]


class FractalWorker(QThread):
    finished = pyqtSignal(tuple)

    def __init__(self, settings, resolution):
        super().__init__()
        self.settings = settings
        self.resolution = resolution

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
            self.resolution,
        )
        sample_time = time()
        max_iterations = get_max_iterattions(
            self.settings.base_iterations, self.settings.iterations_growth, self.settings.scale
        )
        escape_counts = compute_fractal(
            sampled_points,
            max_iterations=int(max_iterations),
            escape_radius=self.settings.escape_radius,
        )
        end_time = time()
        logging.info(
            f"Fractal computation completed in {end_time - start_time:.2f} seconds."
            f" {sample_time - start_time:.2f} seconds for paramerters."
        )
        self.finished.emit((escape_counts, sampled_points))


class FractalApp(QMainWindow):
    MIN_RECTANGLE = (5, 5)  # Minimum rectangle size in pixels
    INPUT_WIDTH = 60
    CONTROLLS_WIDTH = INPUT_WIDTH * 4
    CONTROLLS_AREA_WIDTH = CONTROLLS_WIDTH + INPUT_WIDTH // 2
    VECTOR_COMPONENT_NAMES = ["cₐ", "cᵦ", "zₐ", "zᵦ", "pₐ", "pᵦ"]
    VECTOR_COMPONENT_NAME_INDICES = {name: i for i, name in enumerate(VECTOR_COMPONENT_NAMES)}
    RANDOM_MEAN_DEFAULT = 0.0
    RANDOM_STD_DEFAULT = 1.0
    PERTURB_MEAN_DEFAULT = 0.0
    PERTURB_STD_DEFAULT = 0.1

    def __init__(self, args):
        super().__init__()
        self.load_settings(args.load)
        self.history = [deepcopy(self.settings)]
        # Current view
        self.current_escape_counts = None
        self.current_sampled_points = None
        # Rectangle selection
        self.start_pos = None
        self.selection_rect = None
        self.selection_rect_visual = None
        # Settings
        self.show_basis_vectors = False
        self.basis_vector_pixmap = None
        self.show_parameter_info = False

        self.update_colormap(self.settings.colormap)
        self.init_ui()
        self.reset_view()

    def init_ui(self):
        self.setWindowTitle("6D Fractal Explorer")
        # self.setGeometry(100, 100, 1000, 600)
        self.showMaximized()
        self.keyPressEvent = self.on_key

        # Default font
        app.setFont(get_font())

        # Main layout: Horizontal layout with fractal display and controls
        main_layout = QHBoxLayout()

        # Add the fractal display
        fractal_layout = self.setup_fractal_display()
        main_layout.addLayout(fractal_layout)

        # Add the control buttons
        controls_layout = self.setup_controls()
        main_layout.addWidget(controls_layout)

        # Parameter hover
        self.setup_parameter_info()

        # Main widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def resizeEvent(self, event):
        new_size = event.size()
        logging.info(f"Resizing to width: {new_size.width()}, height: {new_size.height()}")
        if hasattr(self, "graphics_view"):
            self.render_fractal()
        super().resizeEvent(event)

    def setup_fractal_display(self):
        # Fractal view
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setRenderHints(self.graphics_view.renderHints() | Qt.SmoothTransformation)

        # For rectangle selection
        self.graphics_view.setMouseTracking(True)
        self.graphics_view.viewport().installEventFilter(self)

        fractal_layout = QVBoxLayout()
        fractal_layout.addWidget(self.graphics_view)
        return fractal_layout

    def setup_controls(self):
        """Set up the control buttons and input fields with compact frames."""
        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.create_settings_group(), alignment=Qt.AlignTop)
        controls_layout.addWidget(self.create_movement_group(), alignment=Qt.AlignTop)
        controls_layout.addWidget(self.create_translation_group(), alignment=Qt.AlignTop)
        controls_layout.addWidget(self.create_rotation_group(), alignment=Qt.AlignTop)
        controls_layout.addWidget(self.create_parameters_group(), alignment=Qt.AlignTop)
        controls_layout.addWidget(self.create_randomize_group(), alignment=Qt.AlignTop)
        controls_layout.addStretch()

        # Create a scroll area for the controls
        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidget(controls_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Hide scrollbar until needed
        scroll_area.setMaximumWidth(self.CONTROLLS_AREA_WIDTH)

        return scroll_area

    def create_settings_group(self):
        """Create the Settings group combining file, history, and color controls."""
        settings_group = QGroupBox("Settings")
        settings_group.setMaximumWidth(self.CONTROLLS_WIDTH)
        settings_layout = QVBoxLayout()

        # File controls: Load and Save
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.create_button("Load", "Load settings from a file", self.load_settings_dialogue))
        file_layout.addWidget(self.create_button("Save", "Save current settings to a file", self.save_settings))
        file_layout.addWidget(self.create_button("Export", "Export fractal as an image", self.export_fractal))

        settings_layout.addLayout(file_layout)

        # History controls: Undo and Reset
        history_layout = QHBoxLayout()
        history_layout.addWidget(self.create_button("Undo", "Undo last action (Shortcut: Backspace)", self.go_back))
        history_layout.addWidget(self.create_button("Reset", "Reset view (Shortcut: Home)", self.reset_view))
        settings_layout.addLayout(history_layout)

        # Iterations controls
        iterations_layout = QHBoxLayout()
        iterations_label = QLabel("Iterations:")
        iterations_label.setAlignment(Qt.AlignRight)
        max_iterations = get_max_iterattions(
            self.settings.base_iterations, self.settings.iterations_growth, self.settings.scale
        )
        self.current_iterations_field = QLineEdit(str(int(max_iterations)))
        self.current_iterations_field.setToolTip("Current max iterations.")
        self.current_iterations_field.returnPressed.connect(self.update_iterations)
        self.current_iterations_field.setFixedWidth(self.INPUT_WIDTH)
        self.iterations_growth_field = QLineEdit(str(self.settings.iterations_growth))
        self.iterations_growth_field.setToolTip("Growth of max iterations dependent on scale.")
        self.iterations_growth_field.returnPressed.connect(self.update_iterations)
        self.iterations_growth_field.setFixedWidth(self.INPUT_WIDTH)
        iterations_layout.addWidget(iterations_label)
        iterations_layout.addWidget(self.current_iterations_field)
        iterations_layout.addWidget(self.iterations_growth_field)
        settings_layout.addLayout(iterations_layout)

        # Escape radius controls
        radius_layout = QHBoxLayout()
        radius_label = QLabel("Radius:")
        radius_label.setAlignment(Qt.AlignRight)
        self.radius_field = QLineEdit(str(self.settings.escape_radius))
        self.radius_field.setToolTip("Bailout radius.")
        self.radius_field.returnPressed.connect(self.update_radius)
        self.radius_field.setFixedWidth(self.INPUT_WIDTH)
        radius_layout.addWidget(radius_label)
        radius_layout.addWidget(self.radius_field)
        settings_layout.addLayout(radius_layout)

        # Visualization controls
        visualization_layout = QHBoxLayout()
        visualization_layout.addWidget(
            self.create_toggle_button(
                "Basis",
                "Show basis vector projection. (Shortcut: F4)",
                self.toggle_basis_vector_display,
            )
        )
        visualization_layout.addWidget(
            self.create_toggle_button(
                "Parameters",
                "Show parameters on hover. (Shortcut: F4)",
                self.toggle_parameter_info_display,
            )
        )
        settings_layout.addLayout(visualization_layout)

        # Color controls
        settings_layout.addLayout(self.setup_colormap_dropdown())

        settings_group.setLayout(settings_layout)
        return settings_group

    def create_toggle_button(self, label, tooltip, callbacks):
        """Create a reusable toggle button."""
        toggle_button = QPushButton(label)
        toggle_button.setCheckable(True)
        toggle_button.setToolTip(tooltip)
        toggle_button.clicked.connect(lambda: self.on_toggle_button_clicked(toggle_button, label, callbacks))
        return toggle_button

    def setup_parameter_info(self):
        self.parameter_info_label = QLabel(self)
        self.parameter_info_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.parameter_info_label.setVisible(False)
        self.parameter_info_label.setObjectName("hoverLabel")

    def on_toggle_button_clicked(self, button, label, callbacks):
        """Handle toggle button clicks."""
        if isinstance(callbacks, tuple) and len(callbacks) == 2:
            callback_checked, callback_unchecked = callbacks
        else:
            callback_checked, callback_unchecked = callbacks, callbacks

        if button.isChecked():
            logging.info(f"{label} toggled ON")
            button.setStyleSheet(self.get_toggle_style(True))
            callback_checked()
        else:
            logging.info(f"{label} toggled OFF")
            button.setStyleSheet(self.get_toggle_style(False))
            callback_unchecked()

    def create_movement_group(self):
        """Create the Movement group with zoom, move, rotation, and additional parameters."""
        movement_group = QGroupBox("Movement")
        movement_group.setToolTip("Movement along the view plane.")
        movement_group.setMaximumWidth(self.CONTROLLS_WIDTH)
        movement_layout = QGridLayout()

        # Add zoom and movement controls
        movement_layout.addWidget(self.create_button("-", "Zoom Out (Shortcut: E)", self.zoom_out), 0, 0)
        movement_layout.addWidget(
            self.create_button("↑", "Move Up (Shortcut: W)", lambda: self.move("W")),
            0,
            1,
        )
        movement_layout.addWidget(self.create_button("+", "Zoom In (Shortcut: Q)", self.zoom_in), 0, 2)
        movement_layout.addWidget(
            self.create_button("←", "Move Left (Shortcut: A)", lambda: self.move("A")),
            1,
            0,
        )
        movement_layout.addWidget(
            self.create_button("↓", "Move Down (Shortcut: S)", lambda: self.move("S")),
            1,
            1,
        )
        movement_layout.addWidget(
            self.create_button("→", "Move Right (Shortcut: D)", lambda: self.move("D")),
            1,
            2,
        )
        movement_layout.addWidget(
            self.create_button("↺", "Rotate CCW (Shortcut: R)", lambda: self.rotate("CCW")),
            2,
            0,
        )
        movement_layout.addWidget(
            self.create_button("↻", "Rotate CW (Shortcut: F)", lambda: self.rotate("CW")),
            2,
            2,
        )

        # Add Center (Offset) fields
        center_layout = QHBoxLayout()
        center_label = QLabel("Center:")
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

        parameters_group.setLayout(parameters_layout)
        return parameters_group

    def create_randomize_group(self):
        randomize_group = QGroupBox("Randomize")
        randomize_group.setToolTip("Change Points randomly.")
        randomize_group.setMaximumWidth(self.CONTROLLS_WIDTH)
        randomize_layout = QVBoxLayout()

        randomization_input_layout = QHBoxLayout()
        randomization_input_layout.addWidget(
            self.create_button("Randomize", "Randomize Settings (Shortcut: X)", self.randomize_settings)
        )
        self.mean_random_input = QLineEdit(str(self.RANDOM_MEAN_DEFAULT))
        self.mean_random_input.setFixedWidth(self.INPUT_WIDTH)
        self.mean_random_input.returnPressed.connect(self.randomize_settings)
        randomization_input_layout.addWidget(self.mean_random_input)
        self.std_random_input = QLineEdit(str(self.RANDOM_STD_DEFAULT))
        self.std_random_input.setFixedWidth(self.INPUT_WIDTH)
        self.std_random_input.returnPressed.connect(self.randomize_settings)
        randomization_input_layout.addWidget(self.std_random_input)
        randomize_layout.addLayout(randomization_input_layout)

        perturbation_input_layout = QHBoxLayout()
        perturbation_input_layout.addWidget(
            self.create_button("Perturb", "Perturb Settings (Shortcut: Z)", self.perturb_settings)
        )
        self.mean_perturb_input = QLineEdit(str(self.PERTURB_MEAN_DEFAULT))
        self.mean_perturb_input.setFixedWidth(self.INPUT_WIDTH)
        self.mean_perturb_input.returnPressed.connect(self.perturb_settings)
        perturbation_input_layout.addWidget(self.mean_perturb_input)
        self.std_perturb_input = QLineEdit(str(self.PERTURB_STD_DEFAULT))
        self.std_perturb_input.setFixedWidth(self.INPUT_WIDTH)
        self.std_perturb_input.returnPressed.connect(self.perturb_settings)
        perturbation_input_layout.addWidget(self.std_perturb_input)
        randomize_layout.addLayout(perturbation_input_layout)

        randomize_group.setLayout(randomize_layout)
        return randomize_group

    def setup_colormap_dropdown(self):
        """Set up a dropdown menu for selecting colormaps with a label."""
        colormap_layout = QHBoxLayout()

        # Create a label for the colormap
        colormap_label = QLabel("Colors:")
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
        return colormap_layout

    def setup_uov_inputs(self):
        """Create compact input fields for u, o, v vectors with toggleable buttons."""
        uov_layout = QGridLayout()
        self.u_fields = []
        self.o_fields = []
        self.v_fields = []
        self.row_toggled = {i: False for i in range(6)}  # Track toggled state for rows
        self.col_toggled = {i: False for i in range(3)}  # Track toggled state for columns

        # Row toggle buttons
        for row, label in enumerate(self.VECTOR_COMPONENT_NAMES):
            row_button = self.create_toggle_button(
                label=label,
                tooltip=f"Toggle to fix component {label}.",
                callbacks=(
                    lambda r=row: self.toggle_row_state(r, True),
                    lambda r=row: self.toggle_row_state(r, False),
                ),
            )
            row_button.setFixedWidth(self.CONTROLLS_WIDTH // 5)
            uov_layout.addWidget(row_button, row + 1, 0)  # First column for row toggles

        # Column toggle buttons
        column_labels = ["Top", "Center", "Right"]
        fields = [self.u_fields, self.o_fields, self.v_fields]
        values = [self.settings.u, self.settings.o, self.settings.v]
        for col, (header, field_list, value) in enumerate(zip(column_labels, fields, values)):
            col_button = self.create_toggle_button(
                label=header,
                tooltip=f"Toggle to fix the {header.lower()} point of the view plane.",
                callbacks=(
                    lambda c=col: self.toggle_column_state(c, True),
                    lambda c=col: self.toggle_column_state(c, False),
                ),
            )
            uov_layout.addWidget(col_button, 0, col + 1)  # Column headers (shifted by 1)

        # Parameters (fields)
        for col, (header, field_list, value) in enumerate(zip(column_labels, fields, values)):
            for row in range(6):
                line_edit = QLineEdit(str(value[row]))
                line_edit.setToolTip(f"{header} Component {self.VECTOR_COMPONENT_NAMES[row]}")
                line_edit.setFixedWidth(self.CONTROLLS_WIDTH // 5)
                line_edit.returnPressed.connect(self.update_uov)
                uov_layout.addWidget(line_edit, row + 1, col + 1)
                field_list.append(line_edit)

        return uov_layout

    def toggle_row_state(self, row, state):
        """Toggle the state of a row."""
        self.row_toggled[row] = state
        logging.info(f"Row {row} toggled {'ON' if state else 'OFF'}")

    def toggle_column_state(self, col, state):
        """Toggle the state of a column."""
        self.col_toggled[col] = state
        logging.info(f"Column {col} toggled {'ON' if state else 'OFF'}")

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
        move_layout.addWidget(
            self.create_button("↑", "Move Up (Shortcut: W)", lambda: self.move("W")),
            0,
            1,
        )
        move_layout.addWidget(
            self.create_button("←", "Move Left (Shortcut: A)", lambda: self.move("A")),
            1,
            0,
        )
        move_layout.addWidget(
            self.create_button("↓", "Move Down (Shortcut: S)", lambda: self.move("S")),
            1,
            1,
        )
        move_layout.addWidget(
            self.create_button("→", "Move Right (Shortcut: D)", lambda: self.move("D")),
            1,
            2,
        )
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
            header_label.setAlignment(Qt.AlignCenter)
            table_layout.addWidget(header_label, 0, col + 1)  # Top headers

            header_label = QLabel(header)
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
        rotation_amount_label.setAlignment(Qt.AlignRight)
        self.rotation_amount = QLineEdit("0.1")
        self.rotation_amount.setFixedWidth(self.INPUT_WIDTH)
        self.rotation_amount.setToolTip("Set the rotation angle in radians")
        rotation_amount_layout.addWidget(rotation_amount_label)
        rotation_amount_layout.addWidget(self.rotation_amount)
        rotation_layout.addLayout(rotation_amount_layout)
        rotation_group.setLayout(rotation_layout)

        return rotation_group

    def update_iterations(self):
        try:
            current_iterations = float(self.current_iterations_field.text())
            iterations_growth = float(self.iterations_growth_field.text())
            base_iterations = get_base_iterations(current_iterations, iterations_growth, self.settings.scale)
        except ValueError as error:
            logging.error(f"Invalid input in iterations field: {error}")
            return

        logging.info(
            f"Updating current iterations to {current_iterations}: base: {base_iterations}, growth: {iterations_growth}"
        )
        self.settings.base_iterations = base_iterations
        self.settings.iterations_growth = iterations_growth
        self.render_fractal()

    def update_radius(self):
        try:
            escape_radius = float(self.radius_field.text())
        except ValueError as error:
            logging.error(f"Invalid input in radius field: {escape_radius}: {error}")
            return

        logging.info(f"Updating radius to {escape_radius}")
        self.settings.escape_radius = escape_radius
        self.render_fractal()

    def update_uov(self):
        """Update u, o, v vectors from input fields and re-render."""
        try:
            logging.info(f"Updated u, o, v vectors from u={self.settings.u}, o={self.settings.o}, v={self.settings.v}")
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

    def update_controls_inputs(self):
        self.update_iterations_inputs()
        self.update_radius_inputs()

    def update_iterations_inputs(self):
        max_iterations = get_max_iterattions(
            self.settings.base_iterations, self.settings.iterations_growth, self.settings.scale
        )
        self.current_iterations_field.setText(str(int(max_iterations)))
        self.iterations_growth_field.setText(str(self.settings.iterations_growth))

    def update_radius_inputs(self):
        self.radius_field.setText(str(int(self.settings.escape_radius)))

    def update_colormap(self, colormap_name=None):
        """Update the colormap and re-render the fractal."""
        logging.info(f"Changing colormap to: {colormap_name}")
        if not colormap_name:
            colormap_name = self.settings.colormap
        else:
            self.settings.colormap = colormap_name
        self.update_interface_color()
        if self.current_escape_counts is not None:
            self.display_fractal(self.current_escape_counts, self.current_sampled_points)

    def update_interface_color(self):
        """Update the interface colors to match the colormap."""
        colormap = colormaps.get_cmap(self.settings.colormap)
        selector_color = colormap(0.5)

        r_text, g_text, b_text = rgba_to_rgb(colormap(0.5))
        r_bg, g_bg, b_bg = rgba_to_rgb(colormap(0.0))
        r_border, g_border, b_border = rgba_to_rgb(colormap(0.3))
        r_input_bg, g_input_bg, b_input_bg = rgba_to_rgb(colormap(0.1))

        color_variables = {
            "r_text": r_text,
            "g_text": g_text,
            "b_text": b_text,
            "r_bg": r_bg,
            "g_bg": g_bg,
            "b_bg": b_bg,
            "r_border": r_border,
            "g_border": g_border,
            "b_border": b_border,
            "r_input_bg": r_input_bg,
            "g_input_bg": g_input_bg,
            "b_input_bg": b_input_bg,
        }

        # Selector rectangle
        selector_color_rgb = rgba_to_rgb(selector_color)
        self.selector_color = QColor(*selector_color_rgb)

        # Set styles
        self.toggled_style = get_toggled_style().format(**color_variables)
        self.untoggled_style = get_untoggled_style().format(**color_variables)
        stylesheet = get_stylesheet().format(**color_variables)
        self.setStyleSheet(stylesheet)

        # set new styles to all toggle buttons
        for button in self.findChildren(QPushButton):
            if button.isCheckable():
                is_toggled = button.isChecked()
                button.setStyleSheet(self.toggled_style if is_toggled else self.untoggled_style)

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
            self.display_fractal(self.settings.escape_counts, self.settings.sampled_points)
            self.settings.escape_counts = None  # make historic settings changeable again
        else:
            logging.info("Rendering fractal...")
            self.dimensions = self.get_viewport_dimensions()
            self.worker = FractalWorker(self.settings, self.dimensions)
            self.worker.finished.connect(lambda result: self.display_fractal(*result))  # unpack parameters
            self.worker.start()

        self.update_controls_inputs()
        self.update_movement_inputs()
        self.update_uov_inputs()

    def get_viewport_dimensions(self):
        viewport_rect = self.graphics_view.viewport().rect()
        return (viewport_rect.width(), viewport_rect.height())

    def display_fractal(self, escape_counts, sampled_points):
        """Convert fractal data to an image with colormap and display it."""
        logging.info("Displaying fractal...")

        self.current_sampled_points = sampled_points
        self.current_escape_counts = escape_counts
        image_array = self.get_image_from_escape_counts(self.current_escape_counts)
        height, width, _ = image_array.shape
        q_image = QImage(image_array.data, width, height, 3 * width, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        self.graphics_scene.clear()
        self.basis_vector_pixmap = None
        self.graphics_scene.addPixmap(pixmap)

        history_settings = deepcopy(self.settings)
        history_settings.escape_counts = escape_counts
        history_settings.sampled_points = sampled_points
        self.history.append(history_settings)

        self.hande_basis_vector_display()

        logging.info("Fractal display updated.")

    def get_image_from_escape_counts(self, escape_counts):
        min_val = escape_counts.min()
        max_val = escape_counts.max()
        normalized = (escape_counts - min_val) / (max_val - min_val)
        colormap = colormaps.get_cmap(self.settings.colormap)
        image_array = (colormap(normalized)[:, :, :3] * 255).astype(np.uint8)
        return image_array

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
        try:
            mean = float(self.mean_random_input.text())
            std = float(self.std_random_input.text())
        except ValueError:
            logging.warning("Invalid input for randomization mean or std. Using defaults.")
            mean, std = self.RANDOM_MEAN_DEFAULT, self.RANDOM_STD_DEFAULT

        logging.info(f"Randomizing settings with mean={mean}, std={std}...")
        for col in range(3):
            if not self.col_toggled[col]:  # Column is not toggled
                new_values = np.random.normal(loc=mean, scale=std, size=6)
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
        try:
            mean = float(self.mean_perturb_input.text())
            std = float(self.std_perturb_input.text())
        except ValueError:
            logging.warning("Invalid input for perturbation mean or std. Using defaults.")
            mean, std = self.PERTURB_MEAN_DEFAULT, self.PERTURB_STD_DEFAULT

        logging.info(f"Perturbing settings with mean={mean}, std={std}...")
        for col in range(3):
            if not self.col_toggled[col]:  # Column is not toggled
                perturbation = np.random.normal(loc=mean, scale=std, size=6)
                for row in range(6):
                    if not self.row_toggled[row]:  # Row is not toggled
                        if col == 0:  # u
                            self.settings.u[row] += perturbation[row]
                        elif col == 1:  # o
                            self.settings.o[row] += perturbation[row]
                        elif col == 2:  # v
                            self.settings.v[row] += perturbation[row]
        self.render_fractal()

    def toggle_basis_vector_display(self):
        self.show_basis_vectors = not self.show_basis_vectors
        self.hande_basis_vector_display()

    def toggle_parameter_info_display(self):
        self.show_parameter_info = not self.show_parameter_info

    def hande_basis_vector_display(self):
        if self.basis_vector_pixmap:
            self.graphics_scene.removeItem(self.basis_vector_pixmap)
            self.basis_vector_pixmap = None

        if self.show_basis_vectors:
            pil_image = get_basis_projection_image(self.settings)
            data = pil_image.tobytes("raw", "RGBA")
            qimage = QImage(data, pil_image.width, pil_image.height, QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimage)

            self.basis_vector_pixmap = self.graphics_scene.addPixmap(pixmap)

            scene_rect = self.graphics_scene.sceneRect()
            image_width = pixmap.width()
            image_height = pixmap.height()

            center_x = scene_rect.x() + (scene_rect.width() - image_width) / 2
            center_y = scene_rect.y() + (scene_rect.height() - image_height) / 2

            self.basis_vector_pixmap.setPos(center_x, center_y)

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
        logging.info(
            f"Rotated around axis ({self.VECTOR_COMPONENT_NAMES[axis1_index]}, "
            f"{self.VECTOR_COMPONENT_NAMES[axis2_index]}) by {angle_radians:.4f} radians."
        )

    def move(self, direction):
        logging.info(f"Moving {direction}...")
        step = self.settings.scale * 0.1
        if direction == "W":
            self.settings.center = (
                self.settings.center[0],
                self.settings.center[1] - step,
            )
        elif direction == "S":
            self.settings.center = (
                self.settings.center[0],
                self.settings.center[1] + step,
            )
        elif direction == "A":
            self.settings.center = (
                self.settings.center[0] - step,
                self.settings.center[1],
            )
        elif direction == "D":
            self.settings.center = (
                self.settings.center[0] + step,
                self.settings.center[1],
            )
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
        if event.key() == Qt.Key_Escape:
            if self.start_pos:
                self.abort_rectangle_selection()
            else:
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
        elif event.key() == Qt.Key_F4:
            self.toggle_basis_vector_display()

    def save_settings(self):
        """Save the current fractal settings to a YAML file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Fractal Settings",
            DEFAULT_SAVE_PATH,
            "YAML Files (*.yaml);;All Files (*)",
            options=options,
        )
        if file_path:
            if not (file_path.endswith(".yaml") or file_path.endswith(".yml")):
                file_path += ".yaml"
            settings_dict = settings_to_dict(self.settings)
            with open(file_path, "w") as file:
                yaml.dump(settings_dict, file, default_flow_style=False)
            logging.info(f"Settings saved to {file_path}")

    def load_settings_dialogue(self):
        """Load fractal settings from a YAML file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Fractal Settings",
            DEFAULT_SAVE_PATH,
            "YAML Files (*.yaml);;All Files (*)",
            options=options,
        )
        self.load_settings(file_path)
        self.update_colormap()
        self.render_fractal()

    def load_settings(self, file_path):
        try:
            with open(file_path, "r") as file:
                settings_dict = yaml.safe_load(file)
                self.settings = dict_to_settings(settings_dict)
            logging.info(f"Settings loaded from {file_path}")
        except BaseException as error:
            logging.error(f"Could not load settings due to: {error}. Loading default.")
            self.settings = default_settings

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
        """Handle mouse events for rectangle selection and hover."""
        if event.type() == event.MouseButtonPress and event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
        elif event.type() == event.MouseMove and self.start_pos:
            end_pos = event.pos()
            self.draw_selection_rectangle(self.start_pos, end_pos)
        elif event.type() == event.MouseButtonRelease and event.button() == Qt.LeftButton:
            if self.selection_rect:
                self.handle_selection()
                self.start_pos = None

        if source == self.graphics_view.viewport() and event.type() == event.MouseMove:
            self.update_parameter_info_label(event)

        return super().eventFilter(source, event)

    def update_parameter_info_label(self, event):
        if not self.show_parameter_info:
            self.parameter_info_label.setVisible(False)
            return

        mouse_position = self.graphics_view.mapToScene(event.pos())
        y, x = int(mouse_position.x()), int(mouse_position.y())
        string = None

        if self.current_sampled_points is not None:
            shape = self.current_sampled_points.shape
            if x >= 0 and x < shape[0] and y >= 0 and y < shape[1]:
                string = self.get_parameter_info_string(x, y)

        if not string:
            self.parameter_info_label.setVisible(False)
            return
        else:
            self.parameter_info_label.setVisible(True)

        self.parameter_info_label.setText(string)

        font_metrics = QFontMetrics(self.parameter_info_label.font())
        text_width = font_metrics.horizontalAdvance(self.parameter_info_label.text())
        self.parameter_info_label.setFixedWidth(text_width)

        self.parameter_info_label.move(y + 10, x + 10)
        self.parameter_info_label.raise_()

    def get_parameter_info_string(self, x, y):
        parameters = self.current_sampled_points[x, y]

        c, z, p = parameters
        parameter_parts = [c.real, c.imag, z.real, z.imag, p.real, p.imag]

        parameters_max = self.current_sampled_points.max(axis=(0, 1))
        parameters_min = self.current_sampled_points.min(axis=(0, 1))
        parameters_delta = parameters_max - parameters_min
        c_delta, z_delta, p_delta = parameters_delta
        delta_parts = [
            c_delta.real,
            c_delta.imag,
            z_delta.real,
            z_delta.imag,
            p_delta.real,
            p_delta.imag,
        ]

        string = ""
        for parameter, delta, name in zip(parameter_parts, delta_parts, self.VECTOR_COMPONENT_NAMES):
            if delta == 0:
                precision = 1
            else:
                precision = -int(np.floor(np.log10(np.abs(delta))))
                precision += 2  # max 2 decimal difference between numbers on screen
            string += f"{name}: {parameter:+.{precision}f}    "
        return string

    def draw_selection_rectangle(self, start_pos, end_pos):
        """Draw the selection rectangle with aspect ratio constraint and boundary checks."""
        if self.selection_rect_visual:
            self.graphics_scene.removeItem(self.selection_rect_visual)

        # Map start and end positions to scene coordinates
        start_scene = self.graphics_view.mapToScene(start_pos)
        end_scene = self.graphics_view.mapToScene(end_pos)

        # Calculate dx and dy while constraining to the aspect ratio
        dx = end_scene.x() - start_scene.x()
        dy = end_scene.y() - start_scene.y()

        aspect_ratio = self.dimensions[0] / self.dimensions[1]
        if dx / self.dimensions[0] > dy / self.dimensions[1]:
            dy = dx / aspect_ratio
        else:
            dx = dy * aspect_ratio

        # correct for moving up/left instead of down/right
        if dx < 0:
            dx = -dx
        if dy < 0:
            dy = -dy

        # Constrain the rectangle to stay within the scene boundaries
        constrained_end_scene = QPointF(start_scene.x() + dx, start_scene.y() + dy)
        constrained_start_scene = QPointF(start_scene.x() - dx, start_scene.y() - dy)
        scene_rect = self.graphics_scene.sceneRect()
        if not scene_rect.contains(start_scene) or not scene_rect.contains(constrained_end_scene):
            return

        # Create and display the selection rectangle
        self.selection_rect = QRectF(constrained_start_scene, constrained_end_scene)
        self.selection_rect_visual = self.graphics_scene.addRect(self.selection_rect, QPen(self.selector_color, 2))

    def abort_rectangle_selection(self):
        self.start_pos = None
        self.selection_rect = None
        if self.selection_rect_visual:
            self.graphics_scene.removeItem(self.selection_rect_visual)
        self.selection_rect_visual = None

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

        resolution = self.dimensions
        cx_normalized = (cx / self.graphics_scene.sceneRect().width() - 0.5) * resolution[0] / resolution[1]
        cy_normalized = cy / self.graphics_scene.sceneRect().height() - 0.5

        self.settings.center = (
            self.settings.center[0] + cx_normalized * self.settings.scale,
            self.settings.center[1] + cy_normalized * self.settings.scale,
        )
        self.settings.scale *= min(width_pixels / resolution[0], height_pixels / resolution[1])

        self.render_fractal()
        self.selection_rect = None

    def export_fractal(self):
        """Export the fractal as an image at a specified resolution."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Fractal Image",
            DEFAULT_EXPORT_PATH,
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)",
            options=options,
        )

        if not file_path:
            return

        if not (file_path.endswith(".png") or file_path.endswith(".jpg")):
            file_path += ".png"

        resolution = self.dimensions
        resolution_dialog = QDialog(self)
        resolution_dialog.setWindowTitle("Specify Resolution")
        layout = QGridLayout()

        width_label = QLabel("Width:")
        layout.addWidget(width_label, 0, 0)
        width_input = QLineEdit(str(resolution[0]))
        layout.addWidget(width_input, 0, 1)

        height_label = QLabel("Height:")
        layout.addWidget(height_label, 1, 0)
        height_input = QLineEdit(str(resolution[1]))
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
        logging.info(f"Exporting fractal to {file_path} at resolution {resolution}...")
        self.worker = FractalWorker(self.settings, resolution)
        self.worker.finished.connect(lambda escape_counts: self.save_image(self.current_escape_counts, file_path))
        self.worker.start()

    def save_image(self, escape_counts, file_path):
        image_array = self.get_image_from_escape_counts(escape_counts)
        image = Image.fromarray(image_array, mode="RGB")
        image.save(file_path)
        logging.info(f"Fractal successfully exported to {file_path}.")


if __name__ == "__main__":
    args = parse_args()
    app = QApplication(sys.argv)
    main_window = FractalApp(args)
    main_window.show()
    sys.exit(app.exec())

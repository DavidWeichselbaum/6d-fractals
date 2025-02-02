from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget


class DebugTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Debug Object Tree")
        self.resize(600, 400)

        # Main layout
        main_layout = QVBoxLayout()

        # Instructions label
        self.instructions = QLabel("Hover over any widget to see its object tree.")
        self.instructions.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.instructions)

        # Tree widget to display object hierarchy
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Object Name", "Class"])
        main_layout.addWidget(self.tree_widget)

        # Central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Enable mouse tracking
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, source, event):
        if event.type() == event.MouseMove:
            widget = QApplication.instance().widgetAt(event.globalPos())
            if widget:
                self.update_tree(widget)
        return super().eventFilter(source, event)

    def update_tree(self, widget):
        """Update the tree widget with the hierarchy of the given widget."""
        self.tree_widget.clear()

        # Add the current widget and its hierarchy to the tree
        root_item = QTreeWidgetItem([widget.objectName() or "<unnamed>", widget.__class__.__name__])
        self.tree_widget.addTopLevelItem(root_item)

        # Add parent widgets
        self.add_parents_to_tree(widget, root_item)

        # Expand the tree
        self.tree_widget.expandAll()

    def add_parents_to_tree(self, widget, parent_item):
        """Recursively add parent widgets to the tree."""
        parent = widget.parentWidget()
        if parent:
            parent_item = QTreeWidgetItem([parent.objectName() or "<unnamed>", parent.__class__.__name__])
            self.tree_widget.insertTopLevelItem(0, parent_item)
            self.add_parents_to_tree(parent, parent_item)

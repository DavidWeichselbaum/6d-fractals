from PyQt5.QtGui import QFont


def get_font():
    return QFont("FreeSans", 12)


def get_stylesheet():
    """
    Returns the main stylesheet with placeholders for dynamic variables.
    """
    return """
     * {{
        background-color: rgb({r_bg}, {g_bg}, {b_bg});
        color: rgb({r_text}, {g_text}, {b_text});
    }}
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
    QToolTip {{
        background-color: rgb({r_bg}, {g_bg}, {b_bg});
        color: rgb({r_text}, {g_text}, {b_text});
        border: 1px solid rgb({r_border_group}, {g_border_group}, {b_border_group});
    }}
    QGroupBox {{
        color: rgb({r_text}, {g_text}, {b_text});
        background-color: rgb({r_bg}, {g_bg}, {b_bg});
        border: 1px solid rgb({r_border_group}, {g_border_group}, {b_border_group});
        border-radius: 5px;
        font-weight: bold;
        margin-top: 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top right;
        padding: 0 5px;
        background-color: rgb({r_bg}, {g_bg}, {b_bg});
        color: rgb({r_text}, {g_text}, {b_text});
    }}
    QFileDialog, QDialog {{
        background-color: rgb({r_bg}, {g_bg}, {b_bg});
        color: rgb({r_text}, {g_text}, {b_text});
        border: 1px solid rgb({r_border}, {g_border}, {b_border});
    }}
    QFileDialog QLineEdit, QDialog QLineEdit {{
        color: rgb({r_text}, {g_text}, {b_text});
        background-color: rgb({r_input_bg}, {g_input_bg}, {b_input_bg});
        border: 1px solid rgb({r_border}, {g_border}, {b_border});
    }}
    QFileDialog QPushButton, QDialog QPushButton {{
        color: rgb({r_text}, {g_text}, {b_text});
        background-color: rgb({r_bg}, {g_bg}, {b_bg});
        border: 1px solid rgb({r_border}, {g_border}, {b_border});
    }}
    QFileDialog QPushButton:hover, QDialog QPushButton:hover {{
        background-color: rgb({r_border}, {g_border}, {b_border});
        color: rgb({r_bg}, {g_bg}, {b_bg});
    }}
    QFileDialog QScrollBar, QDialog QScrollBar {{
        background-color: rgb({r_bg}, {g_bg}, {b_bg});
    }}
    QFileDialog QScrollBar::handle, QDialog QScrollBar::handle {{
        background-color: rgb({r_border}, {g_border}, {b_border});
    }}
    QFileDialog QTreeView, QDialog QTreeView {{
        background-color: rgb({r_bg}, {g_bg}, {b_bg});
        color: rgb({r_text}, {g_text}, {b_text});
        border: none;
    }}
    QFileDialog QTreeView::item, QDialog QTreeView::item {{
        background-color: rgb({r_bg}, {g_bg}, {b_bg});
        color: rgb({r_text}, {g_text}, {b_text});
    }}
    QFileDialog QListView, QDialog QListView {{
        background-color: rgb({r_bg}, {g_bg}, {b_bg});
        color: rgb({r_text}, {g_text}, {b_text});
    }}
    QFileDialog QHeaderView::section, QDialog QHeaderView::section {{
        background-color: rgb({r_border}, {g_border}, {b_border});
        color: rgb({r_text}, {g_text}, {b_text});
        border: 1px solid rgb({r_border}, {g_border}, {b_border});
    }}
    QMessageBox {{
        background-color: rgb({r_bg}, {g_bg}, {b_bg});
        color: rgb({r_text}, {g_text}, {b_text});
        border: 1px solid rgb({r_border}, {g_border}, {b_border});
    }}
    QMessageBox QPushButton {{
        color: rgb({r_text}, {g_text}, {b_text});
        background-color: rgb({r_bg}, {g_bg}, {b_bg});
        border: 1px solid rgb({r_border}, {g_border}, {b_border});
    }}
    QMessageBox QPushButton:hover {{
        background-color: rgb({r_border}, {g_border}, {b_border});
        color: rgb({r_bg}, {g_bg}, {b_bg});
    }}
    """

def get_toggled_style():
    """
    Returns the style for toggled elements.
    """
    return (
        "background-color: rgb({r_toggled_bg}, {g_toggled_bg}, {b_toggled_bg}); "
        "color: rgb({r_toggled_text}, {g_toggled_text}, {b_toggled_text}); font-weight: bold;"
    )

def get_untoggled_style():
    """
    Returns the style for untoggled elements.
    """
    return (
        "background-color: rgb({r_untoggled_bg}, {g_untoggled_bg}, {b_untoggled_bg}); "
        "color: rgb({r_untoggled_text}, {g_untoggled_text}, {b_untoggled_text}); font-weight: normal;"
    )

from PyQt5.QtGui import QFont


def get_font():
    return QFont("FreeSans", 12)


def get_stylesheet():
    return """
     * {{
        color: rgb({r_text}, {g_text}, {b_text});
        background-color: rgb({r_bg}, {g_bg}, {b_bg});
        border: 1px solid rgb({r_border}, {g_border}, {b_border});
    }}
    QGraphicsView {{
        border: none;
    }}
    QPushButton:hover {{
        background-color: rgb({r_border}, {g_border}, {b_border});
        color: rgb({r_bg}, {g_bg}, {b_bg});
    }}
    QScrollBar::handle {{
        background-color: rgb({r_border}, {g_border}, {b_border});
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{
        background-color: rgb({r_border}, {g_border}, {b_border});
    }}
    QLineEdit {{
        background-color: rgb({r_input_bg}, {g_input_bg}, {b_input_bg});
    }}
    QToolTip {{
        border: 1px solid rgb({r_text}, {g_text}, {b_text});
    }}
    QGroupBox {{
        border: 1px solid rgb({r_text}, {g_text}, {b_text});
        border-radius: 5px;
        font-weight: bold;
        margin-top: 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top right;
        padding: 0 5px;
    }}
    QLabel {{
        border: none;
    }}
    """

def get_untoggled_style():
    return (
        "color: rgb({r_text}, {g_text}, {b_text});"
        "background-color: rgb({r_bg}, {g_bg}, {b_bg}); "
    )

def get_toggled_style():
    return (
        "color: rgb({r_bg}, {g_bg}, {b_bg}); "
        "background-color: rgb({r_text}, {g_text}, {b_text});"
    )

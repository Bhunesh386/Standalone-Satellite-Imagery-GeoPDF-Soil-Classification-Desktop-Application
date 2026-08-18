"""
Modern Dark GIS Theme and Stylesheets for PyQt6 / PySide6 Desktop GUI.
"""

DARK_THEME_QSS = """
/* Global Window & Font */
QMainWindow, QDialog, QWidget {
    background-color: #0B0F19;
    color: #F1F5F9;
    font-family: "Segoe UI", "Inter", "Ubuntu", -apple-system, sans-serif;
    font-size: 13px;
}

/* Scrollbars */
QScrollBar:vertical {
    background: #111827;
    width: 10px;
    margin: 0px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #374151;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: #4B5563;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* Tab Bar */
QTabWidget::pane {
    border: 1px solid #1F2937;
    background-color: #111827;
    border-radius: 8px;
    padding: 6px;
}
QTabBar::tab {
    background: #1F2937;
    color: #9CA3AF;
    padding: 10px 20px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 600;
    font-size: 13px;
}
QTabBar::tab:selected {
    background: #0284C7;
    color: #FFFFFF;
}
QTabBar::tab:hover:!selected {
    background: #374151;
    color: #F3F4F6;
}

/* Group Boxes & Panels */
QGroupBox {
    background-color: #111827;
    border: 1px solid #1F2937;
    border-radius: 8px;
    margin-top: 24px;
    padding: 14px;
    font-weight: bold;
    color: #38BDF8;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    background-color: #111827;
    border-radius: 4px;
}

/* Push Buttons */
QPushButton {
    background-color: #1E293B;
    color: #F8FAFC;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #334155;
    border-color: #475569;
}
QPushButton:pressed {
    background-color: #0F172A;
}
QPushButton:disabled {
    background-color: #1E293B;
    color: #475569;
    border-color: #1E293B;
}

/* Primary Action Buttons */
QPushButton#primaryButton, QPushButton[primary="true"] {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284C7, stop:1 #0369A1);
    color: #FFFFFF;
    border: 1px solid #38BDF8;
    font-size: 14px;
    padding: 10px 22px;
}
QPushButton#primaryButton:hover, QPushButton[primary="true"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0369A1, stop:1 #075985);
    border-color: #7DD3FC;
}

/* Success Action Button */
QPushButton#successButton {
    background: #059669;
    color: #FFFFFF;
    border: 1px solid #10B981;
}
QPushButton#successButton:hover {
    background: #047857;
}

/* Input Fields & Combos */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #1E293B;
    color: #F8FAFC;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 10px;
    selection-background-color: #0284C7;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #38BDF8;
}
QComboBox QAbstractItemView {
    background-color: #1E293B;
    color: #F8FAFC;
    selection-background-color: #0284C7;
    border: 1px solid #334155;
}

/* Tables */
QTableWidget, QTableView {
    background-color: #111827;
    color: #F1F5F9;
    gridline-color: #1F2937;
    border: 1px solid #1F2937;
    border-radius: 6px;
    selection-background-color: #0369A1;
}
QHeaderView::section {
    background-color: #1F2937;
    color: #38BDF8;
    padding: 6px;
    border: 1px solid #111827;
    font-weight: bold;
}

/* Progress Bar */
QProgressBar {
    background-color: #1E293B;
    border: 1px solid #334155;
    border-radius: 6px;
    text-align: center;
    color: #FFFFFF;
    font-weight: bold;
    height: 22px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284C7, stop:1 #38BDF8);
    border-radius: 5px;
}

/* Checkboxes */
QCheckBox {
    color: #F1F5F9;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #475569;
    background-color: #1E293B;
}
QCheckBox::indicator:checked {
    background-color: #0284C7;
    border-color: #38BDF8;
}

/* Text Edit / Console */
QTextEdit, QPlainTextEdit {
    background-color: #090D16;
    color: #E2E8F0;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
    border: 1px solid #1F2937;
    border-radius: 6px;
    padding: 8px;
}

/* Labels */
QLabel#headerTitle {
    font-size: 18px;
    font-weight: bold;
    color: #FFFFFF;
}
QLabel#badge {
    background-color: #1E293B;
    color: #38BDF8;
    border: 1px solid #0284C7;
    border-radius: 4px;
    padding: 2px 8px;
    font-weight: 600;
    font-size: 11px;
}
QLabel#alertBanner {
    background-color: #451A03;
    color: #FDBA74;
    border: 1px solid #D97706;
    border-radius: 6px;
    padding: 10px;
    font-size: 12px;
}
"""

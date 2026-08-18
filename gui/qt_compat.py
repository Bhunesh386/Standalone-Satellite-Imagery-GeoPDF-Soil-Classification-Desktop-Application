"""
Qt Compatibility Bridge for PyQt6 and PySide6.
Prioritizes PyQt6 and provides unified imports across both Qt bindings.
"""

try:
    from PyQt6.QtCore import Qt, pyqtSignal as Signal, pyqtSlot as Slot, QThread, QSize, QRect, QPoint, QTimer
    from PyQt6.QtGui import QIcon, QPixmap, QImage, QColor, QFont, QPainter, QPen, QBrush, QDragEnterEvent, QDropEvent, QMouseEvent
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QDialog, QVBoxLayout, QHBoxLayout,
        QGridLayout, QFormLayout, QSplitter, QStackedWidget, QTabWidget,
        QLabel, QPushButton, QToolButton, QLineEdit, QTextEdit, QPlainTextEdit,
        QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QRadioButton, QSlider,
        QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
        QMessageBox, QGroupBox, QScrollArea, QFrame, QSizePolicy, QListWidget,
        QListWidgetItem
    )
    QT_LIB = "PyQt6"
except ImportError:
    from PySide6.QtCore import Qt, Signal, Slot, QThread, QSize, QRect, QPoint, QTimer
    from PySide6.QtGui import QIcon, QPixmap, QImage, QColor, QFont, QPainter, QPen, QBrush, QDragEnterEvent, QDropEvent, QMouseEvent
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QDialog, QVBoxLayout, QHBoxLayout,
        QGridLayout, QFormLayout, QSplitter, QStackedWidget, QTabWidget,
        QLabel, QPushButton, QToolButton, QLineEdit, QTextEdit, QPlainTextEdit,
        QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QRadioButton, QSlider,
        QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
        QMessageBox, QGroupBox, QScrollArea, QFrame, QSizePolicy, QListWidget,
        QListWidgetItem
    )
    QT_LIB = "PySide6"

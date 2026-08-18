"""
PySide6 / PyQt6 GUI Package for Soil Classification Desktop Application.
"""

from .main_window import SoilClassifierMainWindow
from .qt_compat import QApplication

__all__ = ["SoilClassifierMainWindow", "QApplication"]

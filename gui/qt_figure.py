"""
Custom Qt Matplotlib Figure Widget.
Renders matplotlib figures using FigureCanvasAgg and blits high-performance RGBA pixmaps
directly into Qt paint events, bypassing backend dependency mismatches.
"""

from typing import Optional, Callable
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .qt_compat import (
    QWidget, QImage, QPixmap, QPainter, QSize, QSizePolicy,
    QMouseEvent, Signal, Qt
)


class QtFigureWidget(QWidget):
    """Custom high-performance Qt widget for displaying Matplotlib figures."""

    mouseMoved = Signal(float, float) # (xdata, ydata) in axis coordinates

    def __init__(self, figure: Optional[Figure] = None, parent=None):
        super().__init__(parent)
        self.figure = figure if figure is not None else Figure(facecolor="#0F172A")
        self.canvas = FigureCanvasAgg(self.figure)
        self.pixmap: Optional[QPixmap] = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

    def set_figure(self, figure: Figure):
        self.figure = figure
        self.canvas = FigureCanvasAgg(self.figure)
        self.draw()

    def draw(self):
        """Renders the figure and updates internal pixmap."""
        w, h = max(self.width(), 100), max(self.height(), 100)
        dpi = 100
        self.figure.set_size_inches(w / dpi, h / dpi)
        self.figure.set_dpi(dpi)
        self.canvas.draw()
        
        rgba = np.asarray(self.canvas.buffer_rgba())
        h_buf, w_buf, _ = rgba.shape
        qimg = QImage(rgba.data, w_buf, h_buf, w_buf * 4, QImage.Format.Format_RGBA8888)
        self.pixmap = QPixmap.fromImage(qimg)
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.draw()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.pixmap:
            painter.drawPixmap(0, 0, self.pixmap)
        else:
            painter.fillRect(self.rect(), Qt.GlobalColor.black)

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.figure.axes:
            return
        ax = self.figure.axes[0]
        # Invert Qt y-coordinate for matplotlib
        x_pt = event.position().x()
        y_pt = self.height() - event.position().y()

        inv = ax.transData.inverted()
        try:
            xdata, ydata = inv.transform_point((x_pt, y_pt))
            self.mouseMoved.emit(float(xdata), float(ydata))
        except Exception:
            pass
        super().mouseMoveEvent(event)

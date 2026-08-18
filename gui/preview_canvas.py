"""
Interactive Viewport & Spatial Preview Canvas (Step 4).
Renders interactive geospatial imagery with pan, zoom, coordinate tracker,
and bounding box overlays.
"""

from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import matplotlib.patches as patches

from .qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QGroupBox, Qt
)
from .qt_figure import QtFigureWidget


class InteractivePreviewCanvas(QWidget):
    """Step 4 UI: Interactive preview canvas with coordinate tracking and layer toggles."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_data: Optional[np.ndarray] = None
        self.bounds = None
        self.neatline = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        header = QLabel("Step 4: Interactive Viewport & Spatial Preview")
        header.setObjectName("headerTitle")
        sub = QLabel("Pan, zoom, inspect raster channels, and verify geospatial bounds prior to full-scene inference.")
        sub.setStyleSheet("color: #94A3B8;")

        layout.addWidget(header)
        layout.addWidget(sub)

        # Controls & Coordinate Bar
        ctrl_bar = QHBoxLayout()
        self.btn_reset_view = QPushButton("🔍 Reset View")
        self.btn_reset_view.clicked.connect(self._reset_view)

        self.cb_show_neatline = QCheckBox("Show Geographic Neatline Viewport")
        self.cb_show_neatline.setChecked(True)
        self.cb_show_neatline.toggled.connect(self._redraw)

        self.lbl_coords = QLabel("Cursor: (X: —, Y: —) | Ground: (E: —, N: —)")
        self.lbl_coords.setStyleSheet("color: #38BDF8; font-family: monospace; font-weight: bold;")

        ctrl_bar.addWidget(self.btn_reset_view)
        ctrl_bar.addWidget(self.cb_show_neatline)
        ctrl_bar.addStretch()
        ctrl_bar.addWidget(self.lbl_coords)

        layout.addLayout(ctrl_bar)

        # Matplotlib Figure Canvas Widget
        self.figure = Figure(facecolor="#0F172A", edgecolor="#1F2937")
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#0F172A")
        self.ax.tick_params(colors="#94A3B8")

        self.canvas_widget = QtFigureWidget(self.figure, self)
        self.canvas_widget.mouseMoved.connect(self._on_mouse_move)

        layout.addWidget(self.canvas_widget, stretch=1)
        self._show_placeholder()

    def _show_placeholder(self):
        self.ax.clear()
        self.ax.set_facecolor("#0F172A")
        self.ax.text(
            0.5, 0.5, "No Imagery Loaded\nSelect a file in Step 1 to preview",
            color="#64748B", fontsize=13, ha="center", va="center",
            transform=self.ax.transAxes
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas_widget.draw()

    def set_preview_image(self, img_array: np.ndarray, bounds=None, neatline=None):
        """
        Updates canvas with RGB image array (H, W, 3) or (3, H, W) or (C, H, W).
        """
        if img_array.ndim == 3 and img_array.shape[0] in [1, 3, 4] and img_array.shape[0] < img_array.shape[1]:
            # Convert CHW to HWC
            if img_array.shape[0] >= 3:
                rgb = np.transpose(img_array[:3], (1, 2, 0))
            else:
                rgb = np.repeat(np.transpose(img_array[:1], (1, 2, 0)), 3, axis=2)
        else:
            rgb = img_array

        # Scale to uint8 if float
        if rgb.dtype != np.uint8:
            rmin, rmax = np.min(rgb), np.max(rgb)
            if rmax > rmin:
                rgb = ((rgb - rmin) / (rmax - rmin) * 255.0).astype(np.uint8)
            else:
                rgb = (rgb * 255.0).astype(np.uint8)

        self.image_data = rgb
        self.bounds = bounds
        self.neatline = neatline
        self._redraw()

    def _redraw(self):
        if self.image_data is None:
            self._show_placeholder()
            return

        self.ax.clear()
        self.ax.set_facecolor("#0F172A")
        h, w, _ = self.image_data.shape

        self.ax.imshow(self.image_data)
        self.ax.set_title(f"Spatial Viewport ({w} x {h} px)", color="#F8FAFC", fontsize=11)
        self.ax.tick_params(colors="#94A3B8")

        # Draw neatline bounding rectangle if requested
        if self.cb_show_neatline.isChecked() and self.neatline:
            x0, y0, x1, y1 = self.neatline
            rect = patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=2, edgecolor="#00B4D8", facecolor="none",
                linestyle="--", label="Neatline Viewport"
            )
            self.ax.add_patch(rect)

        self.canvas_widget.draw()

    def _reset_view(self):
        if self.image_data is not None:
            h, w, _ = self.image_data.shape
            self.ax.set_xlim(0, w)
            self.ax.set_ylim(h, 0)
            self.canvas_widget.draw()

    def _on_mouse_move(self, xdata: float, ydata: float):
        if self.image_data is None:
            return

        px = int(round(xdata))
        py = int(round(ydata))
        h, w, _ = self.image_data.shape

        if 0 <= px < w and 0 <= py < h:
            if self.bounds:
                min_x, min_y, max_x, max_y = self.bounds
                gx = min_x + (px / max(w, 1)) * (max_x - min_x)
                gy = max_y - (py / max(h, 1)) * (max_y - min_y)
                self.lbl_coords.setText(f"Pixel: ({px}, {py}) | Ground: ({gx:.1f} E, {gy:.1f} N)")
            else:
                self.lbl_coords.setText(f"Pixel: ({px}, {py})")

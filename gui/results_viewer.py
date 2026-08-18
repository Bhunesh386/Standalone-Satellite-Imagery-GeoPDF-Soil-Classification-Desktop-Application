"""
Results Viewer & Geospatial Analytics Dashboard (Step 5).
Displays side-by-side synchronized input vs classification vs confidence maps,
soil class area distribution charts, metrics audit tables, and export action shortcuts.
"""

import os
import subprocess
import platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from PIL import Image

from .qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
    QGroupBox, QSplitter, QProgressBar, QTextEdit, QScrollArea,
    Signal, Qt
)
from .qt_figure import QtFigureWidget
from reporting.legend import SoilLegendManager


class ResultsViewerPanel(QWidget):
    """Step 5 UI: Comprehensive results visualization and analytics dashboard."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result_data = None
        self.legend_mgr = SoilLegendManager()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header
        header = QLabel("Step 5: Classification Results & Geospatial Analytics")
        header.setObjectName("headerTitle")
        sub = QLabel("Explore the pixel-wise classified soil map, model confidence heatmap, and land-use area statistics.")
        sub.setStyleSheet("color: #94A3B8;")

        layout.addWidget(header)
        layout.addWidget(sub)

        # Action Toolbar (Open Folder, Open PDF, etc.)
        toolbar = QHBoxLayout()
        self.btn_open_folder = QPushButton("📁 Open Output Directory")
        self.btn_open_folder.setObjectName("primaryButton")
        self.btn_open_folder.clicked.connect(self._open_output_folder)

        self.btn_open_pdf = QPushButton("📄 View GeoPDF Report")
        self.btn_open_pdf.clicked.connect(self._open_pdf_report)

        self.btn_export_png = QPushButton("🖼️ Save High-Res PNG")
        self.btn_export_png.clicked.connect(self._save_highres_png)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #10B981; font-weight: bold;")

        toolbar.addWidget(self.btn_open_folder)
        toolbar.addWidget(self.btn_open_pdf)
        toolbar.addWidget(self.btn_export_png)
        toolbar.addStretch()
        toolbar.addWidget(self.lbl_status)

        layout.addLayout(toolbar)

        # Main Splitter: Maps on Left / Center, Statistics on Right
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left / Center: Synchronized Multi-View Tabs
        map_widget = QWidget()
        map_layout = QVBoxLayout(map_widget)
        map_layout.setContentsMargins(0, 0, 0, 0)

        self.map_tabs = QTabWidget()
        
        # 1. Side-by-Side Comparison Figure
        self.fig_compare = Figure(figsize=(10, 5), facecolor="#0F172A")
        self.canvas_compare = QtFigureWidget(self.fig_compare, self)
        self.map_tabs.addTab(self.canvas_compare, "Side-by-Side Tri-View")

        # 2. Classified Soil Map Full Figure
        self.fig_class = Figure(figsize=(8, 6), facecolor="#0F172A")
        self.canvas_class = QtFigureWidget(self.fig_class, self)
        self.map_tabs.addTab(self.canvas_class, "Classified Soil Map")

        # 3. Confidence Heatmap Full Figure
        self.fig_conf = Figure(figsize=(8, 6), facecolor="#0F172A")
        self.canvas_conf = QtFigureWidget(self.fig_conf, self)
        self.map_tabs.addTab(self.canvas_conf, "Inference Confidence Heatmap")

        map_layout.addWidget(self.map_tabs)
        splitter.addWidget(map_widget)

        # Right: Area Breakdown & Audit Table
        stats_widget = QWidget()
        stats_widget.setMinimumWidth(380)
        stats_layout = QVBoxLayout(stats_widget)
        stats_layout.setContentsMargins(0, 0, 0, 0)

        # Soil Distribution Table
        dist_group = QGroupBox("Soil Classification Distribution & Area")
        dist_layout = QVBoxLayout(dist_group)

        self.stats_table = QTableWidget(self)
        self.stats_table.setColumnCount(4)
        self.stats_table.setHorizontalHeaderLabels(["Soil Class", "Coverage %", "Area (km²)", "Area (ha)"])
        self.stats_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        dist_layout.addWidget(self.stats_table)

        stats_layout.addWidget(dist_group)

        # Metrics Card
        metrics_group = QGroupBox("Model Accuracy & Quality Metrics")
        met_layout = QVBoxLayout(metrics_group)

        self.lbl_mean_conf = QLabel("Mean Confidence: —")
        self.lbl_mean_conf.setStyleSheet("font-size: 13px; font-weight: bold; color: #38BDF8;")
        self.lbl_low_conf = QLabel("Low Confidence (<40%): —")
        self.lbl_total_area = QLabel("Total Scene Area: —")
        self.lbl_runtime = QLabel("Execution Time: —")

        met_layout.addWidget(self.lbl_mean_conf)
        met_layout.addWidget(self.lbl_low_conf)
        met_layout.addWidget(self.lbl_total_area)
        met_layout.addWidget(self.lbl_runtime)

        stats_layout.addWidget(metrics_group)
        splitter.addWidget(stats_widget)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

    def display_results(self, result_bundle: dict):
        """Renders results from execution worker."""
        self.result_data = result_bundle
        class_map = result_bundle["class_map"]
        conf_map = result_bundle["confidence_map"]
        raster = result_bundle["input_raster"]
        report = result_bundle.get("report_data")
        elapsed = result_bundle.get("total_elapsed_s", 0.0)

        # 1. Colorize classification map
        rgb_class = self.legend_mgr.colorize_class_map(class_map)

        # 2. Extract displayable optical RGB from input raster
        if raster.ndim == 3 and raster.shape[0] >= 3:
            r = raster[0].astype(np.float32)
            g = raster[1].astype(np.float32)
            b = raster[2].astype(np.float32)
            # Min-max scale
            def norm_b(band):
                p2, p98 = np.percentile(band, [2, 98])
                if p98 > p2:
                    return np.clip((band - p2) / (p98 - p2), 0, 1)
                return np.clip(band / 255.0, 0, 1)
            opt_rgb = np.stack([norm_b(r), norm_b(g), norm_b(b)], axis=-1)
        elif raster.ndim == 3 and raster.shape[0] == 1:
            band = raster[0].astype(np.float32)
            p2, p98 = np.percentile(band, [2, 98])
            nb = np.clip((band - p2) / max(p98 - p2, 1e-3), 0, 1)
            opt_rgb = np.stack([nb, nb, nb], axis=-1)
        else:
            opt_rgb = np.zeros_like(rgb_class)

        # 3. Draw Side-by-Side Comparison
        self.fig_compare.clear()
        ax1 = self.fig_compare.add_subplot(131)
        ax2 = self.fig_compare.add_subplot(132)
        ax3 = self.fig_compare.add_subplot(133)

        ax1.imshow(opt_rgb)
        ax1.set_title("Optical Satellite Imagery", color="#F8FAFC", fontsize=10)
        ax1.axis("off")

        ax2.imshow(rgb_class)
        ax2.set_title("Classified Soil Map", color="#F8FAFC", fontsize=10)
        ax2.axis("off")

        im3 = ax3.imshow(conf_map, cmap="viridis", vmin=0.0, vmax=1.0)
        ax3.set_title("Inference Confidence", color="#F8FAFC", fontsize=10)
        ax3.axis("off")

        self.fig_compare.tight_layout()
        self.canvas_compare.draw()

        # 4. Draw Full Classified View
        self.fig_class.clear()
        ax_c = self.fig_class.add_subplot(111)
        ax_c.imshow(rgb_class)
        ax_c.set_title("Per-Pixel Soil Classification", color="#F8FAFC", fontsize=12)
        ax_c.axis("off")
        self.fig_class.tight_layout()
        self.canvas_class.draw()

        # 5. Draw Full Confidence View
        self.fig_conf.clear()
        ax_conf = self.fig_conf.add_subplot(111)
        im_cf = ax_conf.imshow(conf_map, cmap="turbo", vmin=0.0, vmax=1.0)
        cbar = self.fig_conf.colorbar(im_cf, ax=ax_conf, orientation="horizontal", pad=0.05, shrink=0.7)
        cbar.set_label("Confidence Score (0.0 = Uncertain, 1.0 = High Certainty)", color="#CBD5E1")
        cbar.ax.tick_params(colors="#CBD5E1")
        ax_conf.set_title("Confidence Calibration Heatmap", color="#F8FAFC", fontsize=12)
        ax_conf.axis("off")
        self.fig_conf.tight_layout()
        self.canvas_conf.draw()

        # 6. Populate Statistics Table & Metric Cards
        if report:
            soil_dist = report.get("soil_class_distribution", [])
            self.stats_table.setRowCount(len(soil_dist))

            for row, item in enumerate(soil_dist):
                name_item = QTableWidgetItem(item["class_name"])
                pct_item = QTableWidgetItem(f"{item['percentage_of_scene']:.2f}%")
                km2_item = QTableWidgetItem(f"{item['area_sq_km']:.3f}")
                ha_item = QTableWidgetItem(f"{item['area_hectares']:.2f}")

                for itm in [name_item, pct_item, km2_item, ha_item]:
                    itm.setFlags(itm.flags() ^ Qt.ItemFlag.ItemIsEditable)

                self.stats_table.setItem(row, 0, name_item)
                self.stats_table.setItem(row, 1, pct_item)
                self.stats_table.setItem(row, 2, km2_item)
                self.stats_table.setItem(row, 3, ha_item)

            conf_ast = report.get("confidence_assessment", {})
            self.lbl_mean_conf.setText(f"Mean Confidence: {conf_ast.get('mean_confidence', 0)*100:.1f}%")
            self.lbl_low_conf.setText(f"Low Confidence Pixels: {conf_ast.get('low_confidence_percentage', 0)}%")

            grid_cfg = report.get("processing_configuration", {})
            self.lbl_total_area.setText(f"Total Scene Area: {grid_cfg.get('total_scene_area_km2', 0):.2f} km²")
            self.lbl_runtime.setText(f"Pipeline Runtime: {elapsed:.2f}s")

        self.lbl_status.setText("Job Finished Successfully ✅")

    def _open_output_folder(self):
        if not self.result_data:
            return
        saved_files = self.result_data.get("saved_files", {})
        any_file = list(saved_files.values())[0] if saved_files else "output"
        folder = os.path.dirname(os.path.abspath(any_file))

        if platform.system() == "Windows":
            os.startfile(folder)
        elif platform.system() == "Darwin":
            subprocess.run(["open", folder])
        else:
            subprocess.run(["xdg-open", folder])

    def _open_pdf_report(self):
        if not self.result_data:
            return
        pdf_path = self.result_data.get("saved_files", {}).get("geopdf")
        if pdf_path and os.path.exists(pdf_path):
            if platform.system() == "Windows":
                os.startfile(pdf_path)
            elif platform.system() == "Darwin":
                subprocess.run(["open", pdf_path])
            else:
                subprocess.run(["xdg-open", pdf_path])

    def _save_highres_png(self):
        if not self.result_data:
            return
        out_dir = os.path.dirname(list(self.result_data.get("saved_files", {}).values())[0])
        path = os.path.join(out_dir, "high_res_classified_view.png")
        self.fig_compare.savefig(path, dpi=300, facecolor="#0F172A")
        self.lbl_status.setText(f"Saved PNG to: {os.path.basename(path)} ✅")

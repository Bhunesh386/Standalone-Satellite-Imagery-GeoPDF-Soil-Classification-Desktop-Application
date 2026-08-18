"""
Processing and Export Configuration Panel (Step 3).
Configures target CRS, ground resolution, PDF DPI, sliding window tiling parameters,
and output export artifacts.
"""

import os
from .qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox,
    QGridLayout, QFileDialog, Signal
)


class ConfigPanel(QWidget):
    """Step 3 UI: Processing parameters and export selection."""

    configChanged = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        header = QLabel("Step 3: Processing Grid & Export Options")
        header.setObjectName("headerTitle")
        sub = QLabel("Configure ground resolution, target coordinate reference system, tiling parameters, and output artifacts.")
        sub.setStyleSheet("color: #94A3B8;")

        layout.addWidget(header)
        layout.addWidget(sub)

        # 1. Geospatial Grid & CRS Group
        grid_group = QGroupBox("Geospatial Grid & Resolution")
        grid_layout = QGridLayout(grid_group)
        grid_layout.setSpacing(10)

        grid_layout.addWidget(QLabel("Target CRS:"), 0, 0)
        self.combo_crs = QComboBox(self)
        self.combo_crs.addItems([
            "Auto-Detect UTM (Recommended)",
            "EPSG:32643 (WGS 84 / UTM Zone 43N)",
            "EPSG:4326 (WGS 84 Geographic)",
            "EPSG:3857 (WGS 84 / Pseudo-Mercator)",
            "Custom EPSG"
        ])
        grid_layout.addWidget(self.combo_crs, 0, 1)

        grid_layout.addWidget(QLabel("Ground Resolution (m/px):"), 0, 2)
        self.spin_res = QDoubleSpinBox(self)
        self.spin_res.setRange(0.5, 500.0)
        self.spin_res.setValue(10.0)
        self.spin_res.setSingleStep(1.0)
        grid_layout.addWidget(self.spin_res, 0, 3)

        grid_layout.addWidget(QLabel("PDF Rendering DPI:"), 1, 0)
        self.spin_dpi = QSpinBox(self)
        self.spin_dpi.setRange(72, 600)
        self.spin_dpi.setValue(300)
        self.spin_dpi.setSingleStep(50)
        grid_layout.addWidget(self.spin_dpi, 1, 1)

        layout.addWidget(grid_group)

        # 2. Tiling & Inference Optimization Group
        tile_group = QGroupBox("Sliding-Window Tiling & Blending Engine")
        tile_layout = QGridLayout(tile_group)
        tile_layout.setSpacing(10)

        tile_layout.addWidget(QLabel("Tile Window Size:"), 0, 0)
        self.combo_tile_size = QComboBox(self)
        self.combo_tile_size.addItems(["512 x 512 (Recommended)", "256 x 256 (Fast)", "1024 x 1024 (High-VRAM)"])
        tile_layout.addWidget(self.combo_tile_size, 0, 1)

        tile_layout.addWidget(QLabel("Tile Overlap (Margin):"), 0, 2)
        self.combo_overlap = QComboBox(self)
        self.combo_overlap.addItems(["32 px (Recommended)", "16 px (Fast)", "64 px (Ultra-Smooth)"])
        tile_layout.addWidget(self.combo_overlap, 0, 3)

        tile_layout.addWidget(QLabel("Boundary Blend Filter:"), 1, 0)
        self.combo_blend = QComboBox(self)
        self.combo_blend.addItems(["Cosine / Hann (Recommended)", "Gaussian Bell", "Linear Pyramid"])
        tile_layout.addWidget(self.combo_blend, 1, 1)

        layout.addWidget(tile_group)

        # 3. Export Artifacts Group
        export_group = QGroupBox("Export Artifact Packages")
        exp_layout = QGridLayout(export_group)
        exp_layout.setSpacing(10)

        self.cb_tif = QCheckBox("Classified Soil Map GeoTIFF (.tif with Color Table)")
        self.cb_tif.setChecked(True)
        self.cb_conf = QCheckBox("Pixel Confidence Map GeoTIFF (.tif Float32)")
        self.cb_conf.setChecked(True)
        self.cb_pdf = QCheckBox("Georeferenced Soil Report GeoPDF (.pdf with Legend)")
        self.cb_pdf.setChecked(True)
        self.cb_report = QCheckBox("Classification Audit Report (.json Metrics)")
        self.cb_report.setChecked(True)

        exp_layout.addWidget(self.cb_tif, 0, 0)
        exp_layout.addWidget(self.cb_conf, 0, 1)
        exp_layout.addWidget(self.cb_pdf, 1, 0)
        exp_layout.addWidget(self.cb_report, 1, 1)

        # Output Folder Selection
        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("Output Directory:"))
        self.lbl_out_dir = QLabel(os.path.abspath("output"))
        self.lbl_out_dir.setStyleSheet("color: #38BDF8; font-weight: bold;")
        self.btn_browse_out = QPushButton("Change Folder...")
        self.btn_browse_out.clicked.connect(self._on_browse_output)

        out_layout.addWidget(self.lbl_out_dir)
        out_layout.addWidget(self.btn_browse_out)
        exp_layout.addLayout(out_layout, 2, 0, 1, 2)

        layout.addWidget(export_group)
        layout.addStretch()

    def _on_browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.lbl_out_dir.text())
        if folder:
            self.lbl_out_dir.setText(folder)
            self.configChanged.emit(self.get_config())

    def get_config(self) -> dict:
        crs_text = self.combo_crs.currentText()
        if "32643" in crs_text or "Auto" in crs_text:
            target_crs = "EPSG:32643"
        elif "4326" in crs_text:
            target_crs = "EPSG:4326"
        elif "3857" in crs_text:
            target_crs = "EPSG:3857"
        else:
            target_crs = "EPSG:32643"

        tile_size = 512
        if "256" in self.combo_tile_size.currentText():
            tile_size = 256
        elif "1024" in self.combo_tile_size.currentText():
            tile_size = 1024

        overlap = 32
        if "16" in self.combo_overlap.currentText():
            overlap = 16
        elif "64" in self.combo_overlap.currentText():
            overlap = 64

        blend = "cosine"
        if "Gaussian" in self.combo_blend.currentText():
            blend = "gaussian"
        elif "Linear" in self.combo_blend.currentText():
            blend = "linear"

        return {
            "target_crs": target_crs,
            "resolution_m": self.spin_res.value(),
            "pdf_dpi": self.spin_dpi.value(),
            "tile_size": tile_size,
            "tile_overlap": overlap,
            "blend_method": blend,
            "export_geotiff": self.cb_tif.isChecked(),
            "export_confidence": self.cb_conf.isChecked(),
            "export_geopdf": self.cb_pdf.isChecked(),
            "export_report": self.cb_report.isChecked(),
            "output_dir": self.lbl_out_dir.text()
        }

"""
GeoPDF Inspector and Dynamic Layer-Role Matrix Panel (Step 2).
Provides multi-page visual thumbnail selector, layer table with assignable roles,
RGB fallback warning banner, and processing mode selector (Mode A / B / C).
"""

import os
from .qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QListWidget, QListWidgetItem, QIcon, QPixmap, QImage,
    QGroupBox, QFrame, QRadioButton, Signal, Qt, QSize
)
from ingestion.inspect_geopdf import GeoPDFInspector
from ingestion.render_geopdf import PDFRenderer


class LayerMatrixPanel(QWidget):
    """Step 2 UI: Multi-page selector, layer-role matrix, and processing modes."""

    configUpdated = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_file_path = None
        self.doc_info = None
        self.selected_page_index = 0
        self.layer_roles = {} # layer_id -> role string
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header
        header = QLabel("Step 2: GeoPDF Inspector & Layer-Role Matrix")
        header.setObjectName("headerTitle")
        sub = QLabel("Configure multi-page selection, inspect PDF layers, assign layer roles, and choose processing mode.")
        sub.setStyleSheet("color: #94A3B8;")

        layout.addWidget(header)
        layout.addWidget(sub)

        # RGB Warning Alert Banner (Hidden by default, shown for RGB PDFs)
        self.alert_banner = QLabel(self)
        self.alert_banner.setObjectName("alertBanner")
        self.alert_banner.setWordWrap(True)
        self.alert_banner.setVisible(False)
        layout.addWidget(self.alert_banner)

        # Processing Mode Selection Group
        mode_group = QGroupBox("GeoPDF Processing Mode")
        mode_layout = QHBoxLayout(mode_group)

        self.rb_mode_c = QRadioButton("Mode C (Hybrid - Recommended): Raster Model Input + Vector Exclusion Masks")
        self.rb_mode_a = QRadioButton("Mode A (Raster): Viewport Neatline Render")
        self.rb_mode_b = QRadioButton("Mode B (Vector): Vector Layer Rasterization")
        self.rb_mode_c.setChecked(True)

        mode_layout.addWidget(self.rb_mode_c)
        mode_layout.addWidget(self.rb_mode_a)
        mode_layout.addWidget(self.rb_mode_b)
        layout.addWidget(mode_group)

        # Main Split Content: Page Selector (Left) and Layer Matrix Table (Right)
        content_layout = QHBoxLayout()

        # Left: Page Selector with Thumbnails
        page_group = QGroupBox("Page Selection")
        page_group.setMaximumWidth(220)
        page_layout = QVBoxLayout(page_group)

        self.page_list = QListWidget(self)
        self.page_list.setIconSize(QSize(100, 100))
        self.page_list.currentRowChanged.connect(self._on_page_changed)
        page_layout.addWidget(self.page_list)
        content_layout.addWidget(page_group)

        # Right: Layer Matrix Table
        layer_group = QGroupBox("Layer Hierarchy & Role Assignment")
        layer_layout = QVBoxLayout(layer_group)

        self.layer_table = QTableWidget(self)
        self.layer_table.setColumnCount(4)
        self.layer_table.setHorizontalHeaderLabels(["Layer Name", "Type", "Status", "Assigned Role"])
        self.layer_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.layer_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        layer_layout.addWidget(self.layer_table)

        content_layout.addWidget(layer_group)
        layout.addLayout(content_layout)

    def set_file_info(self, file_info: dict):
        """Populates layer panel when a new file is loaded."""
        self.current_file_path = file_info.get("file_path")
        fmt = file_info.get("format")

        if fmt == "PDF":
            self._load_pdf_details()
        elif fmt == "GEOTIFF":
            self._load_geotiff_details(file_info)

    def _load_pdf_details(self):
        inspector = GeoPDFInspector(self.current_file_path)
        self.doc_info = inspector.inspect_document()

        # Show spectral warning if RGB-only
        warn = self.doc_info.get("spectral_warning")
        if warn and warn.get("is_rgb_only"):
            self.alert_banner.setText(f"ℹ️ <b>{warn['title']}</b>: {warn['message']}")
            self.alert_banner.setVisible(True)
        else:
            self.alert_banner.setVisible(False)

        # Render thumbnails
        self.page_list.clear()
        renderer = PDFRenderer(self.current_file_path)
        thumbnails = renderer.generate_thumbnails(max_size=120)

        for t in thumbnails:
            pil_img = t["image"]
            # Convert PIL image to QPixmap
            img_bytes = pil_img.tobytes("raw", "RGB")
            qimg = QImage(img_bytes, pil_img.width, pil_img.height, pil_img.width * 3, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            item = QListWidgetItem(f"Page {t['page_number']}\n({pil_img.width}x{pil_img.height})")
            item.setIcon(QIcon(pixmap))
            self.page_list.addItem(item)

        if self.page_list.count() > 0:
            self.page_list.setCurrentRow(0)

        # Populate Layer Table
        layers = self.doc_info.get("layers", [])
        self._populate_layer_table(layers)

    def _load_geotiff_details(self, file_info: dict):
        self.alert_banner.setVisible(False)
        self.page_list.clear()
        item = QListWidgetItem("GeoTIFF Scene\n(Single View)")
        self.page_list.addItem(item)
        self.page_list.setCurrentRow(0)

        channels = file_info.get("channels", 1)
        layers = []
        band_names = ["Red (Band 4)", "Green (Band 3)", "Blue (Band 2)", "Near-Infrared (Band 8)", "SWIR 1", "SWIR 2"]
        for i in range(channels):
            b_name = band_names[i] if i < len(band_names) else f"Spectral Band {i+1}"
            layers.append({
                "id": i + 1,
                "name": b_name,
                "type": "Raster Band",
                "visible": True,
                "role": "Model Input"
            })
        self._populate_layer_table(layers)

    def _populate_layer_table(self, layers: list):
        self.layer_table.setRowCount(len(layers))
        self.layer_roles.clear()

        roles = ["Model Input", "Exclusion Mask", "Reference Label", "Ignore"]

        for row, layer in enumerate(layers):
            # Name
            name_item = QTableWidgetItem(layer.get("name", f"Layer {row+1}"))
            name_item.setFlags(name_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.layer_table.setItem(row, 0, name_item)

            # Type
            type_item = QTableWidgetItem(layer.get("type", "Layer"))
            type_item.setFlags(type_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.layer_table.setItem(row, 1, type_item)

            # Status
            status_item = QTableWidgetItem("Active" if layer.get("visible", True) else "Hidden")
            status_item.setFlags(status_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.layer_table.setItem(row, 2, status_item)

            # Role Combo Box
            combo = QComboBox(self)
            combo.addItems(roles)
            default_role = layer.get("role", "Model Input")
            combo.setCurrentText(default_role)

            layer_id = layer.get("id", row)
            self.layer_roles[layer_id] = default_role

            def make_handler(lid, cb):
                return lambda: self._on_role_changed(lid, cb.currentText())

            combo.currentIndexChanged.connect(make_handler(layer_id, combo))
            self.layer_table.setCellWidget(row, 3, combo)

    def _on_role_changed(self, layer_id, role_text):
        self.layer_roles[layer_id] = role_text
        self.configUpdated.emit(self.get_config())

    def _on_page_changed(self, row: int):
        if row >= 0:
            self.selected_page_index = row
            self.configUpdated.emit(self.get_config())

    def get_config(self) -> dict:
        mode = "hybrid"
        if self.rb_mode_a.isChecked():
            mode = "raster"
        elif self.rb_mode_b.isChecked():
            mode = "vector"

        return {
            "page_index": self.selected_page_index,
            "processing_mode": mode,
            "layer_roles": self.layer_roles
        }

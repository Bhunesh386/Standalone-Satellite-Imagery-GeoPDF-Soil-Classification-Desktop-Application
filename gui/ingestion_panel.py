"""
Ingestion Panel Component (Step 1).
Provides drag-and-drop dropzone, file selector dialog, sample loaders,
and validation badges for GeoTIFF and GeoPDF files.
"""

import os
from .qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QFrame, QGroupBox, QGridLayout, Signal, Qt,
    QDragEnterEvent, QDropEvent
)
from ingestion.detect_format import detect_file_format


class DropZoneWidget(QFrame):
    """Interactive drag and drop zone with hover highlights."""

    fileDropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(150)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #0284C7;
                border-radius: 12px;
                background-color: #0F172A;
            }
            QFrame:hover {
                border-color: #38BDF8;
                background-color: #1E293B;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.icon_label = QLabel("📥", self)
        self.icon_label.setStyleSheet("font-size: 32px; background: transparent;")
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.text_label = QLabel("Drag & Drop GeoTIFF (.tif) or GeoPDF (.pdf) Here\n— or click browse below —", self)
        self.text_label.setStyleSheet("font-size: 13px; color: #94A3B8; font-weight: 500; background: transparent;")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                QFrame {
                    border: 2px solid #38BDF8;
                    border-radius: 12px;
                    background-color: #1E293B;
                }
            """)

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #0284C7;
                border-radius: 12px;
                background-color: #0F172A;
            }
        """)

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.fileDropped.emit(file_path)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #0284C7;
                border-radius: 12px;
                background-color: #0F172A;
            }
        """)


class IngestionPanel(QWidget):
    """Step 1 UI: Ingestion and validation interface."""

    fileLoaded = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_file_info = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        # Title / Description
        header = QLabel("Step 1: Ingest Satellite Imagery or GeoPDF Document")
        header.setObjectName("headerTitle")
        sub = QLabel("Select a multi-spectral GeoTIFF or modern GeoPDF document to perform local soil segmentation.")
        sub.setStyleSheet("color: #94A3B8;")

        layout.addWidget(header)
        layout.addWidget(sub)

        # Drag & Drop Zone
        self.drop_zone = DropZoneWidget(self)
        self.drop_zone.fileDropped.connect(self.load_file)
        layout.addWidget(self.drop_zone)

        # Buttons Row (Browse & Sample Buttons)
        btn_layout = QHBoxLayout()
        self.browse_btn = QPushButton("📁 Browse Local Files...")
        self.browse_btn.setObjectName("primaryButton")
        self.browse_btn.clicked.connect(self._on_browse)

        self.load_sample_tif_btn = QPushButton("🧪 Load Sample GeoTIFF")
        self.load_sample_tif_btn.clicked.connect(self._load_sample_tif)

        self.load_sample_pdf_btn = QPushButton("📄 Load Sample GeoPDF")
        self.load_sample_pdf_btn.clicked.connect(self._load_sample_pdf)

        btn_layout.addWidget(self.browse_btn)
        btn_layout.addWidget(self.load_sample_tif_btn)
        btn_layout.addWidget(self.load_sample_pdf_btn)
        btn_layout.addStretch()

        layout.addLayout(btn_layout)

        # File Inspection & Metadata Group
        self.meta_group = QGroupBox("File Metadata & Validation Status")
        grid = QGridLayout(self.meta_group)
        grid.setSpacing(10)

        self.lbl_path = QLabel("No file loaded")
        self.lbl_path.setStyleSheet("font-weight: bold; color: #F8FAFC;")
        self.lbl_format = QLabel("—")
        self.lbl_format.setObjectName("badge")
        self.lbl_crs = QLabel("—")
        self.lbl_dims = QLabel("—")
        self.lbl_channels = QLabel("—")
        self.lbl_size = QLabel("—")

        grid.addWidget(QLabel("Selected File:"), 0, 0)
        grid.addWidget(self.lbl_path, 0, 1, 1, 3)

        grid.addWidget(QLabel("Format:"), 1, 0)
        grid.addWidget(self.lbl_format, 1, 1)

        grid.addWidget(QLabel("CRS Reference:"), 1, 2)
        grid.addWidget(self.lbl_crs, 1, 3)

        grid.addWidget(QLabel("Dimensions:"), 2, 0)
        grid.addWidget(self.lbl_dims, 2, 1)

        grid.addWidget(QLabel("Bands / Pages:"), 2, 2)
        grid.addWidget(self.lbl_channels, 2, 3)

        grid.addWidget(QLabel("File Size:"), 3, 0)
        grid.addWidget(self.lbl_size, 3, 1)

        layout.addWidget(self.meta_group)
        layout.addStretch()

    def _on_browse(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Geospatial Satellite Imagery or GeoPDF",
            "",
            "Geospatial Files (*.tif *.tiff *.pdf);;GeoTIFF (*.tif *.tiff);;GeoPDF (*.pdf);;All Files (*)"
        )
        if file_path:
            self.load_file(file_path)

    def _load_sample_tif(self):
        sample_path = os.path.abspath("samples/sample_geotiff.tif")
        if not os.path.exists(sample_path):
            from samples.generate_sample_data import create_sample_geotiff
            create_sample_geotiff(sample_path)
        self.load_file(sample_path)

    def _load_sample_pdf(self):
        sample_path = os.path.abspath("samples/sample_geopdf.pdf")
        if not os.path.exists(sample_path):
            from samples.generate_sample_data import create_sample_geopdf
            create_sample_geopdf(sample_path)
        self.load_file(sample_path)

    def load_file(self, file_path: str):
        """Validates and loads file into application state."""
        if not file_path or not os.path.exists(file_path):
            return

        file_info = detect_file_format(file_path)
        file_info["file_path"] = file_path
        self.current_file_info = file_info

        if not file_info.get("valid"):
            self.lbl_path.setText(f"<font color='#EF4444'>Error: {file_info.get('error')}</font>")
            self.lbl_format.setText("INVALID")
            return

        fmt = file_info.get("format", "UNKNOWN")
        size_mb = file_info.get("file_size", 0) / (1024 * 1024)

        self.lbl_path.setText(os.path.basename(file_path))
        self.lbl_format.setText(fmt)
        self.lbl_size.setText(f"{size_mb:.2f} MB")

        if fmt == "GEOTIFF":
            self.lbl_crs.setText(file_info.get("crs") or "EPSG:32643 (Assumed)")
            self.lbl_dims.setText(f"{file_info.get('width')} x {file_info.get('height')} px")
            self.lbl_channels.setText(f"{file_info.get('channels')} Bands")
        elif fmt == "PDF":
            self.lbl_crs.setText("GeoPDF Layer Matrix Available")
            self.lbl_dims.setText("Vector / Scalable Points")
            self.lbl_channels.setText(f"{file_info.get('page_count')} Pages")

        self.fileLoaded.emit(file_info)

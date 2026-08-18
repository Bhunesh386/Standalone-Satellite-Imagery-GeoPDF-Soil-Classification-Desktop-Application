"""
Main Desktop Application Window.
Coordinates multi-stage workflow, step navigation, asynchronous execution triggers,
log streaming, and progress tracking.
"""

import os
import rasterio
import numpy as np

from .qt_compat import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTabWidget, QProgressBar, QTextEdit, QGroupBox,
    QMessageBox, Signal, Qt, QSplitter
)
from .theme import DARK_THEME_QSS
from .ingestion_panel import IngestionPanel
from .layer_matrix_panel import LayerMatrixPanel
from .config_panel import ConfigPanel
from .preview_canvas import InteractivePreviewCanvas
from .results_viewer import ResultsViewerPanel
from .execution_worker import PipelineExecutionWorker
from ingestion.render_geopdf import PDFRenderer
from ingestion.inspect_geopdf import GeoPDFInspector


class SoilClassifierMainWindow(QMainWindow):
    """Main window for Standalone Satellite-Imagery & GeoPDF Soil Classifier."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Standalone Satellite-Imagery & GeoPDF Soil Classifier (Offline Native)")
        self.resize(1280, 850)
        self.setStyleSheet(DARK_THEME_QSS)

        self.current_file_path = None
        self.current_file_info = None
        self.worker = None

        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 1. Top Header Bar
        header_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("🛰️ Standalone Satellite-Imagery & GeoPDF Soil Classifier")
        title.setObjectName("headerTitle")
        subtitle = QLabel("Native Local Offline Inference • PyTorch Dual-Head U-Net • GeoTIFF & GeoPDF Engine")
        subtitle.setStyleSheet("color: #94A3B8; font-size: 12px;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        header_layout.addLayout(title_box)
        header_layout.addStretch()

        badge_offline = QLabel("🔒 100% OFFLINE")
        badge_offline.setObjectName("badge")
        badge_offline.setStyleSheet("background-color: #064E3B; color: #34D399; border-color: #059669; padding: 4px 10px;")

        badge_model = QLabel("🧠 U-Net Dual-Head")
        badge_model.setObjectName("badge")

        header_layout.addWidget(badge_offline)
        header_layout.addWidget(badge_model)

        main_layout.addLayout(header_layout)

        # 2. Main Step Tabs
        self.tabs = QTabWidget(self)
        
        # Step 1: Ingestion
        self.panel_ingestion = IngestionPanel(self)
        self.panel_ingestion.fileLoaded.connect(self._on_file_loaded)
        self.tabs.addTab(self.panel_ingestion, "1. Ingestion")

        # Step 2: Layer Matrix
        self.panel_layer_matrix = LayerMatrixPanel(self)
        self.tabs.addTab(self.panel_layer_matrix, "2. GeoPDF Layers")

        # Step 3: Processing Config
        self.panel_config = ConfigPanel(self)
        self.tabs.addTab(self.panel_config, "3. Grid & Export")

        # Step 4: Preview Canvas
        self.panel_preview = InteractivePreviewCanvas(self)
        self.tabs.addTab(self.panel_preview, "4. Spatial Preview")

        # Step 5: Results Viewer
        self.panel_results = ResultsViewerPanel(self)
        self.tabs.addTab(self.panel_results, "5. Results Dashboard")

        self.tabs.currentChanged.connect(self._on_tab_changed)
        main_layout.addWidget(self.tabs, stretch=1)

        # 3. Bottom Execution & Progress Bar
        bottom_box = QVBoxLayout()
        bottom_box.setSpacing(8)

        # Progress bar & status label
        prog_layout = QHBoxLayout()
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        self.lbl_progress_status = QLabel("")
        self.lbl_progress_status.setStyleSheet("color: #38BDF8; font-weight: 500;")

        prog_layout.addWidget(self.progress_bar, stretch=1)
        prog_layout.addWidget(self.lbl_progress_status)
        bottom_box.addLayout(prog_layout)

        # Navigation & Run Action Buttons
        action_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Back")
        self.btn_prev.clicked.connect(self._on_prev_step)
        self.btn_prev.setEnabled(False)

        self.btn_next = QPushButton("Next ▶")
        self.btn_next.clicked.connect(self._on_next_step)

        self.btn_run = QPushButton("⚡ Execute Soil Classification Pipeline")
        self.btn_run.setObjectName("primaryButton")
        self.btn_run.clicked.connect(self._on_run_pipeline)
        self.btn_run.setEnabled(False)

        self.btn_toggle_log = QPushButton("📜 Toggle Log Console")
        self.btn_toggle_log.clicked.connect(self._toggle_log_console)

        action_layout.addWidget(self.btn_prev)
        action_layout.addWidget(self.btn_next)
        action_layout.addStretch()
        action_layout.addWidget(self.btn_toggle_log)
        action_layout.addWidget(self.btn_run)

        bottom_box.addLayout(action_layout)

        # 4. Collapsible Log Console
        self.log_console = QTextEdit(self)
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumHeight(120)
        self.log_console.setVisible(False)
        self.log_console.append("--- Geospatial Soil Classification Engine Initialized ---")
        bottom_box.addWidget(self.log_console)

        main_layout.addLayout(bottom_box)

    def _on_file_loaded(self, file_info: dict):
        self.current_file_info = file_info
        self.current_file_path = file_info.get("file_path")
        self.panel_layer_matrix.set_file_info(file_info)
        self.btn_run.setEnabled(True)
        self.log_console.append(f"📁 Loaded file: {self.current_file_path}")

        # Update preview canvas immediately
        self._update_preview_canvas()

        # Advance to step 2 if PDF, step 3 if GeoTIFF
        if file_info.get("format") == "PDF":
            self.tabs.setCurrentIndex(1)
        else:
            self.tabs.setCurrentIndex(2)

    def _update_preview_canvas(self):
        if not self.current_file_path or not os.path.exists(self.current_file_path):
            return

        fmt = self.current_file_info.get("format")
        if fmt == "GEOTIFF":
            with rasterio.open(self.current_file_path) as src:
                # Read first 3 channels or single channel
                raster = src.read([1, 2, 3] if src.count >= 3 else [1])
                bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
            self.panel_preview.set_preview_image(raster, bounds=bounds)
        elif fmt == "PDF":
            renderer = PDFRenderer(self.current_file_path)
            img_np, _ = renderer.render_page(page_index=0, dpi=150)
            inspector = GeoPDFInspector(self.current_file_path)
            doc_info = inspector.inspect_document()
            neatline = None
            if doc_info["pages"]:
                neatline = doc_info["pages"][0].get("neatline_bbox")
            self.panel_preview.set_preview_image(img_np, neatline=neatline)

    def _on_tab_changed(self, index: int):
        self.btn_prev.setEnabled(index > 0)
        self.btn_next.setEnabled(index < self.tabs.count() - 1)

        # Refresh preview if tab 3 (Spatial Preview) selected
        if index == 3:
            self._update_preview_canvas()

    def _on_prev_step(self):
        cur = self.tabs.currentIndex()
        if cur > 0:
            self.tabs.setCurrentIndex(cur - 1)

    def _on_next_step(self):
        cur = self.tabs.currentIndex()
        if cur < self.tabs.count() - 1:
            self.tabs.setCurrentIndex(cur + 1)

    def _toggle_log_console(self):
        self.log_console.setVisible(not self.log_console.isVisible())

    def _on_run_pipeline(self):
        if not self.current_file_path:
            QMessageBox.warning(self, "No File Selected", "Please select a GeoTIFF or GeoPDF file first in Step 1.")
            return

        # Prepare parameters
        config = self.panel_config.get_config()
        layer_config = self.panel_layer_matrix.get_config()

        # UI state during execution
        self.btn_run.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.lbl_progress_status.setText("Initializing pipeline...")
        self.log_console.setVisible(True)

        # Start QThread worker
        self.worker = PipelineExecutionWorker(
            file_path=self.current_file_path,
            config=config,
            layer_config=layer_config,
            parent=self
        )

        self.worker.progress.connect(self._on_worker_progress)
        self.worker.log_message.connect(self._on_worker_log)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)

        self.worker.start()

    def _on_worker_progress(self, pct: float, msg: str):
        self.progress_bar.setValue(int(pct))
        self.lbl_progress_status.setText(msg)

    def _on_worker_log(self, msg: str):
        self.log_console.append(msg)

    def _on_worker_finished(self, result_bundle: dict):
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_progress_status.setText("Done!")

        # Render in results dashboard
        self.panel_results.display_results(result_bundle)

        # Switch to Results tab (Step 5)
        self.tabs.setCurrentIndex(4)

        QMessageBox.information(
            self,
            "Classification Complete",
            f"Soil classification completed successfully in {result_bundle.get('total_elapsed_s', 0):.2f}s!\nDeliverables exported to: {self.panel_config.get_config()['output_dir']}"
        )

    def _on_worker_error(self, err_msg: str):
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.lbl_progress_status.setText("Failed!")
        QMessageBox.critical(self, "Pipeline Error", f"An error occurred during execution:\n{err_msg}")

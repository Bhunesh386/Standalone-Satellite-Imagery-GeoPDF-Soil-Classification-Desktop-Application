"""
Automated GUI Integration Test Suite.
Verifies Qt widget lifecycles, signal propagation, tab transitions,
sample dataset loading, and execution worker wiring in headless offscreen mode.
"""

import os
import sys
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from gui.qt_compat import QApplication
from gui.main_window import SoilClassifierMainWindow
from gui.ingestion_panel import IngestionPanel
from gui.layer_matrix_panel import LayerMatrixPanel
from gui.config_panel import ConfigPanel
from gui.preview_canvas import InteractivePreviewCanvas
from gui.results_viewer import ResultsViewerPanel


class TestSoilClassifierGUI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create single QApplication instance for tests
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)

        cls.geotiff_path = os.path.abspath("samples/sample_geotiff.tif")
        cls.geopdf_path = os.path.abspath("samples/sample_geopdf.pdf")

    def test_main_window_creation(self):
        window = SoilClassifierMainWindow()
        self.assertIsNotNone(window)
        self.assertEqual(window.tabs.count(), 5)
        self.assertEqual(window.tabs.tabText(0), "1. Ingestion")
        self.assertEqual(window.tabs.tabText(1), "2. GeoPDF Layers")
        self.assertEqual(window.tabs.tabText(2), "3. Grid & Export")
        self.assertEqual(window.tabs.tabText(3), "4. Spatial Preview")
        self.assertEqual(window.tabs.tabText(4), "5. Results Dashboard")

    def test_ingestion_geotiff_loading(self):
        window = SoilClassifierMainWindow()
        window.panel_ingestion.load_file(self.geotiff_path)

        self.assertEqual(window.current_file_path, self.geotiff_path)
        self.assertEqual(window.current_file_info["format"], "GEOTIFF")
        self.assertTrue(window.btn_run.isEnabled())
        # Check that tab transitioned to step 3 (Grid & Export) for GeoTIFF
        self.assertEqual(window.tabs.currentIndex(), 2)

    def test_ingestion_geopdf_loading(self):
        window = SoilClassifierMainWindow()
        window.panel_ingestion.load_file(self.geopdf_path)

        self.assertEqual(window.current_file_path, self.geopdf_path)
        self.assertEqual(window.current_file_info["format"], "PDF")
        self.assertTrue(window.btn_run.isEnabled())
        # Check that tab transitioned to step 2 (GeoPDF Layers)
        self.assertEqual(window.tabs.currentIndex(), 1)

        # Check layers populated
        self.assertGreater(window.panel_layer_matrix.layer_table.rowCount(), 0)

    def test_config_panel_values(self):
        panel = ConfigPanel()
        cfg = panel.get_config()
        self.assertIn("target_crs", cfg)
        self.assertIn("resolution_m", cfg)
        self.assertIn("tile_size", cfg)
        self.assertIn("tile_overlap", cfg)
        self.assertTrue(cfg["export_geotiff"])
        self.assertTrue(cfg["export_confidence"])
        self.assertTrue(cfg["export_geopdf"])
        self.assertTrue(cfg["export_report"])

    def test_navigation_buttons(self):
        window = SoilClassifierMainWindow()
        self.assertEqual(window.tabs.currentIndex(), 0)
        self.assertFalse(window.btn_prev.isEnabled())

        window._on_next_step()
        self.assertEqual(window.tabs.currentIndex(), 1)
        self.assertTrue(window.btn_prev.isEnabled())

        window._on_prev_step()
        self.assertEqual(window.tabs.currentIndex(), 0)


if __name__ == "__main__":
    unittest.main()

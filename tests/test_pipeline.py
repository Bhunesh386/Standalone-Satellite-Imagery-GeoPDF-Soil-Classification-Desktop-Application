"""
Automated unit & integration test suite for geospatial soil classifier backend.
"""

import os
import unittest
import numpy as np
import rasterio

from ingestion.detect_format import detect_file_format
from ingestion.inspect_geopdf import GeoPDFInspector
from ingestion.render_geopdf import PDFRenderer
from ingestion.extract_geopdf_raster import GeoPDFRasterExtractor
from ingestion.extract_geopdf_vector import GeoPDFVectorExtractor
from geospatial.crs import get_crs, get_utm_epsg_for_latlon, transform_bounds
from geospatial.validate import validate_raster_dataset
from geospatial.preprocess import preprocess_raster_for_model
from geospatial.tile import generate_tiles
from geospatial.rasterize import rasterize_vector_geometries
from geospatial.stitch import TileStitcher
from geospatial.export import export_classified_geotiff, export_confidence_geotiff
from models.architecture import DualHeadUNet, load_model
from inference.predict import SoilInferenceEngine
from reporting.legend import SoilLegendManager
from reporting.report import generate_classification_report
from reporting.geopdf_export import export_classified_geopdf


class TestGeospatialSoilPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.geotiff_path = "samples/sample_geotiff.tif"
        cls.geopdf_path = "samples/sample_geopdf.pdf"
        cls.test_output_dir = "tests/test_output"
        os.makedirs(cls.test_output_dir, exist_ok=True)

    def test_format_detection(self):
        tif_info = detect_file_format(self.geotiff_path)
        self.assertTrue(tif_info["valid"])
        self.assertEqual(tif_info["format"], "GEOTIFF")
        self.assertEqual(tif_info["channels"], 4)

        pdf_info = detect_file_format(self.geopdf_path)
        self.assertTrue(pdf_info["valid"])
        self.assertEqual(pdf_info["format"], "PDF")
        self.assertGreaterEqual(pdf_info["page_count"], 1)

    def test_geopdf_inspector(self):
        inspector = GeoPDFInspector(self.geopdf_path)
        doc_info = inspector.inspect_document()
        self.assertGreaterEqual(doc_info["page_count"], 1)
        self.assertTrue(len(doc_info["layers"]) > 0)
        self.assertIn("spectral_warning", doc_info)

    def test_pdf_renderer(self):
        renderer = PDFRenderer(self.geopdf_path)
        img_np, scale = renderer.render_page(page_index=0, dpi=150)
        self.assertEqual(img_np.ndim, 3)
        self.assertEqual(img_np.shape[2], 3)
        self.assertGreater(img_np.shape[0], 100)

        thumbs = renderer.generate_thumbnails(max_size=128)
        self.assertTrue(len(thumbs) >= 1)

    def test_geopdf_raster_and_vector_extraction(self):
        r_extractor = GeoPDFRasterExtractor(self.geopdf_path)
        raster_data = r_extractor.extract_raster_mode_a(page_index=0, dpi=150, strip_furniture=True)
        self.assertIn("raster", raster_data)
        self.assertEqual(raster_data["raster"].ndim, 3)

        v_extractor = GeoPDFVectorExtractor(self.geopdf_path)
        vector_data = v_extractor.extract_vectors(page_index=0)
        self.assertIn("exclusion_masks", vector_data)

    def test_crs_and_validation(self):
        crs = get_crs("EPSG:32643")
        self.assertEqual(crs.to_epsg(), 32643)

        utm = get_utm_epsg_for_latlon(19.0760, 72.8777)
        self.assertEqual(utm, 32643)

        sample_raster = np.random.rand(4, 128, 128).astype(np.float32)
        val = validate_raster_dataset(sample_raster, crs="EPSG:32643")
        self.assertTrue(val["valid"])

    def test_preprocessing_and_tiling(self):
        dummy_raster = np.random.randint(0, 4095, (4, 300, 300), dtype=np.uint16)
        norm, nodata_mask, stats = preprocess_raster_for_model(dummy_raster, target_channels=4)
        self.assertEqual(norm.shape, (4, 300, 300))
        self.assertTrue(0.0 <= np.max(norm) <= 1.0)

        tiles = generate_tiles(norm, tile_size=128, overlap=16)
        self.assertGreater(len(tiles), 1)

    def test_model_and_inference(self):
        engine = SoilInferenceEngine(
            weights_path="models/weights.pt",
            in_channels=4,
            num_classes=6,
            tile_size=256,
            tile_overlap=16,
            batch_size=2,
            device="cpu"
        )
        
        # Load sample GeoTIFF
        with rasterio.open(self.geotiff_path) as src:
            raster = src.read()
            transform = src.transform
            crs = src.crs.to_string()

        # Run inference on sub-window for speed
        sub_raster = raster[:, :512, :512]
        result = engine.run_inference(sub_raster)

        self.assertIn("class_map", result)
        self.assertIn("confidence_map", result)
        self.assertEqual(result["class_map"].shape, (512, 512))
        self.assertEqual(result["confidence_map"].shape, (512, 512))

    def test_export_geotiff_geopdf_and_report(self):
        class_map = np.random.randint(0, 6, (256, 256), dtype=np.uint8)
        conf_map = np.random.uniform(0.5, 0.95, (256, 256)).astype(np.float32)
        transform = rasterio.transform.from_origin(500000, 2100000, 10.0, 10.0)
        crs = "EPSG:32643"

        # GeoTIFF
        tif_out = os.path.join(self.test_output_dir, "test_classified.tif")
        export_classified_geotiff(tif_out, class_map, transform, crs)
        self.assertTrue(os.path.exists(tif_out))

        # Confidence GeoTIFF
        conf_out = os.path.join(self.test_output_dir, "test_conf.tif")
        export_confidence_geotiff(conf_out, conf_map, transform, crs)
        self.assertTrue(os.path.exists(conf_out))

        # JSON Report
        rep_out = os.path.join(self.test_output_dir, "test_report.json")
        generate_classification_report(
            rep_out, class_map, conf_map,
            self.geotiff_path, "GEOTIFF", crs, crs, 10.0, {"test": True}
        )
        self.assertTrue(os.path.exists(rep_out))

        # GeoPDF Export
        pdf_out = os.path.join(self.test_output_dir, "test_classified.pdf")
        export_classified_geopdf(
            pdf_out, class_map,
            (500000, 2097440, 502560, 2100000), crs, 10.0, "sample_geotiff.tif"
        )
        self.assertTrue(os.path.exists(pdf_out))


if __name__ == "__main__":
    unittest.main()

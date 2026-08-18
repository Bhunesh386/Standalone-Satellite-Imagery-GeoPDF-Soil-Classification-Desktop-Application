#!/usr/bin/env python3
"""
Main Entrypoint for Standalone Satellite-Imagery & GeoPDF Soil Classification Desktop Application.
Supports both interactive GUI mode and headless automated processing.
"""

import sys
import os
import argparse

# Add workspace directory to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_gui(preload_file: str = None, test_mode: bool = False):
    """Launches the Qt desktop application."""
    from gui.qt_compat import QApplication
    from gui.main_window import SoilClassifierMainWindow

    # Enable High DPI Scaling if Qt attributes exist
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName("GeoSoilClassifier")
    app.setOrganizationName("GeospatialAnalytics")

    window = SoilClassifierMainWindow()

    if preload_file and os.path.exists(preload_file):
        window.panel_ingestion.load_file(preload_file)

    if test_mode:
        print("Running in GUI test mode (initializing window and self-testing UI components)...")
        window.show()
        # Verify window widgets exist
        assert window.tabs.count() == 5, "Expected 5 step tabs"
        print("GUI components verified successfully!")
        return 0

    window.show()
    return app.exec()


def run_headless_cli(input_path: str, output_dir: str = "output", target_crs: str = "EPSG:32643"):
    """Runs headless inference directly from command line."""
    import rasterio
    from ingestion.detect_format import detect_file_format
    from ingestion.extract_geopdf_raster import GeoPDFRasterExtractor
    from inference.predict import SoilInferenceEngine
    from geospatial.export import export_classified_geotiff, export_confidence_geotiff
    from reporting.legend import SoilLegendManager
    from reporting.report import generate_classification_report
    from reporting.geopdf_export import export_classified_geopdf

    os.makedirs(output_dir, exist_ok=True)
    file_info = detect_file_format(input_path)
    fmt = file_info.get("format")

    print(f"Ingesting {input_path} (Format: {fmt})...")

    if fmt == "GEOTIFF":
        with rasterio.open(input_path) as src:
            raster = src.read()
            transform = src.transform
            src_crs = src.crs.to_string() if src.crs else target_crs
            bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
    elif fmt == "PDF":
        r_ext = GeoPDFRasterExtractor(input_path)
        r_data = r_ext.extract_raster_mode_a(page_index=0, dpi=300, strip_furniture=True, target_crs=target_crs)
        raster = r_data["raster"]
        transform = r_data["transform"]
        src_crs = r_data["crs"]
        bounds = r_data["bounds"]
    else:
        print(f"Error: Unsupported format {fmt}")
        return 1

    print("Running PyTorch U-Net soil classification...")
    engine = SoilInferenceEngine(weights_path="models/weights.pt", in_channels=4, num_classes=6)
    result = engine.run_inference(raster)

    class_map = result["class_map"]
    conf_map = result["confidence_map"]
    metrics = result["metrics"]

    legend_mgr = SoilLegendManager()

    # Export outputs
    tif_out = os.path.join(output_dir, "classified_soil_map.tif")
    export_classified_geotiff(tif_out, class_map, transform, target_crs, legend_mgr.get_color_table(), legend_mgr.get_class_names())

    conf_out = os.path.join(output_dir, "soil_confidence.tif")
    export_confidence_geotiff(conf_out, conf_map, transform, target_crs)

    pdf_out = os.path.join(output_dir, "classified_soil_map.pdf")
    export_classified_geopdf(pdf_out, class_map, tuple(bounds), target_crs, 10.0, os.path.basename(input_path), legend_mgr)

    rep_out = os.path.join(output_dir, "classification_report.json")
    generate_classification_report(rep_out, class_map, conf_map, input_path, fmt, src_crs, target_crs, 10.0, metrics, legend_mgr)

    print(f"✅ All deliverables generated in: {os.path.abspath(output_dir)}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Standalone Satellite-Imagery & GeoPDF Soil Classifier")
    parser.add_argument("--input", "-i", type=str, help="Path to input GeoTIFF or GeoPDF file")
    parser.add_argument("--output", "-o", type=str, default="output", help="Output directory path")
    parser.add_argument("--cli", action="store_true", help="Run in headless CLI mode instead of GUI")
    parser.add_argument("--test", action="store_true", help="Run in headless GUI test mode")
    parser.add_argument("--crs", type=str, default="EPSG:32643", help="Target CRS (default: EPSG:32643)")

    args = parser.parse_args()

    # Ensure sample data exists
    if not os.path.exists("samples/sample_geotiff.tif") or not os.path.exists("samples/sample_geopdf.pdf"):
        from samples.generate_sample_data import create_sample_geotiff, create_sample_geopdf
        create_sample_geotiff()
        create_sample_geopdf()

    if args.cli:
        if not args.input:
            print("Error: --input argument is required in --cli mode.")
            return 1
        return run_headless_cli(args.input, args.output, args.crs)

    return run_gui(preload_file=args.input, test_mode=args.test)


if __name__ == "__main__":
    sys.exit(main())

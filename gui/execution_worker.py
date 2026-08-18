"""
Asynchronous QThread Execution Worker.
Manages the complete pipeline execution in a background thread with granular progress signals.
"""

import os
import time
import numpy as np
import rasterio
from affine import Affine
from PIL import Image

from .qt_compat import QThread, Signal
from ingestion.detect_format import detect_file_format
from ingestion.extract_geopdf_raster import GeoPDFRasterExtractor
from ingestion.extract_geopdf_vector import GeoPDFVectorExtractor
from geospatial.rasterize import rasterize_vector_geometries
from geospatial.export import export_classified_geotiff, export_confidence_geotiff
from inference.predict import SoilInferenceEngine
from reporting.legend import SoilLegendManager
from reporting.report import generate_classification_report
from reporting.geopdf_export import export_classified_geopdf


class PipelineExecutionWorker(QThread):
    """Executes the complete soil classification and export pipeline asynchronously."""

    progress = Signal(float, str)
    log_message = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        file_path: str,
        config: dict,
        layer_config: dict,
        parent=None
    ):
        super().__init__(parent)
        self.file_path = file_path
        self.config = config
        self.layer_config = layer_config
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        t_start = time.time()
        try:
            self.log_message.emit(f"🚀 Starting soil classification job for: {os.path.basename(self.file_path)}")
            self.progress.emit(2.0, "Validating input file format...")

            file_info = detect_file_format(self.file_path)
            fmt = file_info.get("format")
            out_dir = self.config.get("output_dir", os.path.abspath("output"))
            os.makedirs(out_dir, exist_ok=True)

            target_crs = self.config.get("target_crs", "EPSG:32643")
            resolution_m = self.config.get("resolution_m", 10.0)

            # Step 1: Ingestion & Raster Extraction
            if fmt == "GEOTIFF":
                self.log_message.emit("📖 Ingesting multi-band GeoTIFF...")
                with rasterio.open(self.file_path) as src:
                    raster = src.read()
                    transform = src.transform
                    src_crs = src.crs.to_string() if src.crs else target_crs
                    bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
                exclusion_mask = None
            elif fmt == "PDF":
                page_idx = self.layer_config.get("page_index", 0)
                dpi = self.config.get("pdf_dpi", 300)
                self.log_message.emit(f"📄 Rendering GeoPDF Page {page_idx + 1} at {dpi} DPI (Mode A/C)...")
                
                r_extractor = GeoPDFRasterExtractor(self.file_path)
                raster_data = r_extractor.extract_raster_mode_a(
                    page_index=page_idx,
                    dpi=dpi,
                    strip_furniture=True,
                    target_crs=target_crs
                )
                raster = raster_data["raster"]
                transform = raster_data["transform"]
                src_crs = raster_data["crs"]
                bounds = raster_data["bounds"]

                # Extract Vector Exclusion Masks
                self.log_message.emit("📐 Extracting vector layers & road exclusion network...")
                v_extractor = GeoPDFVectorExtractor(self.file_path)
                vector_dict = v_extractor.extract_vectors(
                    page_index=page_idx,
                    layer_role_map=self.layer_config.get("layer_roles", {})
                )
                
                ex_gdf = vector_dict.get("exclusion_masks")
                if ex_gdf is not None and not ex_gdf.empty:
                    h, w = raster.shape[1], raster.shape[2]
                    exclusion_mask = rasterize_vector_geometries(ex_gdf, (h, w), transform, burn_value=1)
                    self.log_message.emit(f"🛡️ Burned {len(ex_gdf)} exclusion geometry features onto mask.")
                else:
                    exclusion_mask = None
            else:
                raise ValueError(f"Unsupported file format: {fmt}")

            if self._is_cancelled:
                return

            # Step 2: Initialize ML Model & Inference Engine
            self.progress.emit(10.0, "Loading PyTorch U-Net Dual-Head Segmentation Model...")
            self.log_message.emit(f"🧠 Initializing U-Net engine (Tile Size: {self.config.get('tile_size', 512)}, Overlap: {self.config.get('tile_overlap', 32)}px)...")

            engine = SoilInferenceEngine(
                weights_path="models/weights.pt",
                in_channels=4,
                num_classes=6,
                tile_size=self.config.get("tile_size", 512),
                tile_overlap=self.config.get("tile_overlap", 32),
                blend_method=self.config.get("blend_method", "cosine"),
                batch_size=4,
                device="auto"
            )

            # Define progress callback
            def on_infer_progress(pct: float, msg: str):
                if not self._is_cancelled:
                    self.progress.emit(pct, msg)
                    if int(pct) % 25 == 0:
                        self.log_message.emit(f"⚙️ {msg}")

            # Step 3: Run Batched Inference
            infer_result = engine.run_inference(
                raster=raster,
                exclusion_mask=exclusion_mask,
                progress_callback=on_infer_progress
            )

            if self._is_cancelled:
                return

            class_map = infer_result["class_map"]
            confidence_map = infer_result["confidence_map"]
            metrics = infer_result["metrics"]

            self.progress.emit(90.0, "Exporting geospatial deliverables...")
            self.log_message.emit("📦 Exporting artifacts to output folder...")

            legend_mgr = SoilLegendManager()
            saved_files = {}

            # 1. GeoTIFF Export
            if self.config.get("export_geotiff", True):
                tif_path = os.path.join(out_dir, "classified_soil_map.tif")
                export_classified_geotiff(
                    output_path=tif_path,
                    class_map=class_map,
                    transform=transform,
                    crs=target_crs,
                    color_table=legend_mgr.get_color_table(),
                    class_names=legend_mgr.get_class_names()
                )
                saved_files["geotiff"] = tif_path
                self.log_message.emit(f"✅ Classified GeoTIFF saved: {tif_path}")

            # 2. Confidence GeoTIFF Export
            if self.config.get("export_confidence", True):
                conf_path = os.path.join(out_dir, "soil_confidence.tif")
                export_confidence_geotiff(
                    output_path=conf_path,
                    confidence_map=confidence_map,
                    transform=transform,
                    crs=target_crs
                )
                saved_files["confidence_tif"] = conf_path
                self.log_message.emit(f"✅ Confidence Heatmap GeoTIFF saved: {conf_path}")

            # 3. GeoPDF Export
            if self.config.get("export_geopdf", True):
                pdf_path = os.path.join(out_dir, "classified_soil_map.pdf")
                export_classified_geopdf(
                    output_pdf_path=pdf_path,
                    class_map=class_map,
                    bounds=tuple(bounds),
                    crs=target_crs,
                    resolution_m=resolution_m,
                    source_filename=os.path.basename(self.file_path),
                    legend_mgr=legend_mgr
                )
                saved_files["geopdf"] = pdf_path
                self.log_message.emit(f"✅ Georeferenced GeoPDF Report saved: {pdf_path}")

            # 4. JSON Report Export
            if self.config.get("export_report", True):
                json_path = os.path.join(out_dir, "classification_report.json")
                report_data = generate_classification_report(
                    output_path=json_path,
                    class_map=class_map,
                    confidence_map=confidence_map,
                    input_file=self.file_path,
                    source_format=fmt,
                    source_crs=src_crs,
                    target_crs=target_crs,
                    resolution_m=resolution_m,
                    metrics=metrics,
                    legend_mgr=legend_mgr
                )
                saved_files["report_json"] = json_path
                self.log_message.emit(f"✅ Classification Audit Report saved: {json_path}")
            else:
                report_data = None

            # 5. Preview PNG & Legend JSON
            preview_png_path = os.path.join(out_dir, "preview.png")
            rgb_preview = legend_mgr.colorize_class_map(class_map)
            Image.fromarray(rgb_preview).save(preview_png_path)
            saved_files["preview_png"] = preview_png_path

            legend_json_path = os.path.join(out_dir, "legend.json")
            legend_mgr.to_json(legend_json_path)
            saved_files["legend_json"] = legend_json_path

            total_elapsed = time.time() - t_start
            self.log_message.emit(f"🎉 Job completed successfully in {total_elapsed:.2f}s!")
            self.progress.emit(100.0, "Complete!")

            result_bundle = {
                "class_map": class_map,
                "confidence_map": confidence_map,
                "input_raster": raster,
                "transform": transform,
                "crs": target_crs,
                "bounds": bounds,
                "saved_files": saved_files,
                "report_data": report_data,
                "total_elapsed_s": total_elapsed
            }
            self.finished.emit(result_bundle)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.log_message.emit(f"❌ Error occurred: {str(e)}\n{tb}")
            self.error.emit(f"{str(e)}\n\nTraceback:\n{tb}")

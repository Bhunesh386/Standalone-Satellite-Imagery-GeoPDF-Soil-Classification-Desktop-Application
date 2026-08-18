"""
JSON Audit and Classification Report Generator.
Computes pixel counts, percentage distributions, ground area calculations (km2, ha),
confidence metrics, and execution runtime audits.
"""

import os
import json
import datetime
from typing import Dict, Any, List, Optional
import numpy as np
from affine import Affine

from .legend import SoilLegendManager


def generate_classification_report(
    output_path: str,
    class_map: np.ndarray,
    confidence_map: np.ndarray,
    input_file: str,
    source_format: str,
    source_crs: str,
    target_crs: str,
    resolution_m: float,
    metrics: Dict[str, Any],
    legend_mgr: Optional[SoilLegendManager] = None
) -> Dict[str, Any]:
    """
    Generates and saves a comprehensive classification audit JSON report.
    
    Args:
        output_path: Path to write classification_report.json.
        class_map: (H, W) uint8 class array.
        confidence_map: (H, W) float32 confidence array.
        input_file: Source file path.
        source_format: 'GEOTIFF' or 'PDF'.
        source_crs: Source CRS string.
        target_crs: Destination target CRS string.
        resolution_m: Ground resolution in meters per pixel.
        metrics: Timing and execution metrics dict from inference engine.
        legend_mgr: Optional SoilLegendManager.
        
    Returns:
        report_dict: Complete report dictionary.
    """
    if legend_mgr is None:
        legend_mgr = SoilLegendManager()

    h, w = class_map.shape
    total_pixels = h * w
    pixel_area_m2 = resolution_m * resolution_m

    # Compute class distributions
    classes_data = legend_mgr.classes_data
    class_stats = []
    valid_pixels_count = 0

    for c in classes_data:
        cid = c["id"]
        count = int(np.sum(class_map == cid))
        if cid != 0:
            valid_pixels_count += count

        area_m2 = count * pixel_area_m2
        area_ha = area_m2 / 10000.0
        area_km2 = area_m2 / 1000000.0
        pct_total = (count / total_pixels) * 100.0

        class_stats.append({
            "class_id": cid,
            "class_name": c["name"],
            "hex": c.get("hex", "#000000"),
            "pixel_count": count,
            "percentage_of_scene": round(pct_total, 2),
            "area_sq_meters": round(area_m2, 2),
            "area_hectares": round(area_ha, 3),
            "area_sq_km": round(area_km2, 4),
            "description": c.get("description", ""),
            "fertility": c.get("fertility", "N/A"),
            "permeability": c.get("permeability", "N/A")
        })

    # Add percentage of valid area (excluding NoData)
    for cs in class_stats:
        if cs["class_id"] == 0 or valid_pixels_count == 0:
            cs["percentage_of_classified_land"] = 0.0
        else:
            cs["percentage_of_classified_land"] = round((cs["pixel_count"] / valid_pixels_count) * 100.0, 2)

    # Compute confidence statistics over valid (non-zero) classified pixels
    valid_mask = (class_map > 0)
    if np.any(valid_mask):
        valid_confs = confidence_map[valid_mask]
        conf_mean = float(np.mean(valid_confs))
        conf_median = float(np.median(valid_confs))
        conf_p10 = float(np.percentile(valid_confs, 10.0))
        conf_p90 = float(np.percentile(valid_confs, 90.0))
        low_conf_count = int(np.sum(valid_confs < 0.40))
        low_conf_pct = round((low_conf_count / len(valid_confs)) * 100.0, 2)
    else:
        conf_mean = 0.0
        conf_median = 0.0
        conf_p10 = 0.0
        conf_p90 = 0.0
        low_conf_count = 0
        low_conf_pct = 0.0

    report = {
        "audit_metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "software": "Standalone Satellite-Imagery & GeoPDF Soil Classifier",
            "version": "1.0.0",
            "offline_mode": True
        },
        "input_dataset": {
            "source_file": os.path.abspath(input_file),
            "filename": os.path.basename(input_file),
            "format": source_format,
            "source_crs": source_crs,
            "dimensions_pixels": {"width": w, "height": h},
            "total_scene_pixels": total_pixels
        },
        "processing_configuration": {
            "target_crs": target_crs,
            "ground_resolution_m_per_pixel": resolution_m,
            "pixel_area_m2": pixel_area_m2,
            "total_scene_area_km2": round((total_pixels * pixel_area_m2) / 1000000.0, 4),
            "classified_land_area_km2": round((valid_pixels_count * pixel_area_m2) / 1000000.0, 4)
        },
        "confidence_assessment": {
            "mean_confidence": round(conf_mean, 4),
            "median_confidence": round(conf_median, 4),
            "percentile_10": round(conf_p10, 4),
            "percentile_90": round(conf_p90, 4),
            "low_confidence_pixels": low_conf_count,
            "low_confidence_percentage": low_conf_pct
        },
        "soil_class_distribution": class_stats,
        "performance_metrics": metrics
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report

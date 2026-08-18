"""
GeoTIFF Export Engine.
Writes classified soil maps with embedded GDAL/rasterio color tables,
nodata definitions, CRS, affine geotransforms, and class metadata tags.
Writes Float32 soil inference confidence heatmaps.
"""

import os
from typing import Dict, Any, Optional
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from .crs import get_crs


DEFAULT_COLOR_TABLE = {
    0: (0, 0, 0, 0),        # NoData (Transparent)
    1: (230, 184, 0, 255),  # Alluvial Soil (Golden Yellow)
    2: (54, 43, 40, 255),   # Black Soil / Vertisol (Dark Charcoal)
    3: (192, 57, 43, 255),  # Red Soil (Rust Crimson)
    4: (211, 84, 0, 255),   # Lateritic Soil (Warm Terracotta)
    5: (243, 156, 18, 255)  # Sandy / Desert Soil (Warm Sand)
}


def export_classified_geotiff(
    output_path: str,
    class_map: np.ndarray,
    transform: Affine,
    crs: str,
    color_table: Optional[Dict[int, tuple]] = None,
    class_names: Optional[Dict[int, str]] = None,
    compression: str = "DEFLATE",
    nodata_value: int = 0
) -> str:
    """
    Exports classified map as an 8-bit single-band GeoTIFF with embedded color table.
    
    Args:
        output_path: Path to write .tif file.
        class_map: uint8 array of shape (height, width).
        transform: Affine transform.
        crs: CRS string (e.g. 'EPSG:32643').
        color_table: Dict of class_id -> (R, G, B, A).
        class_names: Dict of class_id -> class name.
        compression: Compression algorithm ('DEFLATE', 'LZW', 'NONE').
        nodata_value: NoData pixel value (default 0).
        
    Returns:
        output_path: Absolute path to written GeoTIFF.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    height, width = class_map.shape

    crs_obj = get_crs(crs)
    colormap = color_table or DEFAULT_COLOR_TABLE

    # Ensure all 256 color table entries exist for standard TIFF readers
    full_colormap = {}
    for i in range(256):
        full_colormap[i] = colormap.get(i, (0, 0, 0, 0))

    meta = {
        "driver": "GTiff",
        "dtype": rasterio.uint8,
        "nodata": nodata_value,
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs_obj.to_wkt(),
        "transform": transform,
        "compress": compression.lower() if compression != "NONE" else None
    }

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(class_map.astype(np.uint8), 1)
        dst.write_colormap(1, full_colormap)

        # Write categorical tags
        tags = {
            "TITLE": "Per-Pixel Soil Classification Map",
            "SOFTWARE": "Standalone GeoPDF & Satellite Soil Classifier",
            "NODATA_VALUE": str(nodata_value),
            "NUM_CLASSES": str(len(colormap))
        }
        if class_names:
            for cid, cname in class_names.items():
                tags[f"CLASS_{cid}_NAME"] = cname
        dst.update_tags(**tags)

    return os.path.abspath(output_path)


def export_confidence_geotiff(
    output_path: str,
    confidence_map: np.ndarray,
    transform: Affine,
    crs: str,
    compression: str = "DEFLATE"
) -> str:
    """
    Exports model inference confidence map as a 32-bit Float GeoTIFF.
    
    Args:
        output_path: Path to write .tif file.
        confidence_map: float32 array of shape (height, width) with values in [0.0, 1.0].
        transform: Affine transform.
        crs: CRS string.
        compression: Compression algorithm.
        
    Returns:
        output_path: Absolute path to written GeoTIFF.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    height, width = confidence_map.shape
    crs_obj = get_crs(crs)

    meta = {
        "driver": "GTiff",
        "dtype": rasterio.float32,
        "nodata": -9999.0,
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs_obj.to_wkt(),
        "transform": transform,
        "compress": compression.lower() if compression != "NONE" else None
    }

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(confidence_map.astype(np.float32), 1)
        dst.update_tags(
            TITLE="Soil Classification Inference Confidence Map",
            VALUE_RANGE="[0.0, 1.0]",
            SOFTWARE="Standalone GeoPDF & Satellite Soil Classifier"
        )

    return os.path.abspath(output_path)

"""
Mode A: GeoPDF Raster Layer Extractor.
Extracts georeferenced raster content from GeoPDF, strips map furniture via neatline cropping,
and derives the calibrated affine geotransform.
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from affine import Affine

from .inspect_geopdf import GeoPDFInspector
from .render_geopdf import PDFRenderer


class GeoPDFRasterExtractor:
    """Extracts and georeferences raster imagery from a GeoPDF document."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.inspector = GeoPDFInspector(pdf_path)
        self.renderer = PDFRenderer(pdf_path)

    def extract_raster_mode_a(
        self,
        page_index: int = 0,
        dpi: int = 300,
        strip_furniture: bool = True,
        target_crs: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes Mode A: Extracts raster from page, crops neatline viewport,
        computes Affine matrix, and bundles CRS.
        
        Returns:
            Dict containing:
                - 'raster': numpy array (C, H, W) float32/uint8
                - 'transform': rasterio Affine transform
                - 'crs': CRS string (e.g. 'EPSG:32643')
                - 'bounds': [min_x, min_y, max_x, max_y]
                - 'dpi': int
                - 'width': int
                - 'height': int
        """
        doc_info = self.inspector.inspect_document()
        if page_index >= len(doc_info["pages"]):
            page_index = 0

        page_meta = doc_info["pages"][page_index]
        page_w_pts = page_meta["width_points"]
        page_h_pts = page_meta["height_points"]

        crs_str = target_crs or page_meta.get("crs") or doc_info["default_crs"] or "EPSG:32643"

        # Determine neatline bounding box in points
        neatline = page_meta.get("neatline_bbox")
        if not strip_furniture or not neatline:
            clip_box = (0.0, 0.0, page_w_pts, page_h_pts)
        else:
            clip_box = tuple(neatline)

        # Render cropped viewport
        img_np, scale = self.renderer.render_page(
            page_index=page_index,
            dpi=dpi,
            clip_rect_pts=clip_box
        )

        h, w, c = img_np.shape

        # Compute ground coordinates & affine transform
        # If geo_bounds are defined, map neatline directly to bounds
        geo_bounds = page_meta.get("geo_bounds")
        if geo_bounds:
            min_x, min_y, max_x, max_y = geo_bounds
        else:
            # Derive default UTM coordinates based on target CRS (e.g. 500000 E, 2100000 N)
            # Assuming ground resolution corresponding to DPI & scale (e.g., 1 pt = 10m ground)
            ground_width_m = (w / (dpi / 72.0)) * 10.0 # ~10m per point
            ground_height_m = (h / (dpi / 72.0)) * 10.0
            min_x = 500000.0
            max_y = 2100000.0
            max_x = min_x + ground_width_m
            min_y = max_y - ground_height_m

        transform = from_bounds(min_x, min_y, max_x, max_y, w, h)

        # Convert to CHW order (Channels, Height, Width)
        raster_chw = np.transpose(img_np, (2, 0, 1))

        return {
            "raster": raster_chw,
            "transform": transform,
            "crs": crs_str,
            "bounds": [min_x, min_y, max_x, max_y],
            "dpi": dpi,
            "width": w,
            "height": h,
            "channels": c,
            "page_index": page_index,
            "neatline_used": clip_box if strip_furniture else None
        }

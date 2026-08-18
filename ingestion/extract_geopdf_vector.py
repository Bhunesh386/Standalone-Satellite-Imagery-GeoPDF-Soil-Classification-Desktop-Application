"""
Mode B & C: GeoPDF Vector Extractor.
Extracts vector paths, lines, polylines, and polygons from GeoPDF drawings and text,
converting page-point coordinates into georeferenced Shapely geometries.
"""

import os
from typing import Dict, Any, List, Optional
import pymupdf
import numpy as np
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd
from affine import Affine


class GeoPDFVectorExtractor:
    """Extracts vector features from GeoPDF layers and converts to georeferenced geometries."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract_vectors(
        self,
        page_index: int = 0,
        page_transform: Optional[Affine] = None,
        pts_to_geo_transform: Optional[Affine] = None,
        layer_role_map: Optional[Dict[str, str]] = None
    ) -> Dict[str, gpd.GeoDataFrame]:
        """
        Extracts vector drawings from PDF page and categorizes them by role:
        - 'exclusion_masks': roads, buildings, borders, furniture
        - 'reference_labels': ground truth polygons
        - 'other_vectors': general paths
        
        Returns:
            Dict mapping role to GeoDataFrame.
        """
        doc = pymupdf.open(self.pdf_path)
        if page_index >= len(doc):
            page_index = 0

        page = doc[page_index]
        drawings = page.get_drawings()
        page_rect = page.rect

        # Default transform if not provided
        if pts_to_geo_transform is None:
            # Map page (0,0) -> (500000, 2100000)
            pts_to_geo_transform = Affine.translation(500000, 2100000) * Affine.scale(10.0, -10.0)

        exclusion_geoms = []
        reference_geoms = []
        other_geoms = []

        for d in drawings:
            layer_name = d.get("layer", "") or ""
            role = "Exclusion Mask"
            if layer_role_map and layer_name in layer_role_map:
                role = layer_role_map[layer_name]
            elif any(k in layer_name.lower() for k in ["ref", "truth", "soil"]):
                role = "Reference Label"
            elif any(k in layer_name.lower() for k in ["road", "mask", "neat", "border", "furn"]):
                role = "Exclusion Mask"

            # Parse path items (lines, curves, rects)
            rect = d.get("rect")
            items = d.get("items", [])

            pts = []
            for item in items:
                cmd = item[0]
                if cmd == "l": # line (p1, p2)
                    p1, p2 = item[1], item[2]
                    pts.extend([(p1.x, p1.y), (p2.x, p2.y)])
                elif cmd == "re": # rect (Rect)
                    r = item[1]
                    pts.extend([(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1), (r.x0, r.y0)])
                elif cmd == "c": # curve (p1, p2, p3, p4)
                    p1, p4 = item[1], item[4]
                    pts.extend([(p1.x, p1.y), (p4.x, p4.y)])

            if len(pts) >= 2:
                # Transform point coordinates from PDF points to ground coordinates
                geo_pts = [pts_to_geo_transform * (px, py) for px, py in pts]

                try:
                    if len(geo_pts) >= 4 and geo_pts[0] == geo_pts[-1]:
                        geom = Polygon(geo_pts)
                    elif len(geo_pts) >= 3 and d.get("fill") is not None:
                        geom = Polygon(geo_pts + [geo_pts[0]])
                    else:
                        geom = LineString(geo_pts).buffer(5.0) # Buffer line to give road width

                    if geom.is_valid and not geom.is_empty:
                        if role == "Exclusion Mask":
                            exclusion_geoms.append(geom)
                        elif role == "Reference Label":
                            reference_geoms.append(geom)
                        else:
                            other_geoms.append(geom)
                except Exception:
                    pass

        doc.close()

        def make_gdf(geoms: List[Any]) -> gpd.GeoDataFrame:
            if geoms:
                return gpd.GeoDataFrame({"geometry": geoms})
            return gpd.GeoDataFrame({"geometry": []})

        return {
            "exclusion_masks": make_gdf(exclusion_geoms),
            "reference_labels": make_gdf(reference_geoms),
            "other_vectors": make_gdf(other_geoms)
        }

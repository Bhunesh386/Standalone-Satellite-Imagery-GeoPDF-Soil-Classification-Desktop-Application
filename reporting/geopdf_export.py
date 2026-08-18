"""
Geospatial PDF Export Engine.
Uses ReportLab to generate publication-quality georeferenced GeoPDF documents
complete with colorized map viewports, scale bars, coordinate graticules,
and formatted soil class legends.
"""

import os
import tempfile
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

from .legend import SoilLegendManager


def export_classified_geopdf(
    output_pdf_path: str,
    class_map: np.ndarray,
    bounds: tuple[float, float, float, float],
    crs: str,
    resolution_m: float,
    source_filename: str,
    legend_mgr: Optional[SoilLegendManager] = None
) -> str:
    """
    Generates a high-quality Georeferenced PDF map sheet using ReportLab.
    
    Args:
        output_pdf_path: Destination path for .pdf file.
        class_map: (H, W) uint8 class array.
        bounds: (min_x, min_y, max_x, max_y) in ground CRS coordinates.
        crs: CRS identifier string (e.g. 'EPSG:32643').
        resolution_m: Ground resolution in meters per pixel.
        source_filename: Name of input imagery file.
        legend_mgr: SoilLegendManager instance.
        
    Returns:
        output_pdf_path: Absolute path to written PDF.
    """
    if legend_mgr is None:
        legend_mgr = SoilLegendManager()

    os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)

    # Use landscape A4 (841.89 x 595.27 points)
    page_w, page_h = landscape(A4)
    c = canvas.Canvas(output_pdf_path, pagesize=(page_w, page_h))

    # Colorize class map to temporary PNG
    rgb_map = legend_mgr.colorize_class_map(class_map)
    pil_img = Image.fromarray(rgb_map)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_img_path = tmp_file.name
        pil_img.save(tmp_img_path, format="PNG")

    try:
        # Background canvas styling
        c.setFillColor(colors.HexColor("#1A1E24"))
        c.rect(0, 0, page_w, page_h, fill=True, stroke=False)

        # Header Title Bar
        c.setFillColor(colors.HexColor("#242B35"))
        c.rect(15, page_h - 60, page_w - 30, 45, fill=True, stroke=False)

        c.setFillColor(colors.HexColor("#FFFFFF"))
        c.setFont("Helvetica-Bold", 16)
        c.drawString(30, page_h - 38, "GEOSPATIAL SOIL CLASSIFICATION MAP")

        c.setFillColor(colors.HexColor("#00B4D8"))
        c.setFont("Helvetica", 9)
        c.drawString(30, page_h - 52, f"Source: {source_filename}  |  CRS: {crs}  |  Resolution: {resolution_m:.1f} m/px  |  Standard: USDA/FAO")

        # Map Frame Coordinates
        map_x = 25
        map_y = 70
        map_w = 480
        map_h = page_h - 150

        # Draw Map Image
        c.drawImage(tmp_img_path, map_x, map_y, width=map_w, height=map_h, preserveAspectRatio=True)

        # Map Border Neatline
        c.setStrokeColor(colors.HexColor("#00B4D8"))
        c.setLineWidth(1.5)
        c.rect(map_x, map_y, map_w, map_h, fill=False, stroke=True)

        # Coordinate Labels on neatline
        min_x, min_y, max_x, max_y = bounds
        c.setFillColor(colors.HexColor("#94A3B8"))
        c.setFont("Helvetica", 8)
        c.drawString(map_x + 5, map_y + 5, f"SW: ({min_x:.0f}, {min_y:.0f})")
        c.drawRightString(map_x + map_w - 5, map_y + 5, f"SE: ({max_x:.0f}, {min_y:.0f})")
        c.drawString(map_x + 5, map_y + map_h - 12, f"NW: ({min_x:.0f}, {max_y:.0f})")
        c.drawRightString(map_x + map_w - 5, map_y + map_h - 12, f"NE: ({max_x:.0f}, {max_y:.0f})")

        # Scale Bar
        ground_span_m = (max_x - min_x)
        bar_len_pts = 100
        bar_span_km = (ground_span_m * (bar_len_pts / map_w)) / 1000.0
        scale_x = map_x + 20
        scale_y = map_y + 25

        c.setFillColor(colors.HexColor("#1A1E24"))
        c.rect(scale_x - 5, scale_y - 8, bar_len_pts + 10, 24, fill=True, stroke=False)
        c.setStrokeColor(colors.HexColor("#FFFFFF"))
        c.setLineWidth(2)
        c.line(scale_x, scale_y, scale_x + bar_len_pts, scale_y)
        c.line(scale_x, scale_y - 3, scale_x, scale_y + 3)
        c.line(scale_x + bar_len_pts, scale_y - 3, scale_x + bar_len_pts, scale_y + 3)
        c.setFillColor(colors.HexColor("#FFFFFF"))
        c.setFont("Helvetica-Bold", 7)
        c.drawCentredString(scale_x + bar_len_pts / 2, scale_y + 5, f"approx. {bar_span_km:.1f} km")

        # North Arrow
        na_x = map_x + map_w - 30
        na_y = map_y + map_h - 35
        c.setFillColor(colors.HexColor("#1A1E24"))
        c.circle(na_x, na_y, 14, fill=True, stroke=False)
        c.setFillColor(colors.HexColor("#EF4444"))
        p = c.beginPath()
        p.moveTo(na_x, na_y + 10)
        p.lineTo(na_x - 5, na_y - 6)
        p.lineTo(na_x + 5, na_y - 6)
        p.close()
        c.drawPath(p, fill=True, stroke=False)
        c.setFillColor(colors.HexColor("#FFFFFF"))
        c.setFont("Helvetica-Bold", 7)
        c.drawCentredString(na_x, na_y - 5, "N")

        # Right Panel: Legend & Summary Table
        panel_x = map_x + map_w + 20
        panel_w = page_w - panel_x - 25
        panel_y = map_y

        c.setFillColor(colors.HexColor("#242B35"))
        c.rect(panel_x, panel_y, panel_w, map_h, fill=True, stroke=False)
        c.setStrokeColor(colors.HexColor("#334155"))
        c.rect(panel_x, panel_y, panel_w, map_h, fill=False, stroke=True)

        # Legend Header
        c.setFillColor(colors.HexColor("#FFFFFF"))
        c.setFont("Helvetica-Bold", 12)
        c.drawString(panel_x + 15, panel_y + map_h - 25, "SOIL CLASSIFICATION LEGEND")

        c.setStrokeColor(colors.HexColor("#00B4D8"))
        c.setLineWidth(1)
        c.line(panel_x + 15, panel_y + map_h - 32, panel_x + panel_w - 15, panel_y + map_h - 32)

        # Draw Class Swatches & Statistics
        cur_y = panel_y + map_h - 55
        total_valid = np.sum(class_map > 0)

        for item in legend_mgr.classes_data:
            cid = item["id"]
            if cid == 0:
                continue # Skip NoData in legend list

            hex_col = item.get("hex", "#000000")
            name = item.get("name", f"Class {cid}")
            fert = item.get("fertility", "N/A")
            perm = item.get("permeability", "N/A")
            count = int(np.sum(class_map == cid))
            pct = (count / total_valid * 100.0) if total_valid > 0 else 0.0
            area_km2 = (count * resolution_m * resolution_m) / 1000000.0

            # Color Swatch Box
            c.setFillColor(colors.HexColor(hex_col))
            c.rect(panel_x + 15, cur_y - 2, 16, 16, fill=True, stroke=False)
            c.setStrokeColor(colors.HexColor("#FFFFFF"))
            c.setLineWidth(0.5)
            c.rect(panel_x + 15, cur_y - 2, 16, 16, fill=False, stroke=True)

            # Class Name & %
            c.setFillColor(colors.HexColor("#FFFFFF"))
            c.setFont("Helvetica-Bold", 9)
            c.drawString(panel_x + 38, cur_y + 6, name)

            c.setFillColor(colors.HexColor("#00B4D8"))
            c.setFont("Helvetica-Bold", 9)
            c.drawRightString(panel_x + panel_w - 15, cur_y + 6, f"{pct:.1f}% ({area_km2:.2f} km²)")

            # Soil Properties
            c.setFillColor(colors.HexColor("#94A3B8"))
            c.setFont("Helvetica", 7)
            c.drawString(panel_x + 38, cur_y - 4, f"Fertility: {fert}  |  Permeability: {perm}")

            cur_y -= 38

        # Summary Audit Box at bottom of right panel
        audit_y = panel_y + 15
        c.setFillColor(colors.HexColor("#1A1E24"))
        c.rect(panel_x + 10, audit_y, panel_w - 20, 95, fill=True, stroke=False)
        c.setStrokeColor(colors.HexColor("#334155"))
        c.rect(panel_x + 10, audit_y, panel_w - 20, 95, fill=False, stroke=True)

        c.setFillColor(colors.HexColor("#00B4D8"))
        c.setFont("Helvetica-Bold", 8)
        c.drawString(panel_x + 20, audit_y + 78, "PROCESSING SUMMARY")

        c.setFillColor(colors.HexColor("#CBD5E1"))
        c.setFont("Helvetica", 7)
        c.drawString(panel_x + 20, audit_y + 62, f"Total Mapped Pixels: {class_map.size:,}")
        c.drawString(panel_x + 20, audit_y + 48, f"Classified Land Area: {(total_valid * resolution_m**2)/1e6:.2f} km²")
        c.drawString(panel_x + 20, audit_y + 34, f"Segmentation Engine: Dual-Head PyTorch U-Net")
        c.drawString(panel_x + 20, audit_y + 20, f"Confidence Calibration: Entropy-Weighted Softmax")
        c.drawString(panel_x + 20, audit_y + 8, f"Execution Mode: Standalone Offline Local Inference")

        # Footer
        c.setFillColor(colors.HexColor("#64748B"))
        c.setFont("Helvetica", 7)
        c.drawString(25, 25, "Generated with Standalone Satellite-Imagery & GeoPDF Soil Classification Desktop Application")
        c.drawRightString(page_w - 25, 25, "Confidential - For Geospatial Land-Use Analytics")

        c.showPage()
        c.save()

    finally:
        if os.path.exists(tmp_img_path):
            os.remove(tmp_img_path)

    return os.path.abspath(output_pdf_path)

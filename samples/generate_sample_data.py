"""
Sample Geospatial Dataset Generator.
Generates realistic multi-band GeoTIFF and multi-page GeoPDF sample files
with simulated satellite spectral signatures, rivers, terrain, and vector layers for offline testing.
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
import pymupdf
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.pdfgen import canvas


def generate_synthetic_satellite_raster(width: int = 1024, height: int = 1024) -> np.ndarray:
    """
    Generates a 4-band (R, G, B, NIR) synthetic satellite scene (1024x1024)
    with realistic spectral variations for alluvial, black, red, lateritic, and sandy soils.
    """
    np.random.seed(42)
    y, x = np.mgrid[0:height, 0:width]

    # Feature 1: Meandering river and alluvial floodplain (sinusoidal)
    river_center_x = width * 0.4 + np.sin(y / 80.0) * 120.0
    river_dist = np.abs(x - river_center_x)
    river_mask = river_dist < 20.0
    alluvial_mask = (river_dist >= 20.0) & (river_dist < 180.0)

    # Feature 2: Black soil vertisol plain (top-right quadrant)
    black_soil_mask = (x > width * 0.55) & (y < height * 0.5) & (~alluvial_mask) & (~river_mask)

    # Feature 3: Red soil hills (bottom-left quadrant)
    red_soil_mask = (x < width * 0.45) & (y > height * 0.45) & (~alluvial_mask) & (~river_mask)

    # Feature 4: Sandy / desert zone (bottom-right quadrant)
    sandy_mask = (x > width * 0.6) & (y > height * 0.55) & (~alluvial_mask) & (~river_mask)

    # Feature 5: Lateritic plateaus (remaining areas)
    lateritic_mask = ~(river_mask | alluvial_mask | black_soil_mask | red_soil_mask | sandy_mask)

    # Initialize 4 bands: [Red, Green, Blue, NIR]
    bands = np.zeros((4, height, width), dtype=np.float32)

    # Noise textures
    noise = np.random.normal(0, 0.03, (4, height, width)).astype(np.float32)

    # Water (River): Low NIR, moderate Blue/Green
    bands[0, river_mask] = 0.08  # Red
    bands[1, river_mask] = 0.15  # Green
    bands[2, river_mask] = 0.25  # Blue
    bands[3, river_mask] = 0.02  # NIR

    # Alluvial Soil: High fertility, moderate brightness, golden-yellowish reflect
    bands[0, alluvial_mask] = 0.55
    bands[1, alluvial_mask] = 0.48
    bands[2, alluvial_mask] = 0.20
    bands[3, alluvial_mask] = 0.65

    # Black Soil (Vertisol): Very dark, low reflectance across all bands
    bands[0, black_soil_mask] = 0.18
    bands[1, black_soil_mask] = 0.16
    bands[2, black_soil_mask] = 0.15
    bands[3, black_soil_mask] = 0.22

    # Red Soil: High red reflectance, low green/blue
    bands[0, red_soil_mask] = 0.70
    bands[1, red_soil_mask] = 0.25
    bands[2, red_soil_mask] = 0.20
    bands[3, red_soil_mask] = 0.45

    # Sandy / Desert Soil: High reflectance across all visible & NIR
    bands[0, sandy_mask] = 0.80
    bands[1, sandy_mask] = 0.72
    bands[2, sandy_mask] = 0.45
    bands[3, sandy_mask] = 0.85

    # Lateritic Soil: Moderate red-orange, warm tones
    bands[0, lateritic_mask] = 0.62
    bands[1, lateritic_mask] = 0.35
    bands[2, lateritic_mask] = 0.18
    bands[3, lateritic_mask] = 0.50

    # Add realistic texture noise and scale to 12-bit DN (0 - 4095)
    bands = np.clip(bands + noise, 0.01, 1.0)
    dn_raster = (bands * 4095.0).astype(np.uint16)

    return dn_raster


def create_sample_geotiff(output_path: str = "samples/sample_geotiff.tif") -> str:
    """Creates a 4-band georeferenced GeoTIFF in UTM 43N."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    raster_4b = generate_synthetic_satellite_raster(1024, 1024)

    # 10m pixel resolution, starting at UTM Easting 500000, Northing 2100000
    transform = from_origin(500000.0, 2100000.0, 10.0, 10.0)
    crs = "EPSG:32643"

    meta = {
        "driver": "GTiff",
        "dtype": rasterio.uint16,
        "nodata": 0,
        "width": 1024,
        "height": 1024,
        "count": 4,
        "crs": crs,
        "transform": transform,
        "compress": "deflate"
    }

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(raster_4b)
        dst.set_band_description(1, "Red (Band 4)")
        dst.set_band_description(2, "Green (Band 3)")
        dst.set_band_description(3, "Blue (Band 2)")
        dst.set_band_description(4, "Near-Infrared (Band 8)")
        dst.update_tags(
            MISSION="Synthetic Sentinel-2 MSI Simulator",
            SENSOR="MSI MultiSpectral",
            RESOLUTION="10m Ground Sample Distance"
        )

    print(f"Sample GeoTIFF generated at: {os.path.abspath(output_path)}")
    return os.path.abspath(output_path)


def create_sample_geopdf(output_path: str = "samples/sample_geopdf.pdf") -> str:
    """Creates a multi-page GeoPDF with rendered imagery and vector drawing layers."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    page_w, page_h = landscape(A4)

    c = canvas.Canvas(output_pdf_path:=output_path, pagesize=(page_w, page_h))

    # Page 1: Main Map Sheet
    # 1. Background
    c.setFillColor(colors.HexColor("#0F172A"))
    c.rect(0, 0, page_w, page_h, fill=True, stroke=False)

    # 2. Header
    c.setFillColor(colors.HexColor("#1E293B"))
    c.rect(20, page_h - 55, page_w - 40, 40, fill=True, stroke=False)
    c.setFillColor(colors.HexColor("#FFFFFF"))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(35, page_h - 35, "REGIONAL SATELLITE BASEMAP & SOIL RECONNAISSANCE")
    c.setFillColor(colors.HexColor("#38BDF8"))
    c.setFont("Helvetica", 8)
    c.drawString(35, page_h - 48, "EPSG:32643 (WGS 84 / UTM Zone 43N)  |  Scale: 1:50,000  |  Sheet: IN-MH-043")

    # 3. Viewport Map Rect (Simulating Map Sheet Neatline)
    map_x = 30
    map_y = 50
    map_w = 520
    map_h = page_h - 120

    # Draw colored terrain base inside neatline
    c.setFillColor(colors.HexColor("#334155"))
    c.rect(map_x, map_y, map_w, map_h, fill=True, stroke=False)

    # Draw terrain color zones
    c.setFillColor(colors.HexColor("#B45309")) # Laterite
    c.rect(map_x, map_y, map_w, map_h, fill=True, stroke=False)

    c.setFillColor(colors.HexColor("#292524")) # Black soil zone
    c.rect(map_x + map_w * 0.5, map_y + map_h * 0.5, map_w * 0.5, map_h * 0.5, fill=True, stroke=False)

    c.setFillColor(colors.HexColor("#991B1B")) # Red soil zone
    c.rect(map_x, map_y, map_w * 0.45, map_h * 0.55, fill=True, stroke=False)

    c.setFillColor(colors.HexColor("#D97706")) # Sandy zone
    c.rect(map_x + map_w * 0.6, map_y, map_w * 0.4, map_h * 0.45, fill=True, stroke=False)

    # River / Alluvial corridor
    c.setFillColor(colors.HexColor("#CA8A04"))
    p_alluvial = c.beginPath()
    p_alluvial.moveTo(map_x + map_w * 0.35, map_y)
    p_alluvial.curveTo(map_x + map_w * 0.4, map_y + map_h * 0.4, map_x + map_w * 0.3, map_y + map_h * 0.7, map_x + map_w * 0.45, map_y + map_h)
    p_alluvial.lineTo(map_x + map_w * 0.55, map_y + map_h)
    p_alluvial.curveTo(map_x + map_w * 0.4, map_y + map_h * 0.7, map_x + map_w * 0.5, map_y + map_h * 0.4, map_x + map_w * 0.45, map_y)
    p_alluvial.close()
    c.drawPath(p_alluvial, fill=True, stroke=False)

    # Vector River Blue Line
    c.setStrokeColor(colors.HexColor("#0284C7"))
    c.setLineWidth(4)
    p_river = c.beginPath()
    p_river.moveTo(map_x + map_w * 0.4, map_y)
    p_river.curveTo(map_x + map_w * 0.45, map_y + map_h * 0.4, map_x + map_w * 0.35, map_y + map_h * 0.7, map_x + map_w * 0.5, map_y + map_h)
    c.drawPath(p_river, fill=False, stroke=True)

    # Vector Road Exclusion Network (White/Yellow Lines)
    c.setStrokeColor(colors.HexColor("#F8FAFC"))
    c.setLineWidth(2.5)
    c.line(map_x, map_y + map_h * 0.6, map_x + map_w, map_y + map_h * 0.3)
    c.line(map_x + map_w * 0.25, map_y, map_x + map_w * 0.75, map_y + map_h)

    # Neatline Border
    c.setStrokeColor(colors.HexColor("#38BDF8"))
    c.setLineWidth(2)
    c.rect(map_x, map_y, map_w, map_h, fill=False, stroke=True)

    # Right Sidebar Furniture
    side_x = map_x + map_w + 15
    side_w = page_w - side_x - 20
    c.setFillColor(colors.HexColor("#1E293B"))
    c.rect(side_x, map_y, side_w, map_h, fill=True, stroke=False)
    c.setStrokeColor(colors.HexColor("#334155"))
    c.rect(side_x, map_y, side_w, map_h, fill=False, stroke=True)

    c.setFillColor(colors.HexColor("#FFFFFF"))
    c.setFont("Helvetica-Bold", 10)
    c.drawString(side_x + 10, map_y + map_h - 20, "MAP FURNITURE & METADATA")

    c.setFillColor(colors.HexColor("#94A3B8"))
    c.setFont("Helvetica", 7.5)
    c.drawString(side_x + 10, map_y + map_h - 40, "Projection: Transverse Mercator")
    c.drawString(side_x + 10, map_y + map_h - 55, "Datum: WGS 1984 / UTM Zone 43N")
    c.drawString(side_x + 10, map_y + map_h - 70, "Origin: (500000.0 E, 2100000.0 N)")
    c.drawString(side_x + 10, map_y + map_h - 85, "Grid Interval: 10,000 meters")
    c.drawString(side_x + 10, map_y + map_h - 100, "Neatline Bounds: [30, 50, 550, 425]")

    c.showPage()

    # Page 2: Inset Sub-Region
    c.setFillColor(colors.HexColor("#0F172A"))
    c.rect(0, 0, page_w, page_h, fill=True, stroke=False)
    c.setFillColor(colors.HexColor("#FFFFFF"))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(35, page_h - 35, "INSET VIEWPORT: RIVER BASIN AGRICULTURAL ZONE")

    c.setFillColor(colors.HexColor("#1E293B"))
    c.rect(map_x, map_y, map_w, map_h, fill=True, stroke=False)
    c.setFillColor(colors.HexColor("#EAB308"))
    c.circle(map_x + map_w / 2, map_y + map_h / 2, 120, fill=True, stroke=False)

    c.showPage()
    c.save()

    print(f"Sample GeoPDF generated at: {os.path.abspath(output_path)}")
    return os.path.abspath(output_path)


if __name__ == "__main__":
    create_sample_geotiff()
    create_sample_geopdf()

"""
File format and signature validator for GeoTIFF and GeoPDF files.
Checks magic bytes, extensions, and file integrity.
"""

import os
from typing import Dict, Any, Tuple


TIFF_LITTLE_ENDIAN = b'II\x2a\x00'
TIFF_BIG_ENDIAN = b'MM\x00\x2a'
BIGTIFF_LITTLE = b'II\x2b\x00'
BIGTIFF_BIG = b'MM\x00\x2b'
PDF_MAGIC = b'%PDF-'


def detect_file_format(file_path: str) -> Dict[str, Any]:
    """
    Detects and validates whether the file is a GeoTIFF, GeoPDF, or invalid format.
    Returns metadata dict containing format, valid status, mime_type, and size.
    """
    if not os.path.exists(file_path):
        return {
            "valid": False,
            "format": "UNKNOWN",
            "error": f"File does not exist: {file_path}",
            "file_size": 0
        }

    file_size = os.path.getsize(file_path)
    if file_size < 16:
        return {
            "valid": False,
            "format": "INVALID",
            "error": "File size is too small to be a valid GeoTIFF or PDF",
            "file_size": file_size
        }

    with open(file_path, 'rb') as f:
        header = f.read(16)

    ext = os.path.splitext(file_path)[1].lower()

    # Check for TIFF / BigTIFF signatures
    if (header.startswith(TIFF_LITTLE_ENDIAN) or
        header.startswith(TIFF_BIG_ENDIAN) or
        header.startswith(BIGTIFF_LITTLE) or
        header.startswith(BIGTIFF_BIG) or
        ext in ['.tif', '.tiff']):
        
        # Validate TIFF readability with rasterio
        try:
            import rasterio
            with rasterio.open(file_path) as src:
                crs = src.crs.to_string() if src.crs else None
                bounds = list(src.bounds)
                count = src.count
                width = src.width
                height = src.height
                transform = list(src.transform)
                dtype = str(src.dtypes[0])

            return {
                "valid": True,
                "format": "GEOTIFF",
                "ext": ext,
                "file_size": file_size,
                "channels": count,
                "width": width,
                "height": height,
                "crs": crs,
                "bounds": bounds,
                "transform": transform,
                "dtype": dtype,
                "is_georeferenced": crs is not None
            }
        except Exception as e:
            return {
                "valid": False,
                "format": "TIFF_CORRUPT",
                "error": f"Failed to open TIFF raster: {str(e)}",
                "file_size": file_size
            }

    # Check for PDF signature
    if header.startswith(PDF_MAGIC) or ext == '.pdf':
        try:
            import pymupdf
            doc = pymupdf.open(file_path)
            page_count = len(doc)
            doc.close()

            return {
                "valid": True,
                "format": "PDF",
                "ext": ext,
                "file_size": file_size,
                "page_count": page_count,
                "is_georeferenced": False # Will be verified by inspect_geopdf
            }
        except Exception as e:
            return {
                "valid": False,
                "format": "PDF_CORRUPT",
                "error": f"Failed to open PDF document: {str(e)}",
                "file_size": file_size
            }

    return {
        "valid": False,
        "format": "UNSUPPORTED",
        "error": f"Unsupported file type: {ext}. Only GeoTIFF (.tif, .tiff) and GeoPDF (.pdf) are supported.",
        "file_size": file_size
    }

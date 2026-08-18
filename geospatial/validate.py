"""
Raster and Geospatial Metadata Integrity Validator.
Verifies raster shapes, coordinate dimensions, affine transformations, and nodata integrity.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import rasterio
from affine import Affine
from .crs import get_crs


def validate_raster_dataset(
    raster: np.ndarray,
    transform: Optional[Affine] = None,
    crs: Optional[str] = None,
    nodata: Optional[float] = None
) -> Dict[str, Any]:
    """
    Validates raster dataset array and geospatial metadata.
    
    Returns:
        Dict:
            - 'valid': bool
            - 'errors': list of error strings
            - 'warnings': list of warning strings
            - 'channels': int
            - 'height': int
            - 'width': int
            - 'dtype': str
            - 'finite_pct': float
            - 'has_valid_transform': bool
            - 'has_valid_crs': bool
    """
    errors: List[str] = []
    warnings: List[str] = []

    if raster is None or not isinstance(raster, np.ndarray):
        return {
            "valid": False,
            "errors": ["Raster data is None or not a numpy ndarray."],
            "warnings": []
        }

    # Standardize to (C, H, W)
    if raster.ndim == 2:
        c, h, w = 1, raster.shape[0], raster.shape[1]
    elif raster.ndim == 3:
        if raster.shape[0] in [1, 3, 4, 8, 12, 16] and raster.shape[0] < min(raster.shape[1], raster.shape[2]):
            c, h, w = raster.shape[0], raster.shape[1], raster.shape[2]
        else:
            h, w, c = raster.shape[0], raster.shape[1], raster.shape[2]
    else:
        errors.append(f"Invalid raster array dimension: {raster.ndim}D (expected 2D or 3D).")
        return {"valid": False, "errors": errors, "warnings": warnings}

    if h < 16 or w < 16:
        errors.append(f"Raster dimensions too small ({w}x{h}). Minimum required is 16x16.")

    # Check finite values
    finite_mask = np.isfinite(raster)
    finite_pct = float(np.mean(finite_mask) * 100.0)
    if finite_pct < 10.0:
        errors.append(f"Raster contains excessive non-finite values ({finite_pct:.1f}% valid).")
    elif finite_pct < 95.0:
        warnings.append(f"Raster contains {(100.0 - finite_pct):.1f}% NaN/Inf values.")

    # Validate Transform
    has_valid_transform = False
    if transform is not None:
        try:
            det = transform.determinant
            if abs(det) > 1e-12:
                has_valid_transform = True
            else:
                warnings.append("Affine transform determinant is close to zero.")
        except Exception as e:
            warnings.append(f"Could not compute transform determinant: {e}")
    else:
        warnings.append("No Affine transform supplied; using default unit grid.")

    # Validate CRS
    has_valid_crs = False
    if crs:
        try:
            c_obj = get_crs(crs)
            has_valid_crs = True
        except Exception as e:
            warnings.append(f"Could not parse CRS '{crs}': {e}")
    else:
        warnings.append("No CRS supplied; defaulting to EPSG:32643.")

    is_valid = len(errors) == 0

    return {
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "channels": c,
        "height": h,
        "width": w,
        "dtype": str(raster.dtype),
        "finite_pct": finite_pct,
        "has_valid_transform": has_valid_transform,
        "has_valid_crs": has_valid_crs
    }

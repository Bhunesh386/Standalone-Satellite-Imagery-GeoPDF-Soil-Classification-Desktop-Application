"""
Geospatial Coordinate Reference System (CRS) Utilities.
Handles pyproj transformations, auto-UTM zone detection, and bounds reprojection.
"""

from typing import Tuple, Optional, Union
import pyproj
from pyproj import CRS, Transformer
import numpy as np


def get_crs(crs_input: Union[str, int, CRS]) -> CRS:
    """Creates a pyproj.CRS object from EPSG code, WKT, or string."""
    if isinstance(crs_input, CRS):
        return crs_input
    if isinstance(crs_input, int):
        return CRS.from_epsg(crs_input)
    if isinstance(crs_input, str):
        if crs_input.isdigit():
            return CRS.from_epsg(int(crs_input))
        return CRS.from_user_input(crs_input)
    return CRS.from_epsg(32643) # Fallback to UTM 43N


def get_utm_epsg_for_latlon(lat: float, lon: float) -> int:
    """
    Computes the appropriate UTM EPSG code for a given WGS84 latitude and longitude.
    """
    # Clamp longitude to [-180, 180]
    lon = (lon + 180) % 360 - 180
    zone_number = int((lon + 180) / 6) + 1

    # Special UTM zones for Norway / Svalbard if needed
    if 56.0 <= lat < 64.0 and 3.0 <= lon < 12.0:
        zone_number = 32

    if lat >= 0:
        epsg = 32600 + zone_number
    else:
        epsg = 32700 + zone_number

    return epsg


def transform_bounds(
    bounds: Tuple[float, float, float, float],
    src_crs: Union[str, CRS],
    dst_crs: Union[str, CRS]
) -> Tuple[float, float, float, float]:
    """
    Reprojects bounding box coordinates (min_x, min_y, max_x, max_y)
    from src_crs to dst_crs.
    """
    src = get_crs(src_crs)
    dst = get_crs(dst_crs)

    if src == dst:
        return bounds

    transformer = Transformer.from_crs(src, dst, always_xy=True)
    min_x, min_y, max_x, max_y = bounds

    # Transform 4 corner points
    xs = [min_x, max_x, min_x, max_x]
    ys = [min_y, min_y, max_y, max_y]
    tx, ty = transformer.transform(xs, ys)

    return (min(tx), min(ty), max(tx), max(ty))


def transform_points(
    xs: np.ndarray,
    ys: np.ndarray,
    src_crs: Union[str, CRS],
    dst_crs: Union[str, CRS]
) -> Tuple[np.ndarray, np.ndarray]:
    """Reprojects arrays of x and y coordinates."""
    src = get_crs(src_crs)
    dst = get_crs(dst_crs)
    if src == dst:
        return xs, ys

    transformer = Transformer.from_crs(src, dst, always_xy=True)
    tx, ty = transformer.transform(xs, ys)
    return np.array(tx), np.array(ty)

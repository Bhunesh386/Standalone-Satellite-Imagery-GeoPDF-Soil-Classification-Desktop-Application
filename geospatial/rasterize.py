"""
Vector Layer Rasterizer.
Converts vector shapes, polylines, exclusion masks, and reference labels
into categorical raster masks matching the model grid.
"""

from typing import List, Union, Optional
import numpy as np
import geopandas as gpd
from shapely.geometry import base, MultiPolygon, Polygon
import rasterio
from rasterio.features import rasterize
from affine import Affine


def rasterize_vector_geometries(
    geometries: Union[gpd.GeoDataFrame, List[base.BaseGeometry]],
    out_shape: tuple[int, int],
    transform: Affine,
    burn_value: int = 1,
    fill_value: int = 0,
    all_touched: bool = True
) -> np.ndarray:
    """
    Rasterizes vector geometries onto an existing raster grid.
    
    Args:
        geometries: GeoDataFrame or list of Shapely geometries.
        out_shape: (height, width) of output raster.
        transform: Affine transformation matrix.
        burn_value: Value to write into pixels intersecting geometries.
        fill_value: Background pixel value.
        all_touched: If True, all pixels touched by geometries are burned.
        
    Returns:
        mask: uint8 numpy array of shape (height, width).
    """
    if isinstance(geometries, gpd.GeoDataFrame):
        valid_geoms = [geom for geom in geometries.geometry if geom is not None and not geom.is_empty]
    elif isinstance(geometries, (list, tuple)):
        valid_geoms = [geom for geom in geometries if geom is not None and not geom.is_empty]
    else:
        valid_geoms = []

    if not valid_geoms:
        return np.full(out_shape, fill_value, dtype=np.uint8)

    shapes = [(geom, burn_value) for geom in valid_geoms]

    try:
        mask = rasterize(
            shapes=shapes,
            out_shape=out_shape,
            transform=transform,
            fill=fill_value,
            all_touched=all_touched,
            dtype=np.uint8
        )
        return mask
    except Exception as e:
        print(f"Warning: Rasterization error: {e}. Returning blank mask.")
        return np.full(out_shape, fill_value, dtype=np.uint8)

"""
Sliding-Window Tiling Engine for Large Geospatial Rasters.
Extracts overlapping tiles with configurable window size, overlap padding,
and spatial index tracking.
"""

from typing import List, Tuple, Dict, Any, Generator
import numpy as np


class TileWindow:
    """Represents a single tile's position, bounding coordinates, and padding."""

    def __init__(
        self,
        tile_id: int,
        ymin: int,
        ymax: int,
        xmin: int,
        xmax: int,
        pad_top: int = 0,
        pad_bottom: int = 0,
        pad_left: int = 0,
        pad_right: int = 0
    ):
        self.tile_id = tile_id
        self.ymin = ymin
        self.ymax = ymax
        self.xmin = xmin
        self.xmax = xmax
        self.pad_top = pad_top
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.pad_right = pad_right

    @property
    def slice_y(self) -> slice:
        return slice(self.ymin, self.ymax)

    @property
    def slice_x(self) -> slice:
        return slice(self.xmin, self.xmax)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "ymin": self.ymin,
            "ymax": self.ymax,
            "xmin": self.xmin,
            "xmax": self.xmax,
            "pad": [self.pad_top, self.pad_bottom, self.pad_left, self.pad_right]
        }


def generate_tiles(
    raster: np.ndarray,
    tile_size: int = 512,
    overlap: int = 32
) -> List[Tuple[np.ndarray, TileWindow]]:
    """
    Splits large (C, H, W) raster into overlapping tiles.
    
    Args:
        raster: (C, H, W) input array.
        tile_size: width and height of each tile.
        overlap: overlapping boundary margin in pixels.
        
    Returns:
        List of (tile_array, TileWindow).
    """
    c, h, w = raster.shape
    stride = max(tile_size - 2 * overlap, 1)

    # Compute step positions
    y_steps = list(range(0, h, stride))
    x_steps = list(range(0, w, stride))

    tiles: List[Tuple[np.ndarray, TileWindow]] = []
    tile_id = 0

    for y in y_steps:
        for x in x_steps:
            # Crop bounds
            ymin = max(0, y - overlap)
            ymax = min(h, y + stride + overlap)
            xmin = max(0, x - overlap)
            xmax = min(w, x + stride + overlap)

            tile_data = raster[:, ymin:ymax, xmin:xmax]

            # Calculate padding if tile is smaller than tile_size
            curr_h = ymax - ymin
            curr_w = xmax - xmin
            pad_h = max(0, tile_size - curr_h)
            pad_w = max(0, tile_size - curr_w)

            pad_top = 0
            pad_bottom = pad_h
            pad_left = 0
            pad_right = pad_w

            if pad_h > 0 or pad_w > 0:
                tile_data = np.pad(
                    tile_data,
                    ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='reflect'
                )

            window = TileWindow(
                tile_id=tile_id,
                ymin=ymin,
                ymax=ymax,
                xmin=xmin,
                xmax=xmax,
                pad_top=pad_top,
                pad_bottom=pad_bottom,
                pad_left=pad_left,
                pad_right=pad_right
            )

            tiles.append((tile_data, window))
            tile_id += 1

    return tiles

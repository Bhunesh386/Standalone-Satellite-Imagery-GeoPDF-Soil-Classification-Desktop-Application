"""
Overlap Stitching and Blending Engine.
Reconstructs full-scene probability and confidence grids from overlapping tile predictions
using 2D Cosine / Gaussian / Linear spatial blending filters to eliminate boundary artifacts.
"""

from typing import List, Tuple, Literal
import numpy as np
from scipy.ndimage import gaussian_filter
from .tile import TileWindow


def create_blend_weight_mask(
    tile_size: int,
    method: Literal["cosine", "gaussian", "linear"] = "cosine",
    sigma_fraction: float = 0.25
) -> np.ndarray:
    """
    Creates a 2D weighting window (H, W) that smoothly decays to zero at borders.
    """
    if method == "cosine":
        # 2D Hann window / raised cosine
        w1d = np.sin(np.linspace(0, np.pi, tile_size)) ** 2
        w2d = np.outer(w1d, w1d)
    elif method == "gaussian":
        # 2D Gaussian bell
        center = (tile_size - 1) / 2.0
        y, x = np.ogrid[:tile_size, :tile_size]
        sigma = tile_size * sigma_fraction
        w2d = np.exp(-((x - center)**2 + (y - center)**2) / (2.0 * sigma**2))
    elif method == "linear":
        # Linear tent pyramid
        w1d = np.minimum(np.arange(tile_size), np.arange(tile_size)[::-1])
        w1d = w1d / max(np.max(w1d), 1.0)
        w2d = np.outer(w1d, w1d)
    else:
        w2d = np.ones((tile_size, tile_size), dtype=np.float32)

    # Ensure minimum non-zero weight to avoid division by zero
    w2d = np.maximum(w2d, 1e-4).astype(np.float32)
    return w2d


class TileStitcher:
    """Accumulates and stitches overlapping tile predictions into full resolution rasters."""

    def __init__(
        self,
        full_shape: Tuple[int, int],
        num_classes: int = 6,
        tile_size: int = 512,
        blend_method: str = "cosine"
    ):
        self.height, self.width = full_shape
        self.num_classes = num_classes
        self.tile_size = tile_size
        self.blend_weight = create_blend_weight_mask(tile_size, method=blend_method)

        # Accumulators
        self.prob_accumulator = np.zeros((num_classes, self.height, self.width), dtype=np.float32)
        self.conf_accumulator = np.zeros((self.height, self.width), dtype=np.float32)
        self.weight_accumulator = np.zeros((self.height, self.width), dtype=np.float32)

    def add_tile_prediction(
        self,
        probs_tile: np.ndarray,
        conf_tile: np.ndarray,
        window: TileWindow
    ):
        """
        Adds a single tile's prediction probabilities and confidence to the accumulator.
        
        Args:
            probs_tile: (num_classes, tile_size, tile_size)
            conf_tile: (1, tile_size, tile_size) or (tile_size, tile_size)
            window: TileWindow metadata
        """
        if conf_tile.ndim == 3:
            conf_tile = conf_tile[0]

        # Remove padding to get the actual tile region
        valid_h = self.tile_size - (window.pad_top + window.pad_bottom)
        valid_w = self.tile_size - (window.pad_left + window.pad_right)

        probs_valid = probs_tile[:, window.pad_top:window.pad_top + valid_h, window.pad_left:window.pad_left + valid_w]
        conf_valid = conf_tile[window.pad_top:window.pad_top + valid_h, window.pad_left:window.pad_left + valid_w]
        weight_valid = self.blend_weight[window.pad_top:window.pad_top + valid_h, window.pad_left:window.pad_left + valid_w]

        # Accumulate with blending weight
        weighted_probs = probs_valid * weight_valid[np.newaxis, :, :]
        weighted_conf = conf_valid * weight_valid

        self.prob_accumulator[:, window.slice_y, window.slice_x] += weighted_probs
        self.conf_accumulator[window.slice_y, window.slice_x] += weighted_conf
        self.weight_accumulator[window.slice_y, window.slice_x] += weight_valid

    def finalize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalizes accumulators and computes final classified map and confidence map.
        
        Returns:
            class_map: uint8 array (height, width) with class indices [0 .. num_classes-1]
            confidence_map: float32 array (height, width) in [0.0, 1.0]
            prob_map: float32 array (num_classes, height, width)
        """
        safe_weight = np.maximum(self.weight_accumulator, 1e-6)

        norm_probs = self.prob_accumulator / safe_weight[np.newaxis, :, :]
        norm_conf = self.conf_accumulator / safe_weight

        # Compute argmax class map
        class_map = np.argmax(norm_probs, axis=0).astype(np.uint8)

        # Clip confidence to [0.0, 1.0]
        confidence_map = np.clip(norm_conf, 0.0, 1.0).astype(np.float32)

        return class_map, confidence_map, norm_probs

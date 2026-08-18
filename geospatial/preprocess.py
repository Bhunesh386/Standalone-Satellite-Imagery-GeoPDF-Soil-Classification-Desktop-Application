"""
Geospatial Preprocessing Engine.
Performs percentile-based channel normalization, radiometric dynamic range scaling,
synthetic NIR channel generation if RGB, and NoData masking.
"""

from typing import Tuple, Optional
import numpy as np


def preprocess_raster_for_model(
    raster: np.ndarray,
    nodata_value: Optional[float] = None,
    target_channels: int = 4,
    normalize_method: str = "percentile"
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Standardizes input raster into a normalized (C, H, W) float32 tensor
    ready for U-Net inference.
    
    Args:
        raster: Input numpy array (C, H, W) or (H, W, C).
        nodata_value: Specific NoData value to mask (e.g. 0, -9999, 255).
        target_channels: Target number of channels (4 for RGB+NIR, 3 for RGB).
        normalize_method: 'percentile' (2-98%), 'minmax', or 'zscore'.
        
    Returns:
        norm_tensor: (C, H, W) float32 in range [0.0, 1.0].
        nodata_mask: (H, W) boolean mask where True indicates invalid / nodata pixels.
        stats: dict containing channel stats (mins, maxs, means, stds).
    """
    # Standardize to (C, H, W)
    if raster.ndim == 2:
        raster = np.expand_dims(raster, axis=0)
    elif raster.ndim == 3 and raster.shape[2] in [1, 3, 4, 8, 12, 16] and raster.shape[0] > raster.shape[2]:
        raster = np.transpose(raster, (2, 0, 1))

    c, h, w = raster.shape
    raster_f32 = raster.astype(np.float32)

    # Compute NoData mask
    nodata_mask = np.zeros((h, w), dtype=bool)
    if nodata_value is not None:
        nodata_mask |= np.any(np.isclose(raster_f32, nodata_value), axis=0)

    # Also mask NaNs, Infs, and pure zeros across all channels
    nodata_mask |= ~np.all(np.isfinite(raster_f32), axis=0)
    all_zeros = np.all(raster_f32 == 0, axis=0)
    if np.sum(all_zeros) < (h * w * 0.95): # only mask if not entire image
        nodata_mask |= all_zeros

    # Normalize each channel independently
    normalized_channels = []
    stats_list = []

    for i in range(c):
        band = raster_f32[i]
        valid_pixels = band[~nodata_mask]

        if len(valid_pixels) > 0:
            if normalize_method == "percentile":
                p2, p98 = np.percentile(valid_pixels, [2.0, 98.0])
                if p98 > p2:
                    norm_band = np.clip((band - p2) / (p98 - p2), 0.0, 1.0)
                else:
                    norm_band = np.clip(band / 255.0, 0.0, 1.0)
            elif normalize_method == "minmax":
                bmin, bmax = np.min(valid_pixels), np.max(valid_pixels)
                if bmax > bmin:
                    norm_band = np.clip((band - bmin) / (bmax - bmin), 0.0, 1.0)
                else:
                    norm_band = np.clip(band / 255.0, 0.0, 1.0)
            elif normalize_method == "zscore":
                mean, std = np.mean(valid_pixels), np.std(valid_pixels)
                std = max(std, 1e-6)
                norm_band = (band - mean) / std
            else:
                norm_band = np.clip(band / 255.0, 0.0, 1.0)

            stats_list.append({
                "band": i,
                "min": float(np.min(valid_pixels)),
                "max": float(np.max(valid_pixels)),
                "mean": float(np.mean(valid_pixels)),
                "std": float(np.std(valid_pixels))
            })
        else:
            norm_band = np.zeros_like(band)
            stats_list.append({"band": i, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0})

        # Zero out nodata pixels
        norm_band[nodata_mask] = 0.0
        normalized_channels.append(norm_band)

    normalized_raster = np.stack(normalized_channels, axis=0) # (C, H, W)

    # Adapt channel count to target_channels
    if normalized_raster.shape[0] < target_channels:
        if normalized_raster.shape[0] == 3 and target_channels == 4:
            # Generate synthetic NIR from (Red + Green) * 0.7 + 0.3
            r = normalized_raster[0]
            g = normalized_raster[1]
            synthetic_nir = np.clip((r + g) * 0.6 + 0.1, 0.0, 1.0)
            normalized_raster = np.concatenate([normalized_raster, np.expand_dims(synthetic_nir, 0)], axis=0)
        elif normalized_raster.shape[0] == 1:
            repeats = [target_channels] + [1] * (normalized_raster.ndim - 1)
            normalized_raster = np.tile(normalized_raster, repeats)
    elif normalized_raster.shape[0] > target_channels:
        normalized_raster = normalized_raster[:target_channels]

    return normalized_raster.astype(np.float32), nodata_mask, {"channels": stats_list}
